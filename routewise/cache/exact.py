"""Exact-match response cache in Redis.

The key is a SHA-256 of everything that can change the answer: the full (normalized)
message list INCLUDING system prompts, the requested model, and generation params.
Two requests that differ only in the system prompt therefore never collide.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from routewise.api.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

KEY_PREFIX = "rw:exact:v2:"  # v2: max_tokens left out of the key  # bump the version to invalidate every entry after a format change
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    # Collapse whitespace only. We do NOT lowercase: case can matter (code, acronyms),
    # and paraphrase-level matching is the semantic cache's job, not this one's.
    return _WS.sub(" ", text).strip()


def exact_cache_key(req: ChatCompletionRequest) -> str:
    payload = {
        "messages": [[m.role, _normalize(m.text())] for m in req.messages],
        # The *requested* model ("auto", "cheap", or an explicit name). For "auto" the
        # router is deterministic per query, so the same query reuses the same answer.
        "model": req.model,
        "temperature": req.temperature,
        "top_p": req.top_p,
        # max_tokens is deliberately NOT in the key: truncated answers are never cached, so a
        # cached answer is complete; the pipeline only checks it fits the caller's max_tokens.
        "stop": req.stop,
        "response_format": req.response_format,  # JSON mode changes the answer's shape
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return KEY_PREFIX + hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CachedAnswer:
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    route: str | None = None


class ExactCache:
    def __init__(self, redis: Redis | None, ttl_s: int) -> None:
        self._redis = redis
        self.ttl_s = ttl_s

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def get(self, key: str) -> CachedAnswer | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
        except (RedisError, OSError) as e:
            log.warning("exact cache get failed (%s); treating as miss", e)
            return None
        if raw is None:
            return None
        try:
            return CachedAnswer(**json.loads(raw))
        except (ValueError, TypeError):
            return None

    async def set(self, key: str, value: CachedAnswer) -> None:
        if self._redis is None:
            return
        try:
            # TTL bounds staleness: even "timeless" answers get refreshed eventually.
            await self._redis.set(key, json.dumps(asdict(value)), ex=self.ttl_s)
        except (RedisError, OSError) as e:
            log.warning("exact cache set failed (%s)", e)
