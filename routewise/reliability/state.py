"""Small key-value state store used by the circuit breaker.

Shared state lives in Redis so several gateway workers agree on whether a provider is
down. If Redis is unavailable we degrade to per-process memory instead of failing
requests: a slightly less coordinated breaker is far better than no gateway.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)


class StateStore(Protocol):
    async def hgetall(self, key: str) -> dict[str, str]: ...
    async def hset(self, key: str, mapping: dict[str, Any]) -> None: ...
    async def hincrby(self, key: str, field: str, amount: int = 1) -> int: ...
    async def set_nx(self, key: str, value: str, ttl_s: float) -> bool: ...
    async def delete(self, key: str) -> None: ...


class MemoryStateStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._keys: dict[str, float] = {}  # key -> expiry time
        self._clock = clock

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    async def hset(self, key: str, mapping: dict[str, Any]) -> None:
        self._hashes.setdefault(key, {}).update({k: str(v) for k, v in mapping.items()})

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        h = self._hashes.setdefault(key, {})
        h[field] = str(int(h.get(field, "0")) + amount)
        return int(h[field])

    async def set_nx(self, key: str, value: str, ttl_s: float) -> bool:
        now = self._clock()
        expiry = self._keys.get(key)
        if expiry is not None and expiry > now:
            return False
        self._keys[key] = now + ttl_s
        return True

    async def delete(self, key: str) -> None:
        self._hashes.pop(key, None)
        self._keys.pop(key, None)


class RedisStateStore:
    def __init__(self, redis: Redis) -> None:
        self._r = redis

    async def hgetall(self, key: str) -> dict[str, str]:
        raw = await self._r.hgetall(key)
        return {_s(k): _s(v) for k, v in raw.items()}

    async def hset(self, key: str, mapping: dict[str, Any]) -> None:
        await self._r.hset(key, mapping={k: str(v) for k, v in mapping.items()})

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        return int(await self._r.hincrby(key, field, amount))

    async def set_nx(self, key: str, value: str, ttl_s: float) -> bool:
        return bool(await self._r.set(key, value, nx=True, px=int(ttl_s * 1000)))

    async def delete(self, key: str) -> None:
        await self._r.delete(key)


class FallbackStateStore:
    """Try Redis; on any Redis error, use the in-memory store for that call."""

    def __init__(self, primary: StateStore | None, fallback: StateStore) -> None:
        self._primary = primary
        self._fallback = fallback

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._primary is not None:
            try:
                return await getattr(self._primary, name)(*args, **kwargs)
            except (RedisError, OSError) as e:
                log.warning("redis state store failed (%s); using in-memory state", e)
        return await getattr(self._fallback, name)(*args, **kwargs)

    async def hgetall(self, key: str) -> dict[str, str]:
        return await self._call("hgetall", key)

    async def hset(self, key: str, mapping: dict[str, Any]) -> None:
        await self._call("hset", key, mapping)

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        return await self._call("hincrby", key, field, amount)

    async def set_nx(self, key: str, value: str, ttl_s: float) -> bool:
        return await self._call("set_nx", key, value, ttl_s)

    async def delete(self, key: str) -> None:
        await self._call("delete", key)


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
