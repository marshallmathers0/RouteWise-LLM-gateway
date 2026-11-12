"""Token-bucket rate limiter, one bucket per API key.

A bucket holds up to `capacity` tokens and refills at `refill_per_second`. Each request
spends one token; with no token left the request is rejected with 429. This allows short
bursts (up to capacity) while capping the sustained rate (refill rate).

In Redis the read-refill-spend sequence runs as one Lua script so concurrent requests
from many gateway workers can't both spend the last token (no race condition).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

# KEYS[1] = bucket key; ARGV = capacity, refill_per_second, now (seconds)
_TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now
end
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
-- Expire idle buckets: after capacity/rate seconds a bucket is full again anyway.
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) + 1)
return {allowed, tostring(tokens)}
"""


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: float
    retry_after_s: float


class TokenBucketRateLimiter:
    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        redis: Redis | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.capacity = capacity
        self.rate = refill_per_second
        self._redis = redis
        self._clock = clock
        self._memory: dict[str, tuple[float, float]] = {}  # key -> (tokens, ts)
        self._script = redis.register_script(_TOKEN_BUCKET_LUA) if redis is not None else None

    def _result(self, allowed: bool, tokens: float) -> RateLimitResult:
        retry_after = 0.0 if allowed else (1 - tokens) / self.rate
        return RateLimitResult(allowed, tokens, retry_after)

    async def check(self, key: str) -> RateLimitResult:
        now = self._clock()
        if self._script is not None:
            try:
                allowed, tokens = await self._script(keys=[f"rw:rl:{key}"], args=[self.capacity, self.rate, now])
                return self._result(bool(int(allowed)), float(tokens))
            except (RedisError, OSError) as e:
                log.warning("redis rate limiter failed (%s); using in-memory bucket", e)
        return self._check_memory(key, now)

    def _check_memory(self, key: str, now: float) -> RateLimitResult:
        tokens, ts = self._memory.get(key, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + max(0.0, now - ts) * self.rate)
        allowed = tokens >= 1
        if allowed:
            tokens -= 1
        self._memory[key] = (tokens, now)
        return self._result(allowed, tokens)


def retry_after_header(result: RateLimitResult) -> str:
    return str(max(1, math.ceil(result.retry_after_s)))
