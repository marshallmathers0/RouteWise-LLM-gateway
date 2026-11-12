"""Per-provider circuit breaker.

    CLOSED --(N consecutive failures)--> OPEN --(cooldown elapsed)--> HALF_OPEN
      ^                                   ^                              |
      |                                   +-------(probe fails)----------+
      +---------------------(probe succeeds)-----------------------------+

Why: when a provider is down, every request would otherwise wait for timeouts and
retries before falling back. An open breaker skips the dead provider instantly. The
half-open state lets exactly one "probe" request through to test recovery.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable

from routewise.reliability.state import StateStore


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        store: StateStore,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock

    @staticmethod
    def _key(name: str) -> str:
        return f"rw:cb:{name}"

    async def state(self, name: str) -> BreakerState:
        data = await self._store.hgetall(self._key(name))
        state = BreakerState(data.get("state", BreakerState.CLOSED.value))
        if state is BreakerState.OPEN:
            opened_at = float(data.get("opened_at", "0"))
            if self._clock() - opened_at >= self.cooldown_s:
                return BreakerState.HALF_OPEN
        return state

    async def allow_request(self, name: str) -> bool:
        state = await self.state(name)
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        # HALF_OPEN: only one probe at a time. The lock expires after one cooldown so a
        # probe that hangs forever can't wedge the breaker.
        await self._store.hset(self._key(name), {"state": BreakerState.HALF_OPEN.value})
        return await self._store.set_nx(self._key(name) + ":probe", "1", self.cooldown_s)

    async def record_success(self, name: str) -> None:
        await self._store.hset(
            self._key(name), {"state": BreakerState.CLOSED.value, "failures": 0, "opened_at": 0}
        )
        await self._store.delete(self._key(name) + ":probe")

    async def record_failure(self, name: str) -> None:
        key = self._key(name)
        state = await self.state(name)
        if state is BreakerState.HALF_OPEN:
            await self._open(name)
            return
        failures = await self._store.hincrby(key, "failures", 1)
        if failures >= self.failure_threshold:
            await self._open(name)

    async def _open(self, name: str) -> None:
        await self._store.hset(
            self._key(name),
            {"state": BreakerState.OPEN.value, "opened_at": self._clock(), "failures": 0},
        )
        await self._store.delete(self._key(name) + ":probe")
