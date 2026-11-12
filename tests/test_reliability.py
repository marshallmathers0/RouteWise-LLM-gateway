"""Retry/backoff, circuit breaker transitions, fallback chain, token bucket."""

from __future__ import annotations

import random

import fakeredis
import pytest

from routewise.core.config import ReliabilityConfig, TierTarget
from routewise.providers.base import LLMRequest, ProviderError
from routewise.reliability.circuit_breaker import BreakerState, CircuitBreaker
from routewise.reliability.fallback import AllProvidersFailed, call_with_fallback
from routewise.reliability.rate_limiter import TokenBucketRateLimiter
from routewise.reliability.retry import backoff_delay, retry_async
from routewise.reliability.state import MemoryStateStore, RedisStateStore
from tests.conftest import FakeLLMClient


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def _no_sleep(_: float) -> None:
    pass


# ---- retry ------------------------------------------------------------------------------
def test_backoff_is_bounded_and_grows():
    rng = random.Random(0)
    for attempt in range(10):
        d = backoff_delay(attempt, 0.5, 8.0, rng)
        assert 0 <= d <= min(8.0, 0.5 * 2**attempt)


async def test_retry_succeeds_after_transient_errors():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("429", provider="p", status_code=429, retryable=True)
        return "ok"

    assert await retry_async(fn, max_retries=2, base_s=0, max_s=0, sleep=_no_sleep) == "ok"
    assert calls == 3


async def test_non_retryable_error_is_not_retried():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ProviderError("400", provider="p", status_code=400, retryable=False)

    with pytest.raises(ProviderError):
        await retry_async(fn, max_retries=5, base_s=0, max_s=0, sleep=_no_sleep)
    assert calls == 1


async def test_short_retry_after_hint_is_respected():
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    async def fn():
        if not slept:
            raise ProviderError("429", provider="p", retryable=True, retry_after_s=3)
        return "ok"

    await retry_async(fn, max_retries=1, base_s=0.1, max_s=5.0, sleep=sleep)
    assert slept == [3]


async def test_long_retry_after_gives_up_immediately():
    # Waiting 100 s would only add latency; fail fast so the fallback chain moves on.
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    async def fn():
        raise ProviderError("429", provider="p", retryable=True, retry_after_s=100)

    with pytest.raises(ProviderError):
        await retry_async(fn, max_retries=3, base_s=0.1, max_s=5.0, sleep=sleep)
    assert slept == []


# ---- circuit breaker ------------------------------------------------------------------
@pytest.fixture(params=["memory", "redis"])
def breaker(request):
    clock = Clock()
    store = MemoryStateStore(clock) if request.param == "memory" else RedisStateStore(fakeredis.FakeAsyncRedis())
    return CircuitBreaker(store, failure_threshold=3, cooldown_s=30, clock=clock), clock


async def test_breaker_full_cycle(breaker):
    cb, clock = breaker
    assert await cb.state("groq") is BreakerState.CLOSED
    for _ in range(2):
        await cb.record_failure("groq")
    assert await cb.allow_request("groq")  # below threshold: still closed
    await cb.record_failure("groq")
    assert await cb.state("groq") is BreakerState.OPEN
    assert not await cb.allow_request("groq")

    clock.t += 31  # cooldown elapsed
    assert await cb.state("groq") is BreakerState.HALF_OPEN
    assert await cb.allow_request("groq")       # the single probe
    assert not await cb.allow_request("groq")   # concurrent request blocked while probing
    await cb.record_success("groq")
    assert await cb.state("groq") is BreakerState.CLOSED
    assert await cb.allow_request("groq")


async def test_breaker_failed_probe_reopens(breaker):
    cb, clock = breaker
    for _ in range(3):
        await cb.record_failure("groq")
    clock.t += 31
    assert await cb.allow_request("groq")
    await cb.record_failure("groq")
    assert await cb.state("groq") is BreakerState.OPEN
    assert not await cb.allow_request("groq")


async def test_success_resets_failure_count(breaker):
    cb, _ = breaker
    await cb.record_failure("groq")
    await cb.record_failure("groq")
    await cb.record_success("groq")
    await cb.record_failure("groq")
    await cb.record_failure("groq")
    assert await cb.state("groq") is BreakerState.CLOSED


# ---- fallback chain -------------------------------------------------------------------
CHAIN = [TierTarget(provider="a", model="m1"), TierTarget(provider="b", model="m2")]
REQ = LLMRequest(messages=[{"role": "user", "content": "hi"}])
CFG = ReliabilityConfig(max_retries=1, backoff_base_s=0, backoff_max_s=0, breaker_failure_threshold=2)


def _breaker() -> CircuitBreaker:
    return CircuitBreaker(MemoryStateStore(), failure_threshold=2, cooldown_s=30)


async def test_fallback_skips_unconfigured_provider():
    clients = {"a": FakeLLMClient("a", configured=False), "b": FakeLLMClient("b")}
    res = await call_with_fallback(CHAIN, REQ, clients, _breaker(), CFG, sleep=_no_sleep)
    assert res.response.provider == "b" and res.fallback_used
    assert [a.outcome for a in res.attempts] == ["skipped_unconfigured", "ok"]


async def test_open_breaker_skips_provider_without_calling_it():
    clients = {"a": FakeLLMClient("a"), "b": FakeLLMClient("b")}
    cb = _breaker()
    await cb.record_failure("a/m1")
    await cb.record_failure("a/m1")
    res = await call_with_fallback(CHAIN, REQ, clients, cb, CFG, sleep=_no_sleep)
    assert res.response.provider == "b"
    assert clients["a"].calls == []
    assert res.attempts[0].outcome == "skipped_circuit_open"


async def test_client_error_does_not_trip_breaker():
    clients = {"a": FakeLLMClient("a"), "b": FakeLLMClient("b")}
    clients["a"].always_fail = ProviderError("bad request", provider="a", status_code=400)
    cb = _breaker()
    for _ in range(3):
        await call_with_fallback(CHAIN, REQ, clients, cb, CFG, sleep=_no_sleep)
    assert await cb.state("a/m1") is BreakerState.CLOSED


async def test_all_fail_raises():
    clients = {"a": FakeLLMClient("a"), "b": FakeLLMClient("b")}
    for c in clients.values():
        c.always_fail = ProviderError("500", provider=c.name, status_code=500, retryable=True)
    with pytest.raises(AllProvidersFailed) as ei:
        await call_with_fallback(CHAIN, REQ, clients, _breaker(), CFG, sleep=_no_sleep)
    assert len(ei.value.attempts) == 2


# ---- rate limiter ---------------------------------------------------------------------
@pytest.mark.parametrize("use_redis", [False, True])
async def test_token_bucket_burst_then_refill(use_redis):
    clock = Clock()
    rl = TokenBucketRateLimiter(3, 1.0, fakeredis.FakeAsyncRedis() if use_redis else None, clock=clock)
    results = [(await rl.check("k")).allowed for _ in range(4)]
    assert results == [True, True, True, False]
    denied = await rl.check("k")
    assert not denied.allowed and denied.retry_after_s > 0
    clock.t += 1.0  # one token refilled
    assert (await rl.check("k")).allowed
    assert not (await rl.check("k")).allowed
    assert (await rl.check("other-key")).allowed  # buckets are per key


async def test_breaker_is_per_model_not_per_provider():
    # Found in the live demo: gpt-oss-20b out of quota opened the breaker for ALL of Groq,
    # so a healthy gpt-oss-120b on the same provider was skipped.
    groq = FakeLLMClient("groq")
    cb = _breaker()
    await cb.record_failure("groq/small")
    await cb.record_failure("groq/small")              # small model's breaker opens
    chain = [TierTarget(provider="groq", model="big")]
    res = await call_with_fallback(chain, REQ, {"groq": groq}, cb, CFG, sleep=_no_sleep)
    assert res.response.model == "big" and groq.calls[-1][0] == "big"
