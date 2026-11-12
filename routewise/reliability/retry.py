"""Retries with exponential backoff and full jitter."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from routewise.providers.base import ProviderError

T = TypeVar("T")


def backoff_delay(attempt: int, base_s: float, max_s: float, rng: random.Random | None = None) -> float:
    """Full jitter: sleep a random amount in [0, min(max, base * 2^attempt)].

    Plain exponential backoff makes many clients that failed together retry together
    (a "thundering herd"). Randomizing the whole interval spreads them out.
    """
    cap = min(max_s, base_s * (2**attempt))
    return (rng or random).uniform(0, cap)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_s: float,
    max_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, ProviderError], None] | None = None,
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except ProviderError as e:
            if not e.retryable or attempt >= max_retries:
                raise
            if e.retry_after_s is not None and e.retry_after_s > max_s:
                # The provider says it won't recover within our backoff budget (e.g. a
                # rate-limit window). Waiting would only add latency: give up on this
                # provider now so the fallback chain can try the next one.
                raise
            delay = backoff_delay(attempt, base_s, max_s)
            if e.retry_after_s is not None:
                delay = max(delay, e.retry_after_s)  # respect the provider's hint
            if on_retry:
                on_retry(attempt, e)
            await sleep(delay)
            attempt += 1
