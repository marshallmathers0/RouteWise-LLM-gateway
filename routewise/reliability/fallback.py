"""Fallback chain: try each (provider, model) target of a tier in order."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable

from routewise.core.config import ReliabilityConfig, TierTarget
from routewise.providers.base import LLMClient, LLMRequest, LLMResponse, ProviderError
from routewise.reliability.circuit_breaker import CircuitBreaker
from routewise.reliability.retry import retry_async

log = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    provider: str
    model: str
    outcome: str  # "ok" | "skipped_unconfigured" | "skipped_circuit_open" | "error"
    error: str | None = None


@dataclass
class FallbackResult:
    response: LLMResponse
    fallback_used: bool
    attempts: list[AttemptRecord] = field(default_factory=list)


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list[AttemptRecord]) -> None:
        summary = "; ".join(f"{a.provider}/{a.model}: {a.outcome} {a.error or ''}".strip() for a in attempts)
        super().__init__(f"all providers failed: {summary}")
        self.attempts = attempts


async def call_with_fallback(
    chain: list[TierTarget],
    request: LLMRequest,
    clients: dict[str, LLMClient],
    breaker: CircuitBreaker,
    cfg: ReliabilityConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FallbackResult:
    attempts: list[AttemptRecord] = []
    for target in chain:
        client = clients.get(target.provider)
        if client is None or not client.is_configured():
            attempts.append(AttemptRecord(target.provider, target.model, "skipped_unconfigured"))
            continue
        # One breaker per provider+MODEL, not per provider: rate limits (and many outages)
        # are per model, and gpt-oss-20b running out of quota must not take a healthy
        # gpt-oss-120b on the same provider offline. A full provider outage still trips
        # each model's breaker after a few requests.
        breaker_key = f"{target.provider}/{target.model}"
        if not await breaker.allow_request(breaker_key):
            attempts.append(AttemptRecord(target.provider, target.model, "skipped_circuit_open"))
            continue
        try:
            response = await retry_async(
                lambda c=client, t=target: c.chat(t.model, replace(request, extra={**request.extra, **t.params}),
                                                  cfg.request_timeout_s),
                max_retries=cfg.max_retries,
                base_s=cfg.backoff_base_s,
                max_s=cfg.backoff_max_s,
                sleep=sleep,
            )
        except ProviderError as e:
            # Only transient failures say anything about provider health. A 400 for a bad
            # request shouldn't trip the breaker for everyone else.
            if e.retryable:
                await breaker.record_failure(breaker_key)
            log.warning("provider %s/%s failed: %s", target.provider, target.model, e)
            attempts.append(AttemptRecord(target.provider, target.model, "error", str(e)[:200]))
            continue
        await breaker.record_success(breaker_key)
        attempts.append(AttemptRecord(target.provider, target.model, "ok"))
        # "Fallback used" = we served from something other than the tier's first choice.
        return FallbackResult(response, fallback_used=target is not chain[0], attempts=attempts)
    raise AllProvidersFailed(attempts)
