"""The request pipeline: eligibility -> exact cache -> router -> providers -> response + log.

This module has no FastAPI imports, so it can be unit-tested and reused (e.g. by the
playground) without HTTP in the way.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from routewise.api.schemas import (
    CacheStatus,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    RouteWiseMeta,
    Usage,
)
from routewise.cache.eligibility import check_eligibility
from routewise.cache.exact import CachedAnswer, ExactCache, exact_cache_key
from routewise.cache.semantic import SemanticCache, SemanticLookup, semantic_namespace
from routewise.core.config import AppConfig, ModelPrice, TierTarget
from routewise.core.pricing import estimate_cost_usd
from routewise.providers.base import LLMClient, LLMRequest
from routewise.reliability.circuit_breaker import CircuitBreaker
from routewise.reliability.fallback import AllProvidersFailed, call_with_fallback
from routewise.router.baselines import RouteDecision, Router

log = logging.getLogger(__name__)


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        err_type: str,
        code: str | None = None,
        log_record: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.code = code
        self.log_record = log_record  # set when the failure should still be logged


class StageTimer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - start) * 1000, 3)


def api_key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


@dataclass
class GatewayResult:
    response: ChatCompletionResponse
    log_record: dict[str, Any]


class Gateway:
    def __init__(
        self,
        cfg: AppConfig,
        clients: dict[str, LLMClient],
        breaker: CircuitBreaker,
        exact_cache: ExactCache,
        router: Router,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.clients = clients
        self.breaker = breaker
        self.exact_cache = exact_cache
        self.router = router
        self.semantic_cache = semantic_cache

    def _fallback_price_cap(self) -> ModelPrice | None:
        ratio = self.cfg.reliability.fallback_price_cap_ratio
        ref = self.cfg.pricing.get(self.cfg.counterfactual_strong_model)
        if ratio is None or ref is None:
            return None
        return ModelPrice(input=ref.input * ratio, output=ref.output * ratio)

    # ---- model resolution -------------------------------------------------------------
    def _resolve(self, req: ChatCompletionRequest, timer: StageTimer) -> tuple[str, list[TierTarget], RouteDecision]:
        if req.model == "auto":
            with timer.stage("router"):
                decision = self.router.decide(req.last_user_text())
            return decision.tier, self.cfg.tiers[decision.tier], decision
        if req.model in self.cfg.tiers:
            return req.model, self.cfg.tiers[req.model], RouteDecision(req.model, None, "explicit_tier")
        tier = self.cfg.tier_of_model(req.model)
        if tier is None:
            raise GatewayError(
                400,
                f"Unknown model '{req.model}'. Use 'auto', a tier name ({', '.join(self.cfg.tiers)}), "
                "or a model listed in GET /v1/models.",
                "invalid_request_error",
                "model_not_found",
            )
        # Explicit model bypasses the router: try that model first, then the rest of its tier.
        chain = self.cfg.tiers[tier]
        chain = [t for t in chain if t.model == req.model] + [t for t in chain if t.model != req.model]
        return tier, chain, RouteDecision(tier, None, "explicit_model")

    # ---- main entry point -------------------------------------------------------------
    async def handle(self, req: ChatCompletionRequest, headers: dict[str, str], api_key: str) -> GatewayResult:
        request_id = uuid.uuid4().hex
        timer = StageTimer()
        t0 = time.perf_counter()
        meta = RouteWiseMeta(request_id=request_id)

        max_tokens = req.effective_max_tokens()

        def fits(completion_tokens: int) -> bool:
            # A cached answer longer than the caller allows would break their max_tokens contract.
            return max_tokens is None or completion_tokens <= max_tokens

        with timer.stage("eligibility"):
            elig = check_eligibility(req, headers, self.cfg.cache.max_cacheable_temperature)

        use_exact = elig.exact and self.cfg.cache.exact_enabled and self.exact_cache.available
        key = exact_cache_key(req)
        if not elig.exact:
            meta.cache_status = CacheStatus(status="bypassed", reason=elig.reason)
        elif not use_exact and self.exact_cache.configured:
            # Redis is configured but unreachable. Still a "miss", not "bypassed": the semantic
            # cache may run and overwrite the reason (e.g. "semantic_rejected:guard:word_order").
            # Without Redis configured at all there is nothing to report: the exact cache is
            # simply not part of this deployment.
            meta.cache_status = CacheStatus(status="miss", reason="exact_cache_unavailable")

        if use_exact:
            with timer.stage("exact_cache"):
                cached = await self.exact_cache.get(key)
            if cached is not None and not fits(cached.completion_tokens):
                meta.cache_status = CacheStatus(status="miss", reason="cached_answer_exceeds_max_tokens")
                cached = None
            if cached is not None:
                meta.cache_status = CacheStatus(status="exact_hit")
                meta.route = cached.route
                meta.route_reason = "cache_hit"
                meta.model_used = cached.model
                meta.provider_used = "cache"
                # A hit costs nothing; we report the usage of the original answer for reference.
                return self._finish(req, meta, timer, t0, cached.content, cached.model,
                                    cached.prompt_tokens, cached.completion_tokens, api_key, cost=0.0)

        # ---- semantic cache (single-turn, low temperature only) ----
        sem: SemanticLookup | None = None
        namespace: str | None = None
        if elig.semantic and self.semantic_cache is not None and self.cfg.cache.semantic_enabled:
            namespace = semantic_namespace(req)
            try:
                with timer.stage("semantic_cache"):
                    # CPU-bound model inference: run in a worker thread so the event loop
                    # keeps serving other requests meanwhile.
                    sem = await asyncio.to_thread(self.semantic_cache.lookup, req.last_user_text(), namespace)
            except Exception:  # noqa: BLE001 - a cache failure must degrade to a miss, not a 500
                log.exception("semantic cache lookup failed; treating as miss")
                namespace = None
        if sem is not None:
            timer.ms.update(sem.latency_ms)
            best = sem.best
            if best is not None:
                meta.matched_query, meta.similarity, meta.verifier_score = best.query, best.similarity, best.verifier_score
                meta.guard_results = sem.summary()
            if sem.hit is not None and not fits(sem.hit.completion_tokens):
                meta.cache_status = CacheStatus(status="miss", reason="cached_answer_exceeds_max_tokens")
            elif sem.hit is not None:
                h = sem.hit
                meta.cache_status = CacheStatus(status="semantic_hit")
                meta.route, meta.route_reason = h.route, "cache_hit"
                meta.model_used, meta.provider_used = h.model, "cache"
                return self._finish(req, meta, timer, t0, h.answer, h.model, h.prompt_tokens,
                                    h.completion_tokens, api_key, cost=0.0)
            elif best is not None and meta.cache_status.status == "miss":
                meta.cache_status = CacheStatus(status="miss", reason=f"semantic_rejected:{best.rejected_by}")

        tier, chain, decision = self._resolve(req, timer)
        meta.route, meta.route_reason, meta.router_prob = tier, decision.reason, decision.prob_cheap_ok

        llm_req = LLMRequest(
            messages=[{"role": m.role if m.role != "developer" else "system", "content": m.text()} for m in req.messages],
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.effective_max_tokens(),
            stop=req.stop,
            extra={"response_format": req.response_format} if req.response_format else {},
        )
        try:
            with timer.stage("llm"):
                result = await call_with_fallback(chain, llm_req, self.clients, self.breaker, self.cfg.reliability,
                                                  pricing=self.cfg.pricing, price_cap=self._fallback_price_cap())
        except AllProvidersFailed as e:
            meta.latency_ms = {**timer.ms, "total": round((time.perf_counter() - t0) * 1000, 3)}
            record = self._log_record(req, meta, api_key, 0, 0, status_code=503, error=str(e))
            # Include per-provider outcomes (e.g. "skipped_unconfigured") — it makes missing
            # API keys and outages obvious to whoever is calling the gateway.
            raise GatewayError(503, f"Tier '{tier}': {e}", "provider_unavailable",
                               "all_providers_failed", log_record=record) from e

        r = result.response
        meta.provider_used, meta.model_used, meta.fallback_used = r.provider, r.model, result.fallback_used
        # Don't cache empty or truncated (finish_reason="length") answers: serving a cut-off
        # answer to every future paraphrase would multiply one bad response.
        cacheable = bool(r.content.strip()) and r.finish_reason != "length"
        if use_exact and cacheable:
            await self.exact_cache.set(
                key, CachedAnswer(r.content, r.model, r.provider, r.prompt_tokens, r.completion_tokens, tier)
            )
        if sem is not None and namespace is not None and cacheable:
            try:
                await asyncio.to_thread(self.semantic_cache.add, namespace, req.last_user_text(), sem.embedding,  # type: ignore[union-attr]
                                        r.content, r.model, r.provider, r.prompt_tokens, r.completion_tokens, tier)
            except Exception:  # noqa: BLE001
                log.exception("semantic cache add failed")
        cost = estimate_cost_usd(self.cfg.pricing, r.model, r.prompt_tokens, r.completion_tokens)
        return self._finish(req, meta, timer, t0, r.content, r.model, r.prompt_tokens, r.completion_tokens,
                            api_key, cost=cost, finish_reason=r.finish_reason)

    # ---- helpers ----------------------------------------------------------------------
    def _finish(
        self,
        req: ChatCompletionRequest,
        meta: RouteWiseMeta,
        timer: StageTimer,
        t0: float,
        content: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        api_key: str,
        cost: float,
        finish_reason: str | None = "stop",
    ) -> GatewayResult:
        meta.estimated_cost_usd = round(cost, 8)
        meta.latency_ms = {**timer.ms, "total": round((time.perf_counter() - t0) * 1000, 3)}
        response = ChatCompletionResponse(
            id=f"chatcmpl-rw-{meta.request_id[:24]}",
            model=model,
            choices=[Choice(message=ResponseMessage(content=content), finish_reason=finish_reason)],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
            routewise=meta,
        )
        return GatewayResult(response, self._log_record(req, meta, api_key, prompt_tokens, completion_tokens))

    def _log_record(
        self,
        req: ChatCompletionRequest,
        meta: RouteWiseMeta,
        api_key: str,
        prompt_tokens: int,
        completion_tokens: int,
        status_code: int = 200,
        error: str | None = None,
    ) -> dict[str, Any]:
        query = req.last_user_text()
        lc = self.cfg.logging
        counterfactual = estimate_cost_usd(
            self.cfg.pricing, self.cfg.counterfactual_strong_model, prompt_tokens, completion_tokens
        )
        return {
            "request_id": meta.request_id,
            "api_key_id": api_key_id(api_key),
            "prompt_hash": exact_cache_key(req).split(":")[-1],
            "query_text": query[: lc.prompt_max_chars] if lc.log_prompts else None,
            "requested_model": req.model,
            "route": meta.route,
            "route_reason": meta.route_reason,
            "router_prob": meta.router_prob,
            "model": meta.model_used,
            "provider": meta.provider_used,
            "fallback_used": meta.fallback_used,
            "cache_status": meta.cache_status.status,
            "cache_reason": meta.cache_status.reason,
            "matched_query": meta.matched_query,
            "similarity": meta.similarity,
            "verifier_score": meta.verifier_score,
            "guard_results": meta.guard_results,
            "tokens_in": prompt_tokens,
            "tokens_out": completion_tokens,
            "latency_total_ms": meta.latency_ms.get("total", 0.0),
            "latency_ms": meta.latency_ms,
            "estimated_cost_usd": meta.estimated_cost_usd,
            "counterfactual_strong_cost_usd": round(counterfactual, 8),
            "status_code": status_code,
            "error": error[:2000] if error else None,
        }
