"""FastAPI app: OpenAI-compatible endpoints, auth, rate limiting, request logging."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis

from routewise.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from routewise.cache.exact import ExactCache
from routewise.cache.guards import run_guards
from routewise.cache.models import CrossEncoderVerifier, SentenceTransformerEmbedder
from routewise.cache.semantic import SemanticCache
from routewise.cache.store import FaissSqliteStore
from routewise.core.config import PROJECT_ROOT, AppConfig, CacheConfig, Settings
from routewise.core.pipeline import Gateway, GatewayError, api_key_id
from routewise.providers.base import LLMClient
from routewise.providers.mock import MockLLMClient
from routewise.providers.openai_compat import OpenAICompatClient
from routewise.reliability.circuit_breaker import CircuitBreaker
from routewise.reliability.rate_limiter import TokenBucketRateLimiter, retry_after_header
from routewise.reliability.state import FallbackStateStore, MemoryStateStore, RedisStateStore
from routewise.router.baselines import LengthHeuristicRouter, Router
from routewise.router.learned import LearnedRouter
from routewise.storage.repository import RequestLogRepository, make_engine

log = logging.getLogger("routewise")

_AUTO = object()  # sentinel: "build this dependency from settings"


def _error(status: int, message: str, err_type: str, code: str | None = None,
           headers: dict[str, str] | None = None) -> JSONResponse:
    # Same error shape as the OpenAI API so SDKs raise their normal exception types.
    return JSONResponse({"error": {"message": message, "type": err_type, "code": code}},
                        status_code=status, headers=headers)


async def _connect_redis(url: str | None) -> Redis | None:
    if not url:
        return None
    client = Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=1.0)
    try:
        await client.ping()
        return client
    except Exception as e:  # noqa: BLE001
        log.warning("Redis unavailable at startup (%s): exact cache disabled, in-memory "
                    "rate limiting and circuit breaker state", e)
        await client.aclose()
        return None


def load_embedder(model_name: str) -> SentenceTransformerEmbedder | None:
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as e:  # noqa: BLE001
        log.warning("embedding model failed to load (%s): semantic cache and learned router DISABLED", e)
        return None


def build_router(cfg: AppConfig, embedder: Any) -> Router:
    """The trained router if its artifact exists and loads; otherwise the length heuristic."""
    path = Path(cfg.router.artifact_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        log.warning("no router artifact at %s: using the length-heuristic router", path)
        return LengthHeuristicRouter()
    try:
        r = LearnedRouter.load(path, embedder)
        r.decide("warm up")
        log.info("learned router loaded: %s (threshold %.3f)", r.name, r.threshold)
        return r
    except Exception as e:  # noqa: BLE001
        log.warning("router artifact failed to load (%s): using the length-heuristic router", e)
        return LengthHeuristicRouter()


def build_semantic_cache(cache_cfg: CacheConfig, embedder: Any = _AUTO) -> SemanticCache | None:
    """Load models and the store. Any failure disables the semantic cache, never the gateway."""
    if not cache_cfg.semantic_enabled:
        return None
    if embedder is _AUTO:
        embedder = load_embedder(cache_cfg.embedding_model)
    if embedder is None:
        return None
    verifier = None
    if cache_cfg.verifier_enabled:
        try:
            verifier = CrossEncoderVerifier(cache_cfg.verifier_model)
        except Exception as e:  # noqa: BLE001
            # Without the verifier, near-miss false hits would get through. Serving no
            # cached answers is safer than serving wrong ones.
            log.warning("verifier failed to load (%s): semantic cache DISABLED", e)
            return None
    if cache_cfg.guards_enabled:
        # Load spaCy now; otherwise the first real request pays ~1 s for it.
        run_guards("warm up", "warm up", cache_cfg.guards)
    store = FaissSqliteStore(cache_cfg.semantic_db_path, embedder.dim)
    log.info("semantic cache ready: %d entries", len(store))
    return SemanticCache(cache_cfg, embedder, store, verifier)


def create_app(
    settings: Settings | None = None,
    *,
    app_config: AppConfig | None = None,
    clients: dict[str, LLMClient] | None = None,
    redis: Any = _AUTO,
    repo: RequestLogRepository | None = None,
    router: Router | None = None,
    semantic_cache: Any = _AUTO,
) -> FastAPI:
    """App factory. Every dependency can be injected, which is how tests swap in mocks."""
    load_dotenv(override=False)
    settings = settings or Settings()
    cfg = app_config or settings.load_app_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        r = await _connect_redis(settings.redis_url) if redis is _AUTO else redis
        http = httpx.AsyncClient()
        if clients is not None:
            provider_clients = clients
        elif os.environ.get("ROUTEWISE_MOCK_PROVIDERS") == "1":
            log.warning("ROUTEWISE_MOCK_PROVIDERS=1: using canned mock answers (load testing only)")
            delay = float(os.environ.get("ROUTEWISE_MOCK_DELAY_MS", "0"))
            provider_clients = {name: MockLLMClient(name, delay) for name in cfg.providers}
        else:
            provider_clients = {name: OpenAICompatClient(name, pc, http) for name, pc in cfg.providers.items()}
        log_repo = repo or RequestLogRepository(make_engine(settings.database_url))
        log_repo.create_schema()

        breaker_store = FallbackStateStore(RedisStateStore(r) if r is not None else None, MemoryStateStore())
        breaker = CircuitBreaker(breaker_store, cfg.reliability.breaker_failure_threshold,
                                 cfg.reliability.breaker_cooldown_s)
        # One embedding model serves both the semantic cache and the learned router.
        need_embedder = (semantic_cache is _AUTO and cfg.cache.semantic_enabled) or router is None
        embedder = await asyncio.to_thread(load_embedder, cfg.cache.embedding_model) if need_embedder else None
        active_router = router or await asyncio.to_thread(build_router, cfg, embedder)

        app.state.cfg = cfg
        app.state.settings = settings
        app.state.redis = r
        app.state.repo = log_repo
        app.state.router = active_router
        app.state.rate_limiter = TokenBucketRateLimiter(cfg.rate_limit.capacity, cfg.rate_limit.refill_per_second, r)
        sem = (await asyncio.to_thread(build_semantic_cache, cfg.cache, embedder)
               if semantic_cache is _AUTO else semantic_cache)
        app.state.semantic_cache = sem
        app.state.gateway = Gateway(cfg, provider_clients, breaker, ExactCache(r, cfg.cache.exact_ttl_s),
                                    active_router, sem)
        yield
        await http.aclose()
        if r is not None and redis is _AUTO:
            await r.aclose()

    app = FastAPI(title="RouteWise", version="0.1.0", lifespan=lifespan,
                  description="Cost-aware OpenAI-compatible LLM gateway.")

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(400, str(exc.errors()[:3]), "invalid_request_error")

    async def authenticate(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if key not in settings.api_keys:
            raise GatewayError(401, "Invalid or missing API key.", "invalid_request_error", "invalid_api_key")
        rl_cfg = request.app.state.cfg.rate_limit
        if rl_cfg.enabled:
            result = await request.app.state.rate_limiter.check(api_key_id(key))
            if not result.allowed:
                raise _RateLimited(retry_after_header(result))
        return key

    @app.exception_handler(GatewayError)
    async def _gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
        return _error(exc.status_code, exc.message, exc.err_type, exc.code)

    @app.exception_handler(_RateLimited)
    async def _rate_limit_handler(_: Request, exc: _RateLimited) -> JSONResponse:
        return _error(429, "Rate limit exceeded for this API key.", "rate_limit_error",
                      "rate_limit_exceeded", headers={"Retry-After": exc.retry_after})

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        r: Redis | None = request.app.state.redis
        redis_ok = False
        if r is not None:
            try:
                redis_ok = bool(await r.ping())
            except Exception:  # noqa: BLE001
                redis_ok = False
        sem = request.app.state.semantic_cache
        return {"status": "ok", "redis": redis_ok, "database": request.app.state.repo.healthy(),
                "semantic_cache": sem is not None, "semantic_entries": len(sem.store) if sem else 0}

    @app.get("/v1/models")
    async def list_models(request: Request, _: str = Depends(authenticate)) -> dict[str, Any]:
        cfg_: AppConfig = request.app.state.cfg
        ids = ["auto", *cfg_.tiers.keys()] + [t.model for chain in cfg_.tiers.values() for t in chain]
        now = int(time.time())
        return {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "routewise"}
                                           for i in dict.fromkeys(ids)]}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        background: BackgroundTasks,
        api_key: str = Depends(authenticate),
    ) -> JSONResponse | StreamingResponse:
        gateway: Gateway = request.app.state.gateway
        repo: RequestLogRepository = request.app.state.repo
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            result = await gateway.handle(body, headers, api_key)
        except GatewayError as e:
            if e.log_record is not None:
                background.add_task(repo.write, e.log_record)
            resp = _error(e.status_code, e.message, e.err_type, e.code)
            resp.background = background
            return resp
        # Log after the response is sent, so DB latency never adds to user latency.
        background.add_task(repo.write, result.log_record)
        if body.stream:
            return StreamingResponse(_buffered_sse(result.response), media_type="text/event-stream",
                                     background=background)
        return JSONResponse(result.response.model_dump(mode="json"), background=background)

    return app


class _RateLimited(Exception):
    def __init__(self, retry_after: str) -> None:
        self.retry_after = retry_after


async def _buffered_sse(resp: ChatCompletionResponse) -> AsyncIterator[str]:
    """OpenAI-format SSE stream of an already-complete answer.

    This is *buffered* streaming: we wait for the full answer, then emit it in chunks.
    It keeps `stream=True` clients working, and keeps caching/fallback simple (a provider
    that dies mid-stream can't be transparently replaced). Time-to-first-token is
    therefore not improved — a deliberate trade-off, documented in the README.
    """
    base = {"id": resp.id, "object": "chat.completion.chunk", "created": resp.created, "model": resp.model}
    content = resp.choices[0].message.content
    yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                       "finish_reason": None}]}) + "\n\n"
    step = 40
    for i in range(0, len(content), step):
        chunk = {**base, "choices": [{"index": 0, "delta": {"content": content[i:i + step]}, "finish_reason": None}]}
        yield "data: " + json.dumps(chunk) + "\n\n"
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": resp.choices[0].finish_reason}],
             "usage": resp.usage.model_dump(), "routewise": resp.routewise.model_dump(mode="json")}
    yield "data: " + json.dumps(final) + "\n\n"
    yield "data: [DONE]\n\n"
