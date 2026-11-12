"""Shared fixtures. No test ever calls a real LLM provider."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from routewise.router.baselines import LengthHeuristicRouter
from routewise.api.app import create_app
from routewise.core.config import AppConfig, Settings
from routewise.providers.base import LLMRequest, LLMResponse, ProviderError
from routewise.storage.repository import RequestLogRepository, make_engine

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class FakeLLMClient:
    """Stand-in provider. `failures` is a list of ProviderErrors raised before succeeding."""

    def __init__(self, name: str, reply: str | None = None, configured: bool = True) -> None:
        self.name = name
        self.reply = reply or f"answer from {name}"
        self.configured = configured
        self.failures: list[ProviderError] = []
        self.always_fail: ProviderError | None = None
        self.calls: list[tuple[str, LLMRequest]] = []

    def is_configured(self) -> bool:
        return self.configured

    async def chat(self, model: str, request: LLMRequest, timeout_s: float) -> LLMResponse:
        self.calls.append((model, request))
        if self.always_fail is not None:
            raise self.always_fail
        if self.failures:
            raise self.failures.pop(0)
        return LLMResponse(self.reply, prompt_tokens=10, completion_tokens=20, finish_reason="stop",
                           model=model, provider=self.name)

    async def aclose(self) -> None:
        pass


@pytest.fixture
def app_config() -> AppConfig:
    cfg = Settings(_env_file=None).load_app_config()
    # Make tests fast: no real backoff sleeping, small breaker threshold.
    cfg.reliability.backoff_base_s = 0.0
    cfg.reliability.backoff_max_s = 0.0
    cfg.reliability.breaker_failure_threshold = 2
    return cfg


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, routewise_api_keys=API_KEY, redis_url=None,
                    database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}")


@pytest.fixture
def fake_clients() -> dict[str, FakeLLMClient]:
    return {name: FakeLLMClient(name) for name in ("groq", "gemini", "openrouter")}


@pytest.fixture
def fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis()


@pytest.fixture
def repo(settings: Settings) -> RequestLogRepository:
    return RequestLogRepository(make_engine(settings.database_url))


@pytest.fixture
def client(settings, app_config, fake_clients, fake_redis, repo) -> Iterator[TestClient]:
    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=fake_redis, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=None)
    with TestClient(app) as c:
        yield c


def chat_body(text: str = "What is the capital of France?", **kw) -> dict:
    return {"model": "auto", "messages": [{"role": "user", "content": text}], "temperature": 0, **kw}
