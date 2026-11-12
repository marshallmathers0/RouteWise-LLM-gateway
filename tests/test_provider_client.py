"""OpenAICompatClient against mocked HTTP (respx) — error classification and parsing."""

from __future__ import annotations

import httpx
import pytest
import respx

from routewise.core.config import ProviderConfig
from routewise.providers.base import LLMRequest, ProviderError
from routewise.providers.openai_compat import OpenAICompatClient

URL = "https://example.test/v1/chat/completions"
REQ = LLMRequest(messages=[{"role": "user", "content": "hello there"}], temperature=0)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-test")
    return OpenAICompatClient("testp", ProviderConfig(base_url="https://example.test/v1", api_key_env="TEST_PROVIDER_KEY"))


def _ok(usage: bool = True) -> dict:
    body = {"choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}]}
    if usage:
        body["usage"] = {"prompt_tokens": 5, "completion_tokens": 2}
    return body


@respx.mock
async def test_success_parses_content_and_usage(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    r = await client.chat("m", REQ, timeout_s=5)
    assert (r.content, r.prompt_tokens, r.completion_tokens, r.tokens_estimated) == ("Hi!", 5, 2, False)
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer sk-test"
    assert b'"temperature":0' in sent.content.replace(b" ", b"")


@respx.mock
async def test_missing_usage_is_estimated(client):
    respx.post(URL).mock(return_value=httpx.Response(200, json=_ok(usage=False)))
    r = await client.chat("m", REQ, timeout_s=5)
    assert r.tokens_estimated and r.prompt_tokens >= 1


@respx.mock
async def test_429_is_retryable_with_retry_after(client):
    respx.post(URL).mock(return_value=httpx.Response(429, headers={"retry-after": "3"}, text="slow down"))
    with pytest.raises(ProviderError) as ei:
        await client.chat("m", REQ, timeout_s=5)
    assert ei.value.retryable and ei.value.retry_after_s == 3.0 and ei.value.status_code == 429


@respx.mock
async def test_400_is_not_retryable(client):
    respx.post(URL).mock(return_value=httpx.Response(400, text="bad model"))
    with pytest.raises(ProviderError) as ei:
        await client.chat("m", REQ, timeout_s=5)
    assert not ei.value.retryable


@respx.mock
async def test_timeout_is_retryable(client):
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError) as ei:
        await client.chat("m", REQ, timeout_s=5)
    assert ei.value.retryable


async def test_missing_key_means_unconfigured(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    c = OpenAICompatClient("x", ProviderConfig(base_url="https://x", api_key_env="NOPE_KEY"))
    assert not c.is_configured()
