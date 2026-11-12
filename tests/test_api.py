"""End-to-end gateway behaviour through the HTTP API (providers mocked)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from routewise.providers.base import ProviderError
from routewise.storage.models import RequestLog
from tests.conftest import AUTH, chat_body


def _logs(repo) -> list[RequestLog]:
    with Session(repo.engine) as s:
        return list(s.scalars(select(RequestLog).order_by(RequestLog.id)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["database"] is True


def test_requires_api_key(client):
    r = client.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_basic_completion_has_openai_shape_and_metadata(client, fake_clients):
    r = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "answer from groq"}
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    meta = body["routewise"]
    # Short factual query -> heuristic router sends it to the cheap tier (groq 8B first).
    assert meta["route"] == "cheap"
    assert meta["provider_used"] == "groq"
    assert meta["model_used"] == "openai/gpt-oss-20b"
    assert meta["cache_status"]["status"] == "miss"
    assert meta["fallback_used"] is False
    assert meta["estimated_cost_usd"] > 0 and meta["cost_is_estimate"] is True
    assert {"llm", "total"} <= meta["latency_ms"].keys()


def test_request_is_logged(client, repo):
    client.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
    rows = _logs(repo)
    assert len(rows) == 1
    row = rows[0]
    assert row.route == "cheap" and row.provider == "groq" and row.cache_status == "miss"
    assert row.tokens_in == 10 and row.tokens_out == 20
    assert row.api_key_id != "test-key"  # raw key never stored
    # 70B counterfactual must cost more than the 8B call actually used.
    assert row.counterfactual_strong_cost_usd > row.estimated_cost_usd


def test_explicit_model_bypasses_router(client, fake_clients):
    r = client.post("/v1/chat/completions", json=chat_body(model="openai/gpt-oss-120b"), headers=AUTH)
    meta = r.json()["routewise"]
    assert meta["route"] == "strong" and meta["route_reason"] == "explicit_model"
    assert fake_clients["groq"].calls[-1][0] == "openai/gpt-oss-120b"


def test_response_format_is_passed_to_provider(client, fake_clients):
    client.post("/v1/chat/completions", json=chat_body(response_format={"type": "json_object"}), headers=AUTH)
    sent = fake_clients["groq"].calls[-1][1]
    assert sent.extra["response_format"] == {"type": "json_object"}
    assert sent.extra["reasoning_effort"] == "low"  # tier params from config are merged in too


def test_explicit_tier_name(client):
    r = client.post("/v1/chat/completions", json=chat_body(model="strong"), headers=AUTH)
    assert r.json()["routewise"]["route"] == "strong"


def test_unknown_model_is_400(client):
    r = client.post("/v1/chat/completions", json=chat_body(model="gpt-9"), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_found"


def test_exact_cache_hit_on_repeat(client, fake_clients):
    first = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH).json()
    second = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH).json()
    assert first["routewise"]["cache_status"]["status"] == "miss"
    assert second["routewise"]["cache_status"]["status"] == "exact_hit"
    assert second["routewise"]["estimated_cost_usd"] == 0
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]
    assert len(fake_clients["groq"].calls) == 1  # provider only called once


def test_different_system_prompts_do_not_share_cache(client, fake_clients):
    def body(system: str) -> dict:
        return {"model": "auto", "temperature": 0, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": "Say hi"}]}

    a = client.post("/v1/chat/completions", json=body("You are a pirate."), headers=AUTH).json()
    b = client.post("/v1/chat/completions", json=body("You are a lawyer."), headers=AUTH).json()
    assert a["routewise"]["cache_status"]["status"] == "miss"
    assert b["routewise"]["cache_status"]["status"] == "miss"
    assert len(fake_clients["groq"].calls) == 2


def test_high_temperature_bypasses_cache(client, fake_clients):
    for _ in range(2):
        r = client.post("/v1/chat/completions", json=chat_body(temperature=0.9), headers=AUTH).json()
        assert r["routewise"]["cache_status"] == {"status": "bypassed", "reason": "temperature>0.3"}
    assert len(fake_clients["groq"].calls) == 2


def test_opt_out_header_bypasses_cache(client):
    client.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
    r = client.post("/v1/chat/completions", json=chat_body(), headers={**AUTH, "X-RouteWise-Cache": "off"}).json()
    assert r["routewise"]["cache_status"]["status"] == "bypassed"


def test_fallback_within_budget(client, fake_clients):
    # gpt-oss-20b down -> gpt-oss-120b (same price as the always-strong reference: allowed).
    fake_clients["groq"].failures = [ProviderError("down", provider="groq", status_code=503, retryable=True)] * 3
    meta = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH).json()["routewise"]
    assert meta["model_used"] == "openai/gpt-oss-120b" and meta["fallback_used"] is True


def test_cost_guard_skips_fallback_pricier_than_strong(client, fake_clients):
    # Both Groq models down. Gemini Flash-Lite costs more per token than the strong model,
    # so the cost guard skips it and the gateway returns 503 instead of overspending.
    fake_clients["groq"].always_fail = ProviderError("down", provider="groq", status_code=503, retryable=True)
    r = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
    assert r.status_code == 503
    assert "gemini/gemini-3.5-flash-lite: skipped_over_budget" in r.json()["error"]["message"]
    assert fake_clients["gemini"].calls == []
    # Cheap chain: groq 20b -> groq 120b; max_retries=2 -> 3 attempts per Groq model.
    models_tried = [m for m, _ in fake_clients["groq"].calls]
    assert models_tried == ["openai/gpt-oss-20b"] * 3 + ["openai/gpt-oss-120b"] * 3


def test_cost_guard_can_be_disabled(client, fake_clients, app_config):
    app_config.reliability.fallback_price_cap_ratio = None  # availability over cost
    fake_clients["groq"].always_fail = ProviderError("down", provider="groq", status_code=503, retryable=True)
    meta = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH).json()["routewise"]
    assert meta["provider_used"] == "gemini" and meta["fallback_used"] is True


def test_all_providers_fail_returns_503_and_logs(client, fake_clients, repo):
    err = ProviderError("down", provider="x", status_code=500, retryable=True)
    for c in fake_clients.values():
        c.always_fail = err
    r = client.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "all_providers_failed"
    rows = _logs(repo)
    assert rows[-1].status_code == 503 and "all providers failed" in rows[-1].error


def test_rate_limit_returns_429(client):
    from routewise.reliability.rate_limiter import TokenBucketRateLimiter

    # Swap in a tiny bucket: burst of 2, effectively no refill during the test.
    client.app.state.rate_limiter = TokenBucketRateLimiter(2, 0.001, None)
    codes = [client.post("/v1/chat/completions", json=chat_body(f"q{i}"), headers=AUTH).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/v1/chat/completions", json=chat_body("q"), headers=AUTH)
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert int(r.headers["Retry-After"]) >= 1


def test_list_models(client):
    ids = [m["id"] for m in client.get("/v1/models", headers=AUTH).json()["data"]]
    assert ids[:3] == ["auto", "cheap", "strong"]
    assert "openai/gpt-oss-120b" in ids


def test_buffered_streaming(client):
    with client.stream("POST", "/v1/chat/completions", json=chat_body(stream=True), headers=AUTH) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert "answer from groq" in "".join(lines).replace('"', "")


def test_cached_answer_reused_across_max_tokens_but_never_too_long(client, fake_clients):
    # The fake answer uses 20 completion tokens.
    client.post("/v1/chat/completions", json=chat_body(max_tokens=500), headers=AUTH)
    roomy = client.post("/v1/chat/completions", json=chat_body(max_tokens=1000), headers=AUTH).json()
    assert roomy["routewise"]["cache_status"]["status"] == "exact_hit"  # different max_tokens, same entry
    tight = client.post("/v1/chat/completions", json=chat_body(max_tokens=10), headers=AUTH).json()
    assert tight["routewise"]["cache_status"] == {"status": "miss", "reason": "cached_answer_exceeds_max_tokens"}
    assert len(fake_clients["groq"].calls) == 2
