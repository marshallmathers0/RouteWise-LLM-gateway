"""Cache eligibility rules and exact-cache key correctness."""

from __future__ import annotations

import pytest

from routewise.api.schemas import ChatCompletionRequest
from routewise.cache.eligibility import check_eligibility, is_time_sensitive
from routewise.cache.exact import exact_cache_key


def req(*msgs: tuple[str, str], **kw) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[{"role": r, "content": c} for r, c in msgs], **{"temperature": 0, **kw})


@pytest.mark.parametrize("q", ["What's the weather today?", "latest iPhone price", "Who is the current PM of UK?",
                               "news about AI this week",
                               # added after the live demo cached "iphone 17 price":
                               "iphone 17 price", "How much does a Tesla Model 3 cost?", "bitcoin prices",
                               "GTA 6 release date", "upcoming Marvel movies", "Who won the IPL final?",
                               "Is the Taj Mahal still open?"])
def test_time_sensitive_detected(q):
    assert is_time_sensitive(q)


@pytest.mark.parametrize("q", ["What is the capital of India?", "Explain photosynthesis", "Write a haiku about rain",
                               "back propagation", "What is pagination?", "Explain the time complexity of quicksort"])
def test_timeless_not_flagged(q):
    assert not is_time_sensitive(q)


def test_eligibility_rules():
    single = req(("user", "Capital of France?"))
    assert check_eligibility(single, {}, 0.3).semantic
    assert check_eligibility(req(("user", "x"), temperature=0.7), {}, 0.3).reason == "temperature>0.3"
    # Unset temperature = provider default (~1.0): not deterministic, so not cacheable.
    assert not check_eligibility(req(("user", "x"), temperature=None), {}, 0.3).exact
    assert check_eligibility(single, {"x-routewise-cache": "off"}, 0.3).reason == "client_opt_out"
    assert check_eligibility(single, {"cache-control": "no-cache"}, 0.3).reason == "client_opt_out"
    multi = req(("user", "Name a fruit"), ("assistant", "Apple"), ("user", "Another one?"))
    e = check_eligibility(multi, {}, 0.3)
    assert e.exact and not e.semantic and e.reason == "multi_turn"
    # A system prompt alone doesn't make it multi-turn.
    assert check_eligibility(req(("system", "Be brief"), ("user", "Hi")), {}, 0.3).semantic


def test_exact_key_normalizes_whitespace_only():
    assert exact_cache_key(req(("user", "Hello   world "))) == exact_cache_key(req(("user", "Hello world")))
    assert exact_cache_key(req(("user", "Hello world"))) != exact_cache_key(req(("user", "hello world")))


def test_exact_key_includes_system_prompt_model_and_params():
    base = exact_cache_key(req(("system", "A"), ("user", "Q")))
    assert base != exact_cache_key(req(("system", "B"), ("user", "Q")))
    assert base != exact_cache_key(req(("system", "A"), ("user", "Q"), model="strong"))
    # max_tokens is NOT part of the key (the pipeline checks the cached answer fits instead).
    assert base == exact_cache_key(req(("system", "A"), ("user", "Q"), max_tokens=10))
    # JSON mode changes the answer format, so it must change the key.
    assert base != exact_cache_key(req(("system", "A"), ("user", "Q"), response_format={"type": "json_object"}))
    # The role matters: same text as system vs user must not collide.
    assert exact_cache_key(req(("system", "Q"))) != exact_cache_key(req(("user", "Q")))
