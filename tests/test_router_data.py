"""Router dataset pipeline: extraction, labeling rules, judge parsing, throttling, resume."""

from __future__ import annotations

import json

import pytest

import eval.router_collect as collect
import eval.router_label as label
from eval.llm_batch import DailyLimitReached, Throttle, call_rate_limited, is_daily_limit, read_jsonl
from eval.router_label import (
    build_judge_prompt,
    extract_gsm8k,
    extract_mmlu,
    ground_truth_label,
    is_correct,
    judge_label,
    order_cheap_first,
    parse_judge,
)
from routewise.providers.base import LLMRequest, LLMResponse, ProviderError


# ---------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,expected", [
    ("Step 1 ... 3 + 4 = 7\nAnswer: 18", 18.0),
    ("so she earns $1,250.50 per week.\nAnswer: $1,250.50", 1250.5),
    ("The answer is 42.", 42.0),                       # no tag: last number
    ("x = 5, y = 10. **Answer:** -3", -3.0),
    ("I'm not sure.", None),
])
def test_extract_gsm8k(text, expected):
    assert extract_gsm8k(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Reasoning...\nAnswer: B", "B"),
    ("The answer is (c).", "C"),
    ("Answer: **D**", "D"),
    ("blah\nA", "A"),                                  # bare letter on last line
    ("I think it is about economics", None),
])
def test_extract_mmlu(text, expected):
    assert extract_mmlu(text) == expected


def test_is_correct():
    assert is_correct({"source": "gsm8k", "gold": "1250"}, "Answer: 1,250")
    assert not is_correct({"source": "gsm8k", "gold": "18"}, "Answer: 17")
    assert is_correct({"source": "mmlu", "gold": "B"}, "Answer: B")
    assert is_correct({"source": "mmlu_pro", "gold": "I"}, "Reasoning...\nAnswer: I")
    assert not is_correct({"source": "mmlu", "gold": "I"}, "Answer: I")  # MMLU only has A-D


@pytest.mark.parametrize("gold,answer,ok", [
    (r"\frac{1}{2}", r"so the answer is \boxed{0.5}", True),
    (r"\frac{1}{2}", r"\boxed{\dfrac12}", True),
    (r"\left( 3, \frac{\pi}{2} \right)", r"$(r,\theta)=\boxed{(3, \pi/2)}$", True),
    ("3", r"\boxed{4}", False),
    ("3", "I could not solve it.", False),
])
def test_math_equivalence(gold, answer, ok):
    assert is_correct({"source": "math", "gold": gold}, answer) is ok


# ---------------------------------------------------------------- labeling rules
def test_ground_truth_label_rule():
    assert ground_truth_label(cheap_ok=True, strong_ok=True) == 1
    assert ground_truth_label(cheap_ok=True, strong_ok=False) == 1
    assert ground_truth_label(cheap_ok=False, strong_ok=True) == 0   # the only case worth paying for
    assert ground_truth_label(cheap_ok=False, strong_ok=False) == 1  # strong bought nothing


def test_judge_label_margin():
    assert judge_label(7, 8, margin=1) == 1
    assert judge_label(6, 8, margin=1) == 0
    assert judge_label(9, 6, margin=1) == 1
    assert judge_label(7, 8, margin=0) == 0


# ---------------------------------------------------------------- judge
def test_parse_judge():
    assert parse_judge('{"reason": "B has a bug", "score_a": 9, "score_b": 4}') == (9, 4)
    assert parse_judge('```json\n{"reason":"x","score_a":7,"score_b":7}\n```') == (7, 7)
    assert parse_judge('score_a: 8, score_b: 3') == (8, 3)
    assert parse_judge('{"score_a": 11, "score_b": 3}') is None
    assert parse_judge("no idea") is None


def test_order_is_deterministic_and_balanced():
    ids = [f"q{i}" for i in range(2000)]
    flips = [order_cheap_first(i) for i in ids]
    assert flips == [order_cheap_first(i) for i in ids]          # stable across runs
    assert 0.45 < sum(flips) / len(flips) < 0.55                  # ~half each way


def test_judge_prompt_places_answers_by_order():
    q = {"query": "Q?", "meta": {"reference": "REF"}}
    p1 = build_judge_prompt(q, "CHEAP", "STRONG", cheap_is_a=True, with_reference=True)
    p2 = build_judge_prompt(q, "CHEAP", "STRONG", cheap_is_a=False, with_reference=False)
    assert p1.index("CHEAP") < p1.index("STRONG") and "REF" in p1
    assert p2.index("STRONG") < p2.index("CHEAP") and "REF" not in p2


# ---------------------------------------------------------------- throttle / rate limiting
class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += s


async def test_throttle_rpm_and_tpm():
    clock = FakeClock()
    th = Throttle(rpm=2, tpm=1000, clock=clock, sleep=clock.sleep)
    await th.acquire(100)
    await th.acquire(100)
    assert clock.t == 0
    await th.acquire(100)               # 3rd request in the same minute must wait
    assert clock.t >= 60
    clock.t = 1000
    h = await th.acquire(900)
    th.record_actual(h, 950)            # real usage replaces the estimate
    await th.acquire(100)               # 950 + 100 > 1000 tokens -> wait
    assert clock.t >= 1060


class ScriptedClient:
    name = "fake"

    def __init__(self, script):
        self.script = list(script)

    def is_configured(self):
        return True

    async def chat(self, model, req, timeout_s):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(item, 10, 5, "stop", model, "fake")


async def test_rate_limited_call_retries_then_stops_on_daily_quota():
    clock = FakeClock()
    th = Throttle(100, None, clock=clock, sleep=clock.sleep)
    req = LLMRequest(messages=[{"role": "user", "content": "x"}])
    transient = ProviderError("HTTP 429: slow down", provider="p", status_code=429, retryable=True, retry_after_s=2)
    ok = await call_rate_limited(ScriptedClient([transient, "fine"]), "m", req, th, 10, sleep=clock.sleep)
    assert ok.content == "fine"
    daily = ProviderError("HTTP 429: Rate limit reached on tokens per day (TPD)", provider="p", status_code=429, retryable=True)
    assert is_daily_limit(daily)
    with pytest.raises(DailyLimitReached):
        await call_rate_limited(ScriptedClient([daily]), "m", req, th, 10, sleep=clock.sleep)


def test_parse_try_again():
    from eval.llm_batch import parse_try_again_s

    assert parse_try_again_s("Please try again in 2m9.168s. Need more tokens?") == pytest.approx(129.168)
    assert parse_try_again_s("try again in 1h2m") == 3720
    assert parse_try_again_s("try again in 45.5s") == 45.5
    assert parse_try_again_s("no hint here") is None


async def test_wait_on_daily_limit_sleeps_then_succeeds():
    clock = FakeClock()
    th = Throttle(100, None, clock=clock, sleep=clock.sleep)
    req = LLMRequest(messages=[{"role": "user", "content": "x"}])
    daily = ProviderError("HTTP 429: Rate limit reached on tokens per day (TPD). Please try again in 2m9s.",
                          provider="p", status_code=429, retryable=True)
    # Two daily-quota errors in a row, then success: no DailyLimitReached, no attempt budget used.
    r = await call_rate_limited(ScriptedClient([daily, daily, "ok"]), "m", req, th, 10, max_attempts=1,
                                sleep=clock.sleep, wait_on_daily_limit=True)
    assert r.content == "ok" and clock.t >= 2 * 129


# ---------------------------------------------------------------- collector resume + end-to-end labels
QUERIES = [
    {"id": "gsm8k_1", "source": "gsm8k", "query": "2+2?", "prompt": "2+2? Answer:", "gold": "4", "meta": {}, "split": "train"},
    {"id": "gsm8k_2", "source": "gsm8k", "query": "3*3?", "prompt": "3*3? Answer:", "gold": "9", "meta": {}, "split": "test"},
    {"id": "mmlu_1", "source": "mmlu", "query": "Pick", "prompt": "Pick", "gold": "C", "meta": {}, "split": "val"},
    {"id": "dolly_1", "source": "dolly", "query": "Name a fruit", "prompt": "Name a fruit", "gold": None,
     "meta": {"reference": "Apple"}, "split": "train"},
]
ANSWERS = {  # (id, tier) -> content
    ("gsm8k_1", "cheap"): "Answer: 4", ("gsm8k_1", "strong"): "Answer: 4",
    ("gsm8k_2", "cheap"): "Answer: 6", ("gsm8k_2", "strong"): "Answer: 9",
    ("mmlu_1", "cheap"): "Answer: A", ("mmlu_1", "strong"): "Answer: B",
    ("dolly_1", "cheap"): "Banana", ("dolly_1", "strong"): "An apple is a fruit.",
}


class AnswerClient:
    name = "fake"

    def __init__(self, tier):
        self.tier, self.calls = tier, 0

    def is_configured(self):
        return True

    async def chat(self, model, req, timeout_s):
        self.calls += 1
        prompt = req.messages[-1]["content"]
        qid = next(q["id"] for q in QUERIES if q["prompt"] == prompt)
        return LLMResponse(ANSWERS[(qid, self.tier)], 20, 10, "stop", model, "fake")


@pytest.fixture
def router_paths(tmp_path, monkeypatch):
    qpath = tmp_path / "queries.jsonl"
    qpath.write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    paths = {"QUERIES": qpath, "RESPONSES": tmp_path / "responses.jsonl", "ERRORS": tmp_path / "errors.jsonl"}
    for k, v in paths.items():
        monkeypatch.setattr(collect, k, v)
    for k, v in {"QUERIES": qpath, "RESPONSES": paths["RESPONSES"], "JUDGMENTS": tmp_path / "judgments.jsonl",
                 "LABELS": tmp_path / "labels.jsonl", "REPORT": tmp_path / "router_labels.md",
                 "JUDGE_ERRORS": tmp_path / "judge_errors.jsonl"}.items():
        monkeypatch.setattr(label, k, v)
    return tmp_path


async def test_collector_resumes_without_repeating(router_paths):
    queries = read_jsonl(collect.QUERIES)
    for tier in ("cheap", "strong"):
        client = AnswerClient(tier)
        await collect.collect_tier(tier, "fake", "m", client, collect.pending(queries, tier)[:2], Throttle(1000))
        assert client.calls == 2
    # Second run: only the 2 remaining queries per tier are pending.
    for tier in ("cheap", "strong"):
        todo = collect.pending(queries, tier)
        assert [q["id"] for q in todo] == ["mmlu_1", "dolly_1"]
        await collect.collect_tier(tier, "fake", "m", AnswerClient(tier), todo, Throttle(1000))
        assert collect.pending(queries, tier) == []
    assert len(read_jsonl(collect.RESPONSES)) == 8


async def test_end_to_end_labels(router_paths, monkeypatch):
    queries = read_jsonl(collect.QUERIES)
    for tier in ("cheap", "strong"):
        await collect.collect_tier(tier, "fake", "m", AnswerClient(tier), queries, Throttle(1000))

    class JudgeClient:
        name = "fake"

        def __init__(self, *a, **k):
            pass

        def is_configured(self):
            return True

        async def chat(self, model, req, timeout_s):
            p = req.messages[-1]["content"]
            # Prefer whichever answer mentions "apple" (the reference answer), wherever it is placed.
            a_part = p.split("[Answer A]")[1].split("[Answer B]")[0]
            a_good = "apple" in a_part.lower()
            sa, sb = (9, 3) if a_good else (3, 9)
            return LLMResponse(json.dumps({"reason": "r", "score_a": sa, "score_b": sb}), 50, 20, "stop", model, "fake")

    monkeypatch.setattr(label, "OpenAICompatClient", JudgeClient)
    await label.run_judge("gemini:judge-model", rpm=1000, tpm=None, limit=None)
    label.build(margin=1.0)
    labels = {r["id"]: r for r in read_jsonl(label.LABELS)}
    assert labels["gsm8k_1"]["label"] == 1      # both right
    assert labels["gsm8k_2"]["label"] == 0      # only strong right
    assert labels["mmlu_1"]["label"] == 1       # both wrong
    d = labels["dolly_1"]
    assert d["method"] == "judge" and d["label"] == 0 and d["cheap_quality"] == 0.3 and d["strong_quality"] == 0.9
    report = label.REPORT.read_text()
    assert "Label balance" in report and "gsm8k" in report and "Position bias" in report
