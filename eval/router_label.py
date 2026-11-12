"""Label each router query: 1 = the cheap model's answer is good enough, 0 = pay for strong.

Ground truth (GSM8K, MATH, MMLU-Pro)
  Extract the final answer (number / LaTeX via math-verify / letter) and compare with the
  gold answer. label = 1 if cheap is correct OR both are wrong. ("Both wrong" -> 1
  because paying for the strong model bought nothing.)  quality = 1/0 correctness.

LLM judge (Dolly, MBPP: no automatic answer key)
  A judge from a DIFFERENT model family than both tiers (Qwen judging OpenAI gpt-oss
  answers, to reduce self-preference bias) scores both answers 1-10 with a fixed rubric. The two
  answers appear as "A" and "B" in a random order (seeded per query) to cancel position
  bias. Dataset reference answers are shown as a hint ("reference-guided grading").
  label = 1 if score_cheap >= score_strong - margin (default margin 1).  quality = score/10.

Judge validation
  * vs ground truth: the judge also grades a sample of GSM8K/MMLU answer pairs WITHOUT
    seeing the gold answer; we measure how often it prefers the correct answer.
  * position bias: a sample of open-ended pairs is judged twice with A/B swapped.

  python -m eval.router_label judge   # calls the judge API (resumable)
  python -m eval.router_label build   # no API calls: labels.jsonl + reports/router_labels.md
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from eval.llm_batch import DailyLimitReached, Progress, Throttle, append_jsonl, call_rate_limited, read_jsonl
from routewise.core.config import Settings
from routewise.providers.base import LLMRequest, ProviderError
from routewise.providers.openai_compat import OpenAICompatClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "router"
QUERIES, RESPONSES = DATA / "queries.jsonl", DATA / "responses.jsonl"
JUDGMENTS, LABELS = DATA / "judgments.jsonl", DATA / "labels.jsonl"
JUDGE_ERRORS = DATA / "judge_errors.jsonl"
REPORT = ROOT / "reports" / "router_labels.md"

GROUND_TRUTH_SOURCES = {"gsm8k", "mmlu", "mmlu_pro", "math"}  # graded automatically
JUDGE_SOURCES = {"dolly", "mbpp"}
# Sized for the free tier's daily token budget (each judge call is ~1k tokens).
N_VALIDATION_DISAGREE, N_VALIDATION_AGREE, N_POSITION = 40, 20, 30


# ------------------------------------------------------------------ answer extraction
_NUM = r"-?\$?\s*[\d,]*\.?\d+"


def _num(s: str) -> float | None:
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_gsm8k(text: str) -> float | None:
    """Final numeric answer: last 'Answer: <n>' if present, else the last number in the text."""
    tagged = re.findall(r"answer\s*[:=]?\s*\**\s*(" + _NUM + ")", text, flags=re.I)
    nums = tagged or re.findall(_NUM, text)
    return _num(nums[-1]) if nums else None


def extract_mmlu(text: str, letters: str = "A-D") -> str | None:
    """Final choice letter. MMLU uses A-D, MMLU-Pro uses A-J."""
    tagged = re.findall(r"answer\s*(?:is)?\s*[:=]?\s*\**\s*\(?([" + letters + r"])\b", text, flags=re.I)
    if tagged:
        return tagged[-1].upper()
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.fullmatch(r"\W*\(?([" + letters + r"])\)?\W*", last.strip())
    return m.group(1).upper() if m else None


def math_equivalent(gold_latex: str, answer_text: str) -> bool:
    r"""Symbolic equivalence using Hugging Face's math-verify (\frac12 == 0.5, (3, pi/2) == (3, \frac{\pi}{2})).

    String matching would mark many correct LaTeX answers wrong. math-verify's built-in
    timeout uses multiprocessing, which breaks on Windows, so it is disabled; any parsing
    exception counts as "not correct".
    """
    from math_verify import parse, verify

    try:
        gold = parse("$" + gold_latex + "$", parsing_timeout=None)
        pred = parse(answer_text, parsing_timeout=None)
        return bool(pred) and bool(verify(gold, pred, timeout_seconds=None))
    except Exception:  # noqa: BLE001
        return False


def is_correct(query: dict, content: str) -> bool:
    src = query["source"]
    if src == "gsm8k":
        pred, gold = extract_gsm8k(content), _num(query["gold"])
        return pred is not None and gold is not None and abs(pred - gold) < 1e-6
    if src == "mmlu":
        return extract_mmlu(content) == query["gold"]
    if src == "mmlu_pro":
        return extract_mmlu(content, "A-J") == query["gold"]
    if src == "math":
        return math_equivalent(query["gold"], content)
    raise ValueError(src)


def ground_truth_label(cheap_ok: bool, strong_ok: bool) -> int:
    return int(cheap_ok or not strong_ok)


def judge_label(score_cheap: float, score_strong: float, margin: float) -> int:
    return int(score_cheap >= score_strong - margin)


# ------------------------------------------------------------------ judge prompt/parsing
JUDGE_SYSTEM = (
    "You are a strict, impartial grader of AI assistant answers. Judge each answer on its own merits."
)
JUDGE_TEMPLATE = """Grade two answers to the same user request.

Rubric (score each answer 1-10):
- Correctness is most important: factual errors, wrong results or code that would fail are severe.
- Then: does it fully address the request, and is it clear?
- Do NOT reward length for its own sake. A short correct answer can score 10.
- The order of the answers is random and means nothing.

[User request]
{query}
{reference}
[Answer A]
{a}

[Answer B]
{b}

Reply with ONLY a JSON object: {{"reason": "<one or two sentences>", "score_a": <1-10>, "score_b": <1-10>}}"""


def order_cheap_first(qid: str, salt: str = "") -> bool:
    """Deterministic per-query coin flip (stable across re-runs, unlike random.random())."""
    return hashlib.sha256((qid + salt).encode()).digest()[0] % 2 == 0


def build_judge_prompt(query: dict, cheap: str, strong: str, cheap_is_a: bool, with_reference: bool) -> str:
    ref = ""
    if with_reference and query["meta"].get("reference"):
        ref = f"\n[Reference answer - written by a person; may be imperfect or not the only valid answer]\n{query['meta']['reference']}\n"
    a, b = (cheap, strong) if cheap_is_a else (strong, cheap)
    return JUDGE_TEMPLATE.format(query=query["query"], reference=ref, a=a, b=b)


JUDGE_MAX_TOKENS = 600


def parse_judge(text: str) -> tuple[int, int] | None:
    # The verdict is the LAST valid JSON object: a model may "think out loud" first,
    # and that text can contain braces of its own.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            d = json.loads(blob)
            a, b = int(d["score_a"]), int(d["score_b"])
            if 1 <= a <= 10 and 1 <= b <= 10:
                return a, b
        except (ValueError, KeyError, TypeError):
            continue
    ma = list(re.finditer(r"score_a\D{0,5}(\d+)", text))
    mb = list(re.finditer(r"score_b\D{0,5}(\d+)", text))
    ma, mb = (ma[-1] if ma else None), (mb[-1] if mb else None)
    if ma and mb and 1 <= int(ma.group(1)) <= 10 and 1 <= int(mb.group(1)) <= 10:
        return int(ma.group(1)), int(mb.group(1))
    return None


# ------------------------------------------------------------------ data assembly
def load_pairs() -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    queries = {q["id"]: q for q in read_jsonl(QUERIES)}
    resp: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(RESPONSES):
        resp[r["id"]][r["tier"]] = r  # later rows win (none expected)
    return queries, resp


def judge_jobs(queries: dict, resp: dict) -> list[dict]:
    """All judge calls we want: labels, ground-truth validation, position-swap check."""
    complete = [qid for qid in queries if {"cheap", "strong"} <= resp.get(qid, {}).keys()]
    jobs = []
    for qid in complete:
        if queries[qid]["source"] in JUDGE_SOURCES:
            jobs.append({"id": qid, "kind": "label", "cheap_is_a": order_cheap_first(qid)})
    # Validation: prefer pairs where exactly one model is right (the informative case).
    rng = random.Random(42)
    gt = [qid for qid in complete if queries[qid]["source"] in GROUND_TRUTH_SOURCES]
    dis = [q for q in gt if is_correct(queries[q], resp[q]["cheap"]["content"]) != is_correct(queries[q], resp[q]["strong"]["content"])]
    agr = [q for q in gt if q not in set(dis)]
    for qid in rng.sample(dis, min(N_VALIDATION_DISAGREE, len(dis))) + rng.sample(agr, min(N_VALIDATION_AGREE, len(agr))):
        jobs.append({"id": qid, "kind": "validation", "cheap_is_a": order_cheap_first(qid)})
    # Position bias: same pair, opposite order to its label job.
    open_ended = [qid for qid in complete if queries[qid]["source"] in JUDGE_SOURCES]
    for qid in rng.sample(open_ended, min(N_POSITION, len(open_ended))):
        jobs.append({"id": qid, "kind": "position_swap", "cheap_is_a": not order_cheap_first(qid)})
    return jobs


async def run_judge(judge: str, rpm: float, tpm: float | None, limit: int | None, wait: bool = False) -> None:
    cfg = Settings().load_app_config()
    provider, model = judge.split(":", 1)
    queries, resp = load_pairs()
    done = {(j["id"], j["kind"]) for j in read_jsonl(JUDGMENTS)}
    todo = [j for j in judge_jobs(queries, resp) if (j["id"], j["kind"]) not in done][:limit]
    print(f"judge {judge}: {len(todo)} pending judgments")
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient(provider, cfg.providers[provider], http)
        if not client.is_configured():
            raise SystemExit(f"{cfg.providers[provider].api_key_env} not set")
        throttle, prog = Throttle(rpm, tpm), Progress(len(todo), f"judge:{model}")
        for job in todo:
            q = queries[job["id"]]
            prompt = build_judge_prompt(q, resp[job["id"]]["cheap"]["content"], resp[job["id"]]["strong"]["content"],
                                        job["cheap_is_a"], with_reference=job["kind"] != "validation")
            req = LLMRequest(messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                             temperature=0, max_tokens=JUDGE_MAX_TOKENS)
            try:
                r = await call_rate_limited(client, model, req, throttle, len(prompt) // 4 + JUDGE_MAX_TOKENS,
                                            wait_on_daily_limit=wait)
            except DailyLimitReached as e:
                print(f"judge daily quota reached after {prog.done} - re-run later to resume.\n  {e}")
                return
            except ProviderError as e:
                append_jsonl(JUDGE_ERRORS, {**job, "error": str(e)[:500]})
                prog.tick()
                continue
            scores = parse_judge(r.content)
            if scores is None:
                append_jsonl(JUDGE_ERRORS, {**job, "error": "unparseable", "raw": r.content[:500]})
                prog.tick()
                continue
            a, b = scores
            sc, ss = (a, b) if job["cheap_is_a"] else (b, a)
            append_jsonl(JUDGMENTS, {**job, "judge_model": model, "score_cheap": sc, "score_strong": ss,
                                     "raw": r.content[:1000], "ts": datetime.now(timezone.utc).isoformat()})
            prog.tick()


# ------------------------------------------------------------------ build labels + report
def build(margin: float) -> None:
    queries, resp = load_pairs()
    judgments: dict[tuple[str, str], dict] = {(j["id"], j["kind"]): j for j in read_jsonl(JUDGMENTS)}
    rows, missing = [], defaultdict(int)
    for qid, q in queries.items():
        r = resp.get(qid, {})
        if not {"cheap", "strong"} <= r.keys():
            missing[(q["source"], "no_response")] += 1
            continue
        c, s = r["cheap"], r["strong"]
        row = {"id": qid, "source": q["source"], "split": q["split"],
               "cheap_model": c["model"], "strong_model": s["model"],
               "cheap_tokens": [c["prompt_tokens"], c["completion_tokens"]],
               "strong_tokens": [s["prompt_tokens"], s["completion_tokens"]],
               "cheap_truncated": c["finish_reason"] == "length", "strong_truncated": s["finish_reason"] == "length"}
        if q["source"] in GROUND_TRUTH_SOURCES:
            co, so = is_correct(q, c["content"]), is_correct(q, s["content"])
            row.update(method="ground_truth", cheap_quality=float(co), strong_quality=float(so),
                       label=ground_truth_label(co, so))
        else:
            j = judgments.get((qid, "label"))
            if j is None:
                missing[(q["source"], "no_judgment")] += 1
                continue
            row.update(method="judge", cheap_quality=j["score_cheap"] / 10, strong_quality=j["score_strong"] / 10,
                       label=judge_label(j["score_cheap"], j["score_strong"], margin))
        rows.append(row)
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    write_report(rows, missing, queries, resp, judgments, margin)
    print(f"wrote {len(rows)} labels to {LABELS} and {REPORT}")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(rows, missing, queries, resp, judgments, margin) -> None:
    L: list[str] = []
    w = L.append
    models = {t: next((r[t]["model"] for r in resp.values() if t in r), "?") for t in ("cheap", "strong")}
    judge_models = sorted({j["judge_model"] for j in judgments.values()}) or ["(none yet)"]
    w("# Router labels\n")
    w("_Generated by `python -m eval.router_label build`. All numbers come from saved model outputs._\n")
    w(f"* Cheap model: `{models['cheap']}` - strong model: `{models['strong']}` - judge: `{', '.join(judge_models)}`")
    w(f"* Label = 1 means \"the cheap answer is good enough\". Judge margin = {margin:g}.\n")

    w("## Label balance\n")
    w("| Source | Method | n | label=1 (cheap OK) | cheap quality | strong quality | cheap truncated | strong truncated |")
    w("|---|---|---|---|---|---|---|---|")
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src in ("gsm8k", "math", "mmlu_pro", "mmlu", "dolly", "mbpp"):
        rs = by_src.get(src, [])
        if not rs:
            continue
        n = len(rs)
        w(f"| {src} | {rs[0]['method']} | {n} | {_pct(sum(r['label'] for r in rs) / n)} | "
          f"{sum(r['cheap_quality'] for r in rs) / n:.3f} | {sum(r['strong_quality'] for r in rs) / n:.3f} | "
          f"{sum(r['cheap_truncated'] for r in rs)} | {sum(r['strong_truncated'] for r in rs)} |")
    if rows:
        w(f"| **all** | | {len(rows)} | {_pct(sum(r['label'] for r in rows) / len(rows))} | | | | |")
    w("\nQuality = accuracy for ground-truth sources, mean judge score / 10 for judged sources.\n")
    w("| Split | n | label=1 |")
    w("|---|---|---|")
    for sp in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == sp]
        if rs:
            w(f"| {sp} | {len(rs)} | {_pct(sum(r['label'] for r in rs) / len(rs))} |")
    if missing:
        w("\nMissing (not labeled yet): " + ", ".join(f"{s}/{k}: {v}" for (s, k), v in sorted(missing.items())))

    # Ground-truth breakdown: who is right?
    w("\n## Ground-truth outcomes (GSM8K, MATH, MMLU-Pro)\n")
    w("| Source | both right | only cheap right | only strong right | both wrong |")
    w("|---|---|---|---|---|")
    for src in ("gsm8k", "math", "mmlu_pro", "mmlu"):
        rs = by_src.get(src, [])
        if rs:
            cnt = defaultdict(int)
            for r in rs:
                cnt[(r["cheap_quality"] == 1, r["strong_quality"] == 1)] += 1
            w(f"| {src} | {cnt[(True, True)]} | {cnt[(True, False)]} | {cnt[(False, True)]} | {cnt[(False, False)]} |")

    # Judge validation vs ground truth.
    val = [j for (qid, kind), j in judgments.items() if kind == "validation"]
    w("\n## Judge validation against ground truth\n")
    if val:
        agree = ties = n_dis = 0
        for j in val:
            q = queries[j["id"]]
            co, so = is_correct(q, resp[j["id"]]["cheap"]["content"]), is_correct(q, resp[j["id"]]["strong"]["content"])
            if co == so:
                continue
            n_dis += 1
            correct_score, wrong_score = (j["score_cheap"], j["score_strong"]) if co else (j["score_strong"], j["score_cheap"])
            agree += correct_score > wrong_score
            ties += correct_score == wrong_score
        # Score-vs-correctness AUC over all validated answers (prob. a correct answer outscores a wrong one).
        pos, neg = [], []
        for j in val:
            q = queries[j["id"]]
            for tier, score in (("cheap", j["score_cheap"]), ("strong", j["score_strong"])):
                (pos if is_correct(q, resp[j["id"]][tier]["content"]) else neg).append(score)
        auc = (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))) if pos and neg else float("nan")
        w(f"The judge graded {len(val)} ground-truth (math / multiple-choice) answer pairs **without** the gold answer.\n")
        if n_dis:
            w(f"* Pairs where exactly one answer is correct: **{n_dis}**. The judge scored the correct one higher "
              f"in **{_pct(agree / n_dis)}**, tied in {_pct(ties / n_dis)}, and preferred the wrong one in "
              f"{_pct((n_dis - agree - ties) / n_dis)}.")
        w(f"* AUC of judge score for separating correct from incorrect answers: **{auc:.3f}** "
          f"({len(pos)} correct, {len(neg)} incorrect answers).")
        w("* Math/multiple-choice is a hard case for a judge without the key, so this is a conservative "
          "check of its reliability; open-ended labels also get a human reference answer.")
    else:
        w("_Not run yet._")

    # Position bias.
    w("\n## Position bias check\n")
    swaps = [(judgments[(qid, "label")], j) for (qid, kind), j in judgments.items()
             if kind == "position_swap" and (qid, "label") in judgments]
    if swaps:
        flips = sum((a["score_cheap"] > a["score_strong"]) != (b["score_cheap"] > b["score_strong"]) for a, b in swaps)
        label_flips = sum(judge_label(a["score_cheap"], a["score_strong"], margin) != judge_label(b["score_cheap"], b["score_strong"], margin)
                          for a, b in swaps)
        first_pos = []
        for a, b in swaps:
            for j in (a, b):
                first = j["score_cheap"] if j["cheap_is_a"] else j["score_strong"]
                second = j["score_strong"] if j["cheap_is_a"] else j["score_cheap"]
                first_pos.append(first - second)
        w(f"{len(swaps)} open-ended pairs were judged in both orders.\n")
        w(f"* \"Cheap strictly better\" verdict changed with order in {_pct(flips / len(swaps))} of pairs; "
          f"the final **label** changed in {_pct(label_flips / len(swaps))}.")
        w(f"* Mean score advantage of whichever answer was shown first: {sum(first_pos) / len(first_pos):+.2f} points "
          "(0 = no position bias). Randomizing the order per query cancels this on average.")
    else:
        w("_Not run yet._")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--judge", default="groq:qwen/qwen3.8-27b", help="provider:model")
    j.add_argument("--rpm", type=float, default=25)
    j.add_argument("--tpm", type=float, default=7500)
    j.add_argument("--limit", type=int, default=None)
    j.add_argument("--wait", action="store_true", help="wait out daily quotas instead of stopping")
    b = sub.add_parser("build")
    b.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()
    if args.cmd == "judge":
        asyncio.run(run_judge(args.judge, args.rpm, args.tpm, args.limit, args.wait))
    else:
        build(args.margin)


if __name__ == "__main__":
    main()
