"""Build the router query set: data/router/queries.jsonl.

Pool (1,000, mixed difficulty on purpose), saved to queries_full.jsonl:
  easy   - GSM8K 150 (grade-school math), Dolly 250 (everyday instructions), MBPP 150 (simple Python)
  hard   - MATH 200 (competition math, levels 3-5), MMLU-Pro 250 (10-option expert questions)
Used set (default: a stratified 50% = 500 queries), saved to queries.jsonl. Groq's free tier
allows 200k tokens per model per day and the full pool needs ~540k per model, so 500
queries keeps collection to roughly a day.
Why not the original plan (MMLU, 1,500 queries)? A pilot showed the cheap model
(gpt-oss-20b) already near-perfect on GSM8K and MMLU, so almost every label would be
"cheap is fine" and a router would have nothing to learn. And the free tier allows
~1,000 requests/model/day, so 1,000 queries per tier fits a single day's quota.
Queries used in that pilot (data/router/pilot_ids.json) are excluded.

Two text fields per query:
* `query`  - what a user would type. The ROUTER only ever sees this.
* `prompt` - `query` + an answer-format instruction we add so answers can be graded
             automatically ("...last line as 'Answer: <number>'"). It goes to the LLMs.
Keeping them separate matters: the format suffix is a fixed per-source template, so a
router trained on `prompt` could recognise the source from it - leaking the label just
like using `source` as a feature would.

Split: stratified by source, 70/15/15 train/val/test (seed 42). Stratifying (rather than
holding out whole sources) keeps every query type in every split: with only four sources,
a leave-one-source-out split would test on a domain the router has never seen, which is
a different (harder) question - reported separately in Phase 5 if time allows.

Run:  python -m eval.router_queries
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "router" / "queries.jsonl"

MMLU_SUBJECTS = [  # chosen to span easy -> hard for an 8B model
    "high_school_geography", "high_school_biology", "elementary_mathematics",
    "high_school_macroeconomics", "college_physics", "formal_logic",
]
DOLLY_MAX_CONTEXT = 1500  # chars; keeps token cost bounded
LETTERS = "ABCDEFGHIJ"

GSM8K_SUFFIX = "\n\nSolve the problem step by step. On the last line, write the final answer as 'Answer: <number>'."
MMLU_SUFFIX = "\n\nThink briefly, then on the last line write the final answer as 'Answer: <letter>'."


def gsm8k(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        q = r["question"].strip()
        out.append({"id": f"gsm8k_{i}", "source": "gsm8k", "query": q, "prompt": q + GSM8K_SUFFIX,
                    "gold": gold, "meta": {}})
    return out


def mmlu(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    per = [n // len(MMLU_SUBJECTS) + (1 if k < n % len(MMLU_SUBJECTS) else 0) for k in range(len(MMLU_SUBJECTS))]
    out = []
    for subject, k in zip(MMLU_SUBJECTS, per):
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i in rng.sample(range(len(ds)), min(k, len(ds))):
            r = ds[i]
            q = r["question"].strip() + "\n\n" + "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["choices"]))
            out.append({"id": f"mmlu_{subject}_{i}", "source": "mmlu", "query": q, "prompt": q + MMLU_SUFFIX,
                        "gold": LETTERS[r["answer"]], "meta": {"subject": subject}})
    return out


MATH_SUFFIX = "\n\nSolve the problem. Put your final answer in \\boxed{}."
MATH_LEVELS = (3, 4, 5)  # competition levels 1-5; 1-2 are close to GSM8K difficulty


def math_competition(n: int, rng: random.Random, exclude: set[str] = frozenset()) -> list[dict]:
    """MATH (Hendrycks et al.) competition problems via the MATH-500 subset; LaTeX answers."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    pool = [i for i, r in enumerate(ds) if r["level"] in MATH_LEVELS and f"math_{i}" not in exclude]
    out = []
    for i in rng.sample(pool, min(n, len(pool))):
        r = ds[i]
        q = r["problem"].strip()
        out.append({"id": f"math_{i}", "source": "math", "query": q, "prompt": q + MATH_SUFFIX,
                    "gold": r["answer"], "meta": {"level": r["level"], "subject": r["subject"]}})
    return out


def mmlu_pro(n: int, rng: random.Random, exclude: set[str] = frozenset()) -> list[dict]:
    """MMLU-Pro: harder MMLU variant with up to 10 options (A-J) across 14 subject areas."""
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    pool = [i for i in range(len(ds)) if f"mmlu_pro_{ds[i]['question_id']}" not in exclude]
    out = []
    for i in rng.sample(pool, n):
        r = ds[i]
        q = r["question"].strip() + "\n\n" + "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["options"]))
        out.append({"id": f"mmlu_pro_{r['question_id']}", "source": "mmlu_pro", "query": q, "prompt": q + MMLU_SUFFIX,
                    "gold": r["answer"], "meta": {"category": r["category"]}})
    return out


def dolly(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(ds):
        if len(r["context"]) <= DOLLY_MAX_CONTEXT and r["instruction"].strip():
            by_cat[r["category"]].append(i)
    cats = sorted(by_cat)
    per = [n // len(cats) + (1 if k < n % len(cats) else 0) for k in range(len(cats))]
    out = []
    for cat, k in zip(cats, per):
        for i in rng.sample(by_cat[cat], k):
            r = ds[i]
            q = r["instruction"].strip()
            if r["context"].strip():
                q += "\n\n" + r["context"].strip()
            # Dolly's human reference answer is kept for the judge's context, never as a label.
            out.append({"id": f"dolly_{i}", "source": "dolly", "query": q, "prompt": q, "gold": None,
                        "meta": {"category": cat, "reference": r["response"]}})
    return out


def mbpp(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        q = f"{r['text'].strip()}\nYour code should pass this test:\n{r['test_list'][0]}"
        out.append({"id": f"mbpp_{r['task_id']}", "source": "mbpp", "query": q,
                    "prompt": q + "\n\nReturn the Python function in a code block.", "gold": None,
                    "meta": {"tests": r["test_list"], "reference": r["code"]}})
    return out


def assign_splits(rows: list[dict], rng: random.Random, fracs=(0.70, 0.15, 0.15)) -> None:
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for members in by_src.values():
        rng.shuffle(members)
        n_train = round(len(members) * fracs[0])
        n_val = round(len(members) * fracs[1])
        for i, r in enumerate(members):
            r["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", type=int, default=150)
    ap.add_argument("--math", type=int, default=200)
    ap.add_argument("--mmlu-pro", type=int, default=250)
    ap.add_argument("--mmlu", type=int, default=0, help="original MMLU (too easy for gpt-oss; off by default)")
    ap.add_argument("--dolly", type=int, default=250)
    ap.add_argument("--mbpp", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-fraction", type=float, default=0.5,
                    help="downsample the 1,000-query pool (stratified by source and split); see module docstring")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    # Queries used in the model-selection pilot never enter this dataset.
    pilot_path = OUT.parent / "pilot_ids.json"
    exclude = set(json.loads(pilot_path.read_text())) if pilot_path.exists() else set()

    rows = (gsm8k(args.gsm8k, rng) + math_competition(args.math, rng, exclude) + mmlu_pro(args.mmlu_pro, rng, exclude)
            + (mmlu(args.mmlu, rng) if args.mmlu else []) + dolly(args.dolly, rng) + mbpp(args.mbpp, rng))
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    assign_splits(rows, rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with (OUT.parent / "queries_full.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Free-tier budget: Groq allows 200k tokens/model/day and the full pool needs ~540k
    # per model, so by default we keep a stratified half. Selection is random within each
    # (source, split) cell and never looks at model answers, so it can't bias the labels.
    if args.keep_fraction < 1:
        cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            cells[(r["source"], r["split"])].append(r)
        sub_rng = random.Random(args.seed + 1)
        keep = {r["id"] for members in cells.values()
                for r in sub_rng.sample(members, max(1, round(len(members) * args.keep_fraction)))}
        rows = [r for r in rows if r["id"] in keep]
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows):,} queries to {OUT.relative_to(ROOT)}")
    print("by source/split:", dict(sorted(Counter((r["source"], r["split"]) for r in rows).items())))
    chars = [len(r["prompt"]) for r in rows]
    print(f"prompt length: mean {sum(chars) / len(chars):.0f} chars, max {max(chars)}")


if __name__ == "__main__":
    main()
