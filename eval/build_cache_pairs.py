"""Build eval/cache_pairs.jsonl: labeled query pairs for the semantic-cache evaluation.

Sources
-------
* qqp_positive      - human-labeled duplicate pairs from GLUE QQP (validation split).
* qqp_hard_negative - human-labeled NON-duplicate QQP pairs with the HIGHEST embedding
                      similarity under our cache's embedder (sampled from the top-5000 pool,
                      see below). These are the pairs a naive cosine cache would wrongly
                      serve, i.e. "hard negatives". Labels come from QQP's human annotators.
* near_miss         - eval/data/near_miss_pairs.jsonl: LLM-drafted near-misses in five
                      categories + true paraphrases. MUST be verified by a human.

The pair mix is deliberately hard (hard negatives only, no easy random negatives), so
absolute false-hit rates are a worst case, not what typical traffic would see.

Each pair is assigned to "dev" (choose thresholds) or "test" (report numbers), stratified
by source/category/label, so the operating point is never tuned on the reported data.

Run:  python -m eval.build_cache_pairs
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from routewise.cache.models import SentenceTransformerEmbedder
from routewise.core.config import CacheConfig

ROOT = Path(__file__).resolve().parents[1]
NEAR_MISS = ROOT / "eval" / "data" / "near_miss_pairs.jsonl"
OUT = ROOT / "eval" / "cache_pairs.jsonl"
SCORED = ROOT / "data" / "eval" / "qqp_val_scored.jsonl"  # cache of mined cosines (gitignored)


def _clean(q: str) -> str:
    return " ".join(q.split())


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def load_qqp() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    seen, rows = set(), []
    for r in ds:
        q1, q2 = _clean(r["question1"]), _clean(r["question2"])
        if not q1 or not q2 or q1.lower() == q2.lower() or len(q1) > 300 or len(q2) > 300:
            continue
        key = tuple(sorted((q1.lower(), q2.lower())))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"qqp_idx": int(r["idx"]), "q1": q1, "q2": q2, "label": int(r["label"])})
    return rows


def pair_cosines(rows: list[dict], embedder: SentenceTransformerEmbedder, batch: int = 512) -> np.ndarray:
    texts = sorted({r["q1"] for r in rows} | {r["q2"] for r in rows})
    index = {t: i for i, t in enumerate(texts)}
    print(f"embedding {len(texts):,} unique questions ...")
    vecs = np.concatenate([embedder.embed(texts[i:i + batch]) for i in range(0, len(texts), batch)])
    a = vecs[[index[r["q1"]] for r in rows]]
    b = vecs[[index[r["q2"]] for r in rows]]
    return (a * b).sum(axis=1)


def stratified_split(rows: list[dict], rng: random.Random, dev_frac: float = 0.5) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["category"], r["label"])].append(r)
    for members in groups.values():
        rng.shuffle(members)
        n_dev = round(len(members) * dev_frac)
        for i, r in enumerate(members):
            r["split"] = "dev" if i < n_dev else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=1000)
    ap.add_argument("--n-hard-neg", type=int, default=1000)
    ap.add_argument("--hard-neg-pool", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if SCORED.exists():
        qqp = [json.loads(line) for line in SCORED.open(encoding="utf-8")]
        print(f"loaded {len(qqp):,} scored QQP pairs from cache {SCORED.relative_to(ROOT)}")
    else:
        qqp = load_qqp()
        print(f"QQP validation pairs after cleaning: {len(qqp):,}")
        embedder = SentenceTransformerEmbedder(CacheConfig().embedding_model)
        cos = pair_cosines(qqp, embedder)
        for r, c in zip(qqp, cos):
            r["mined_cosine"] = round(float(c), 4)
        SCORED.parent.mkdir(parents=True, exist_ok=True)
        with SCORED.open("w", encoding="utf-8") as f:
            for r in qqp:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    positives = [r for r in qqp if r["label"] == 1]
    # QQP is known to be label-noisy, and the noise concentrates exactly where we mine:
    # the very highest-cosine "non-duplicates" are often typo/punctuation variants that are
    # really duplicates ("this video ?" vs "this video?"). So (1) drop pairs whose
    # normalized text is identical, and (2) SAMPLE from the top-N pool instead of taking
    # the top-k, spreading negatives over the high-similarity band.
    negatives = sorted((r for r in qqp if r["label"] == 0 and _norm(r["q1"]) != _norm(r["q2"])),
                       key=lambda r: -r["mined_cosine"])
    pool = negatives[: args.hard_neg_pool]
    pos_sample = rng.sample(positives, args.n_pos)
    hard_neg = rng.sample(pool, args.n_hard_neg)
    cos_vals = [r["mined_cosine"] for r in hard_neg]
    print(f"hard-negative pool: top {len(pool):,} by cosine ({pool[-1]['mined_cosine']:.3f}..{pool[0]['mined_cosine']:.3f}); "
          f"sampled {len(hard_neg)} (median cosine {np.median(cos_vals):.3f})")

    rows: list[dict] = []
    for r in pos_sample:
        rows.append({"id": f"qqp_{r['qqp_idx']}", "q1": r["q1"], "q2": r["q2"], "label": 1,
                     "source": "qqp_positive", "category": "qqp_duplicate"})
    for r in hard_neg:
        rows.append({"id": f"qqp_{r['qqp_idx']}", "q1": r["q1"], "q2": r["q2"], "label": 0,
                     "source": "qqp_hard_negative", "category": "qqp_hard_negative"})
    with open(NEAR_MISS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: r[k] for k in ("id", "q1", "q2", "label", "source", "category")})

    stratified_split(rows, rng)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        counts[(r["source"], r["label"], r["split"])] += 1
    print(f"wrote {len(rows):,} pairs to {OUT.relative_to(ROOT)}")
    for k in sorted(counts):
        print("  ", k, counts[k])


if __name__ == "__main__":
    main()
