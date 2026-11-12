"""Replay a realistic query mix through the gateway so the dashboard has data.

The mix: seed questions, later paraphrases of them (should become semantic hits),
exact repeats (exact hits, if Redis is up), near-miss traps (must NOT hit), a few
time-sensitive and high-temperature requests (bypassed), and held-out router test
queries (exercise easy-vs-hard routing).

    python scripts/simulate_traffic.py --n 120 --rpm 20
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [
    ("What is the capital of India?", ["Which city is India's capital?", "India's capital city?"],
     ["What is the capital of Indonesia?"]),
    ("How do I reverse a list in Python?", ["Python: how to reverse a list?", "What's the way to reverse a Python list?"],
     ["How do I reverse a string in Python?"]),
    ("What are the main new features in Python 3.10?", ["What's new in Python 3.10?"],
     ["What are the main new features in Python 3.12?"]),
    ("Convert 5 miles to km", ["How many kilometers is 5 miles?"], ["Convert 5 km to miles"]),
    ("Is coffee bad for your health?", ["Is coffee unhealthy?"], ["Is coffee not bad for your health?"]),
    ("Who wrote Pride and Prejudice?", ["Who is the author of Pride and Prejudice?"], ["Who wrote Jane Eyre?"]),
    ("How do vaccines work?", ["How does a vaccine work in the body?"], ["How do antibiotics work?"]),
    ("What causes earthquakes?", ["Why do earthquakes happen?"], ["What causes volcanoes to erupt?"]),
    ("How do I center a div in CSS?", ["CSS: how to center a div?"], ["How do I center a div vertically in CSS?"]),
    ("What is the boiling point of water?", ["At what temperature does water boil?"],
     ["What is the freezing point of water?"]),
]
BYPASS = ["What is the latest news about AI today?", "What's the weather in Mumbai right now?"]


def build_mix(n: int, rng: random.Random) -> list[tuple[str, float]]:
    """(question, temperature) in a plausible arrival order: seeds before their variants."""
    test_queries = []
    qpath = ROOT / "data" / "router" / "queries.jsonl"
    if qpath.exists():
        test_queries = [json.loads(line)["query"] for line in qpath.open(encoding="utf-8")
                        if json.loads(line)["split"] == "test"]
    mix: list[tuple[str, float]] = [(s, 0.0) for s, _, _ in SEEDS]
    later: list[tuple[str, float]] = []
    for s, paras, traps in SEEDS:
        later += [(p, 0.0) for p in paras] + [(t, 0.0) for t in traps] + [(s, 0.0)]  # repeat -> exact hit
    later += [(b, 0.0) for b in BYPASS] + [(SEEDS[0][0], 0.9)]  # high temperature -> bypass
    rng.shuffle(later)
    mix += later
    rng.shuffle(test_queries)
    mix += [(q, 0.0) for q in test_queries[: max(0, n - len(mix))]]
    return mix[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="rw-dev-key")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--rpm", type=float, default=20, help="stay under free-tier provider limits")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    mix = build_mix(args.n, random.Random(args.seed))
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(timeout=180) as client:
        for i, (q, temp) in enumerate(mix, 1):
            t0 = time.time()
            r = client.post(f"{args.url}/v1/chat/completions", headers=headers,
                            json={"model": "auto", "temperature": temp, "max_tokens": 600,
                                  "messages": [{"role": "user", "content": q}]})
            if r.status_code == 200:
                m = r.json()["routewise"]
                print(f"[{i}/{len(mix)}] {m['cache_status']['status']:12s} route={m['route'] or '-':6s} "
                      f"{m['latency_ms'].get('total', 0):7.0f}ms  {q[:60]!r}")
            else:
                print(f"[{i}/{len(mix)}] HTTP {r.status_code}  {q[:60]!r}")
            time.sleep(max(0.0, 60 / args.rpm - (time.time() - t0)))


if __name__ == "__main__":
    main()
