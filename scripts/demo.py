"""Command-line demo of RouteWise using the official OpenAI SDK.

Start the gateway first (see README), then:
    python scripts/demo.py                      # against http://localhost:8000
    python scripts/demo.py --base-url http://host:8000/v1 --api-key rw-dev-key

It sends a paraphrase (should be a semantic cache hit) and the "trap" near-misses
(should NOT be served from cache), and prints RouteWise's routing/cache metadata.
"""

from __future__ import annotations

import argparse
import textwrap

from openai import OpenAI

SCRIPT = [
    ("seed", "What is the capital of India?"),
    ("paraphrase -> expect semantic_hit", "Which city is India's capital?"),
    ("trap: entity swap -> expect miss", "What is the capital of Indonesia?"),
    ("seed", "What are the main new features in Python 3.10?"),
    ("trap: version change -> expect miss", "What are the main new features in Python 3.12?"),
    ("seed", "Convert 5 miles to km"),
    ("trap: direction swap -> expect miss", "Convert 5 km to miles"),
    ("time-sensitive -> expect bypassed", "What is the latest news about AI today?"),
    ("hard query -> router may pick strong",
     "Prove that the square root of 2 is irrational, step by step."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="rw-dev-key")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0, timeout=120)

    for expectation, question in SCRIPT:
        try:
            resp = client.chat.completions.create(
                # Generous cap: truncated answers are never cached, so a cut-off seed answer
                # would leave its trap with nothing to be compared against.
                model="auto", temperature=0, max_tokens=2000,
                messages=[{"role": "user", "content": question}],
            )
        except Exception as e:  # noqa: BLE001 - keep the demo going and show the gateway's error
            print("=" * 100)
            print(f"Q: {question}   [{expectation}]")
            print(f"   ERROR from gateway: {str(e)[:300]}")
            continue
        m = resp.model_extra["routewise"]
        cs = m["cache_status"]
        print("=" * 100)
        print(f"Q: {question}   [{expectation}]")
        print(f"   cache: {cs['status']}" + (f" ({cs['reason']})" if cs.get("reason") else ""))
        if m.get("matched_query"):
            ver = m.get("verifier_score")
            print(f"   nearest cached: {m['matched_query']!r}  cosine={m['similarity']}"
                  + (f"  verifier={ver}" if ver is not None else ""))
        print(f"   route: {m['route']} ({m['route_reason']})  model: {m['model_used']}  provider: {m['provider_used']}"
              + ("  [FALLBACK]" if m["fallback_used"] else ""))
        lat = ", ".join(f"{k}={v:.0f}ms" for k, v in m["latency_ms"].items())
        print(f"   latency: {lat}")
        print(f"   estimated cost: ${m['estimated_cost_usd']:.6f}")
        answer = (resp.choices[0].message.content or "").strip().replace("\n", " ")
        print("   answer: " + textwrap.shorten(answer, 160))


if __name__ == "__main__":
    main()
