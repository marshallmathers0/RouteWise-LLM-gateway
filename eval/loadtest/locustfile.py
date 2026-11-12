"""Locust load test for the gateway, run against ROUTEWISE_MOCK_PROVIDERS=1 (instant LLM).

Each simulated user mixes: new questions (full pipeline: eligibility, exact cache,
embedding + FAISS + guards + verifier, router, mock LLM, logging) and repeats/paraphrases
(cache hits). We record RouteWise's own per-stage timings from the response metadata, so
"gateway overhead" = total - llm, independent of client/network timing.

Driven by `python -m eval.load_test`, which starts the server and writes the report.
"""

from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("ROUTEWISE_API_KEY", "rw-dev-key")
OUT = os.environ.get("LOADTEST_SAMPLES", "loadtest_samples.jsonl")
TOPICS = ["Python", "Rust", "photosynthesis", "inflation", "black holes", "vaccines", "the Roman Empire",
          "TCP", "neural networks", "climate change", "the French Revolution", "DNA", "blockchain", "jazz"]
TEMPLATES = ["Explain {t} in simple terms.", "What is {t}?", "Give me three facts about {t}.",
             "Why does {t} matter?", "Summarize the history of {t}."]
_samples: list[dict] = []


class GatewayUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def _ask(self, q: str, name: str) -> None:
        with self.client.post("/v1/chat/completions", name=name, catch_response=True,
                              headers={"Authorization": f"Bearer {API_KEY}"},
                              json={"model": "auto", "temperature": 0, "messages": [{"role": "user", "content": q}]}) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            m = r.json()["routewise"]
            lat = m["latency_ms"]
            _samples.append({"kind": name, "cache": m["cache_status"]["status"], "total": lat.get("total", 0.0),
                             "llm": lat.get("llm", 0.0), "semantic": lat.get("semantic_cache", 0.0)})

    @task(3)
    def new_question(self) -> None:
        # Random suffix -> (almost always) a cache miss through the full pipeline.
        q = random.choice(TEMPLATES).format(t=random.choice(TOPICS)) + f" (variant {random.randint(0, 10**9)})"
        self._ask(q, "miss path")

    @task(1)
    def repeated_question(self) -> None:
        self._ask(random.choice(TEMPLATES).format(t=random.choice(TOPICS)), "repeat / hit path")


@events.quitting.add_listener
def _dump(environment, **_):
    with open(OUT, "w", encoding="utf-8") as f:
        for s in _samples:
            f.write(json.dumps(s) + "\n")
