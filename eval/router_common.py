"""Shared loading + cost/quality helpers for router training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.llm_batch import read_jsonl
from routewise.core.config import Settings
from routewise.core.pricing import estimate_cost_usd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "router"
EMB_CACHE = DATA / "query_embeddings.npz"
ARTIFACTS = ROOT / "artifacts"
QUALITY_TARGET = 0.95  # keep >= 95% of always-strong quality


@dataclass
class RouterData:
    ids: list[str]
    queries: list[str]
    sources: np.ndarray
    splits: np.ndarray
    y: np.ndarray               # 1 = cheap is good enough
    q_cheap: np.ndarray         # per-query quality if routed to cheap (0/1 or judge/10)
    q_strong: np.ndarray
    c_cheap: np.ndarray         # per-query estimated USD if routed to cheap
    c_strong: np.ndarray
    method: np.ndarray          # "ground_truth" | "judge"
    models: dict[str, str]

    def subset(self, mask: np.ndarray) -> "RouterData":
        idx = np.flatnonzero(mask)
        return RouterData([self.ids[i] for i in idx], [self.queries[i] for i in idx], self.sources[idx],
                          self.splits[idx], self.y[idx], self.q_cheap[idx], self.q_strong[idx],
                          self.c_cheap[idx], self.c_strong[idx], self.method[idx], self.models)


def load_router_data() -> RouterData:
    pricing = Settings().load_app_config().pricing
    queries = {q["id"]: q for q in read_jsonl(DATA / "queries.jsonl")}
    rows = read_jsonl(DATA / "labels.jsonl")
    if not rows:
        raise SystemExit("no labels: run `make collect` and `make label` first")
    models = {"cheap": rows[0]["cheap_model"], "strong": rows[0]["strong_model"]}
    for t, m in models.items():
        if m not in pricing:
            raise SystemExit(f"no price for {t} model {m} in config pricing")

    def cost(model: str, toks: list[int]) -> float:
        return estimate_cost_usd(pricing, model, toks[0], toks[1])

    return RouterData(
        ids=[r["id"] for r in rows],
        queries=[queries[r["id"]]["query"] for r in rows],
        sources=np.array([r["source"] for r in rows]),
        splits=np.array([r["split"] for r in rows]),
        y=np.array([r["label"] for r in rows], dtype=int),
        q_cheap=np.array([r["cheap_quality"] for r in rows], dtype=float),
        q_strong=np.array([r["strong_quality"] for r in rows], dtype=float),
        c_cheap=np.array([cost(r["cheap_model"], r["cheap_tokens"]) for r in rows]),
        c_strong=np.array([cost(r["strong_model"], r["strong_tokens"]) for r in rows]),
        method=np.array([r["method"] for r in rows]),
        models=models,
    )


def embeddings_for(data: RouterData, embedder) -> np.ndarray:
    """Query embeddings, cached on disk keyed by query id (embedding is the slow part)."""
    cache: dict[str, np.ndarray] = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["emb"]))
    missing = [i for i, qid in enumerate(data.ids) if qid not in cache]
    if missing:
        vecs = embedder.embed([data.queries[i] for i in missing])
        for i, v in zip(missing, vecs):
            cache[data.ids[i]] = v
        np.savez(EMB_CACHE, ids=np.array(list(cache)), emb=np.stack(list(cache.values())))
    return np.stack([cache[qid] for qid in data.ids])


def route_outcome(route_cheap: np.ndarray, d: RouterData) -> tuple[float, float]:
    """(mean quality, total cost) when routing query i to cheap iff route_cheap[i]."""
    q = np.where(route_cheap, d.q_cheap, d.q_strong).mean()
    c = np.where(route_cheap, d.c_cheap, d.c_strong).sum()
    return float(q), float(c)


def sweep(scores: np.ndarray, d: RouterData) -> list[dict]:
    """Cost/quality for every threshold on `scores` (higher score => route to cheap)."""
    out = []
    for t in np.unique(np.concatenate([scores, [np.inf]])):
        cheap = scores >= t
        q, c = route_outcome(cheap, d)
        out.append({"threshold": float(t), "quality": q, "cost": c, "cheap_frac": float(cheap.mean())})
    return out


def pick_threshold(scores: np.ndarray, d: RouterData, target: float = QUALITY_TARGET) -> float:
    """Lowest-cost threshold whose quality >= target x always-strong quality (on data d)."""
    q_strong = d.q_strong.mean()
    ok = [p for p in sweep(scores, d) if p["quality"] >= target * q_strong]
    if not ok:  # unreachable target: route everything to strong
        return float("inf")
    return min(ok, key=lambda p: (p["cost"], -p["quality"]))["threshold"]
