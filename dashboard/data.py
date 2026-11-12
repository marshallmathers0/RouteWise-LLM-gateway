"""Read request logs and compute dashboard metrics (no Streamlit here, so it is testable)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

STAGES = ("eligibility", "exact_cache", "semantic_cache", "sem_embed", "sem_guards", "sem_verifier", "router", "llm", "total")


def load_logs(engine: Engine, limit: int = 50_000) -> pd.DataFrame:
    with engine.connect() as c:
        df = pd.read_sql(text(f"SELECT * FROM request_logs ORDER BY ts DESC LIMIT {int(limit)}"), c)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    for col in ("latency_ms", "guard_results"):
        if col in df:
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    return df.sort_values("ts")


def summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"requests": 0}
    ok = df[df["status_code"] == 200]
    n = len(ok)
    status = ok["cache_status"].value_counts()
    hits = status.get("exact_hit", 0) + status.get("semantic_hit", 0)
    cost = float(ok["estimated_cost_usd"].sum())
    counterfactual = float(ok["counterfactual_strong_cost_usd"].sum())
    return {
        "requests": int(len(df)),
        "errors": int((df["status_code"] != 200).sum()),
        "fallbacks": int(ok["fallback_used"].sum()),
        "hit_rate": hits / n if n else 0.0,
        "exact_hit_rate": status.get("exact_hit", 0) / n if n else 0.0,
        "semantic_hit_rate": status.get("semantic_hit", 0) / n if n else 0.0,
        "bypass_rate": status.get("bypassed", 0) / n if n else 0.0,
        "cost_usd": cost,
        "counterfactual_strong_usd": counterfactual,
        "saved_frac": 1 - cost / counterfactual if counterfactual else 0.0,
        "route_counts": ok["route"].fillna("cache").value_counts().to_dict(),
    }


def stage_latency(df: pd.DataFrame) -> pd.DataFrame:
    """p50/p95 per pipeline stage (ms), over requests where the stage ran."""
    rows = []
    for stage in STAGES:
        vals = [d[stage] for d in df.get("latency_ms", []) if isinstance(d, dict) and stage in d]
        if vals:
            rows.append({"stage": stage, "p50_ms": float(np.percentile(vals, 50)),
                         "p95_ms": float(np.percentile(vals, 95)), "n": len(vals)})
    return pd.DataFrame(rows)


def semantic_hits(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Recent semantic hits for the manual false-hit audit."""
    if df.empty:
        return df
    hits = df[df["cache_status"] == "semantic_hit"]
    cols = ["ts", "query_text", "matched_query", "similarity", "verifier_score"]
    return hits[cols].sort_values("ts", ascending=False).head(n)
