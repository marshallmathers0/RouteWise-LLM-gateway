"""Dashboard metrics computed from request logs written by the real gateway."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.data import load_logs, semantic_hits, stage_latency, summary
from routewise.api.app import create_app
from routewise.router.baselines import LengthHeuristicRouter
from tests.conftest import AUTH, chat_body


def test_dashboard_metrics_from_real_logs(settings, app_config, fake_clients, fake_redis, repo):
    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=fake_redis, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=None)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json=chat_body("q1"), headers=AUTH)
        c.post("/v1/chat/completions", json=chat_body("q1"), headers=AUTH)          # exact hit
        c.post("/v1/chat/completions", json=chat_body("q2", temperature=0.9), headers=AUTH)  # bypassed
    df = load_logs(repo.engine)
    assert len(df) == 3
    s = summary(df)
    assert s["requests"] == 3 and s["errors"] == 0
    assert abs(s["exact_hit_rate"] - 1 / 3) < 1e-9 and abs(s["bypass_rate"] - 1 / 3) < 1e-9
    assert 0 < s["saved_frac"] < 1  # cheap route + a free hit cost less than always-strong
    lat = stage_latency(df)
    assert {"total", "llm"} <= set(lat["stage"]) and (lat["p95_ms"] >= lat["p50_ms"]).all()
    assert semantic_hits(df).empty


def test_summary_of_empty_logs():
    import pandas as pd

    assert summary(pd.DataFrame()) == {"requests": 0}


def test_savings_breakdown_adds_up_and_isolates_fallback_overhead():
    import pandas as pd

    df = pd.DataFrame([
        # cache hit: saves its whole always-strong cost
        {"status_code": 200, "cache_status": "semantic_hit", "fallback_used": False, "route": "cheap",
         "estimated_cost_usd": 0.0, "counterfactual_strong_cost_usd": 1.0},
        # routed to the cheap first-choice model at half the strong price
        {"status_code": 200, "cache_status": "miss", "fallback_used": False, "route": "cheap",
         "estimated_cost_usd": 0.5, "counterfactual_strong_cost_usd": 1.0},
        # fallback to a backup 4x pricier than the strong model
        {"status_code": 200, "cache_status": "miss", "fallback_used": True, "route": "cheap",
         "estimated_cost_usd": 4.0, "counterfactual_strong_cost_usd": 1.0},
    ])
    s = summary(df)
    assert s["saved_by_cache_usd"] == 1.0 and s["saved_by_routing_usd"] == 0.5
    assert s["fallback_overhead_usd"] == 3.0
    total_saved = s["counterfactual_strong_usd"] - s["cost_usd"]
    assert total_saved == s["saved_by_cache_usd"] + s["saved_by_routing_usd"] - s["fallback_overhead_usd"]
    assert s["saved_frac"] < 0  # overall negative, and the breakdown shows it is the fallback
