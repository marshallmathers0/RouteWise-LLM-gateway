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
