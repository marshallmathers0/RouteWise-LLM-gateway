"""Router features (no label leakage), learned-router loading/decisions, gateway wiring."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from routewise.api.app import build_router, create_app
from routewise.router.baselines import LengthHeuristicRouter
from routewise.router.features import FORBIDDEN_FEATURES, HANDCRAFTED_NAMES, RouterFeaturizer, handcrafted
from routewise.router.learned import LearnedRouter
from tests.conftest import AUTH, chat_body


class TinyEmbedder:
    dim = 4

    def embed(self, texts):
        return np.array([[len(t) % 7, t.count(" "), float("?" in t), 1.0] for t in texts], dtype=np.float32)


def test_handcrafted_features_shape_and_signals():
    plain = handcrafted("What is the capital of France?")
    code = handcrafted("def add(a, b):\n    return a + b\nassert add(1, 2) == 3")
    mc = handcrafted("Which is a mammal?\n\nA. shark\nB. whale\nC. trout\nD. eel")
    names = list(HANDCRAFTED_NAMES)
    assert plain.shape == (len(names),)
    assert code[names.index("has_code")] == 1 and plain[names.index("has_code")] == 0
    assert mc[names.index("n_options")] == 4
    assert plain[names.index("qw_what")] == 1


def test_no_label_leaking_features():
    feat = RouterFeaturizer(TinyEmbedder())
    names = feat.feature_names
    assert not FORBIDDEN_FEATURES & set(names)
    assert not any(n in names for n in ("source", "split", "label"))
    # Features depend only on the query text: same text -> same vector regardless of metadata.
    a = feat.transform(["Solve 2+2"])
    b = feat.transform(["Solve 2+2"])
    assert np.array_equal(a, b) and a.shape == (1, len(names))


def test_handcrafted_only_mode_needs_no_embedder():
    feat = RouterFeaturizer(None, use_embedding=False)
    assert feat.transform(["hi"]).shape == (1, len(HANDCRAFTED_NAMES))
    with pytest.raises(ValueError):
        RouterFeaturizer(None, use_embedding=True)


@pytest.fixture
def artifact(tmp_path):
    # A toy model: long queries -> "needs strong" (label 0), short -> "cheap ok" (label 1).
    emb = TinyEmbedder()
    feat = RouterFeaturizer(emb)
    short = ["hi there?", "capital of France?", "2+2?", "name a color"]
    long = ["Prove that " + "x " * 60, "Write a full compiler in " + "y " * 70,
            "Derive the formula " + "z " * 50, "Explain in depth " + "w " * 80]
    X = feat.transform(short + long)
    model = LogisticRegression(max_iter=2000).fit(X, [1, 1, 1, 1, 0, 0, 0, 0])
    path = tmp_path / "router.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "model_name": "toy-logreg", "use_embedding": True,
                 "metadata": {"test": True}}, path)
    return path, emb


def test_learned_router_decides(artifact):
    path, emb = artifact
    r = LearnedRouter.load(path, emb)
    easy, hard = r.decide("capital of Spain?"), r.decide("Prove that " + "q " * 70)
    assert easy.tier == "cheap" and easy.prob_cheap_ok > 0.5 and easy.reason == "learned:toy-logreg"
    assert hard.tier == "strong" and hard.prob_cheap_ok < 0.5


def test_build_router_falls_back_to_heuristic(app_config, tmp_path):
    app_config.router.artifact_path = str(tmp_path / "missing.joblib")
    assert isinstance(build_router(app_config, TinyEmbedder()), LengthHeuristicRouter)
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    app_config.router.artifact_path = str(bad)
    assert isinstance(build_router(app_config, TinyEmbedder()), LengthHeuristicRouter)


def test_gateway_uses_learned_router(settings, app_config, fake_clients, fake_redis, repo, artifact):
    path, emb = artifact
    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=fake_redis, repo=repo,
                     router=LearnedRouter.load(path, emb), semantic_cache=None)
    with TestClient(app) as c:
        meta = c.post("/v1/chat/completions", json=chat_body("capital of Spain?"), headers=AUTH).json()["routewise"]
        assert meta["route"] == "cheap" and meta["route_reason"] == "learned:toy-logreg"
        assert 0.5 < meta["router_prob"] <= 1
        hard = c.post("/v1/chat/completions", json=chat_body("Prove that " + "q " * 70), headers=AUTH).json()["routewise"]
        assert hard["route"] == "strong"


def test_mock_provider_mode(settings, app_config, fake_redis, repo, monkeypatch):
    monkeypatch.setenv("ROUTEWISE_MOCK_PROVIDERS", "1")
    app = create_app(settings, app_config=app_config, redis=fake_redis, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=None)
    with TestClient(app) as c:
        body = c.post("/v1/chat/completions", json=chat_body(), headers=AUTH).json()
        assert body["choices"][0]["message"]["content"].startswith("[mock groq/")
