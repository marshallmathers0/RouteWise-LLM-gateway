"""Semantic cache orchestration (fake models), store behaviour, and API integration."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import routewise.api.app as app_module
from routewise.router.baselines import LengthHeuristicRouter
from routewise.api.app import build_semantic_cache, create_app
from routewise.cache.semantic import SemanticCache
from routewise.cache.store import FaissSqliteStore
from routewise.core.config import CacheConfig
from tests.conftest import AUTH, chat_body

DIM = 8


def unit(*xs: float) -> np.ndarray:
    v = np.array(list(xs) + [0.0] * (DIM - len(xs)), dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Returns hand-picked vectors so similarities in tests are exact and readable."""

    dim = DIM

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed(self, texts):
        return np.stack([self.vectors.get(t, unit(0, 0, 0, 0, 0, 0, 0, 1)) for t in texts])


class FakeVerifier:
    def __init__(self, scores: dict[tuple[str, str], float], default: float = 0.99) -> None:
        self.scores = scores
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def score(self, pairs):
        self.calls.extend(pairs)
        return [self.scores.get(tuple(p), self.default) for p in pairs]


V = {
    "What is the capital of France?": unit(1, 0),
    "Which city is the capital of France?": unit(1, 0.2),        # cos ~0.98 paraphrase
    "What is the capital of Germany?": unit(1, 0.25),            # cos ~0.97 near-miss
    "Python 3.10 features": unit(0, 1),
    "Python 3.12 features": unit(0.05, 1),
    "How do volcanoes form?": unit(0, 0, 1),
    "Why do volcanoes erupt?": unit(0, 0.3, 1),                  # cos ~0.96, verifier says no
    "Tell me a joke": unit(0.5, 0.5, 0.5),                        # unrelated
}


def make_cache(tmp_path, verifier=None, **cfg_kw) -> SemanticCache:
    cfg = CacheConfig(semantic_db_path=str(tmp_path / "sem.db"), **cfg_kw)
    verifier = verifier if verifier is not None else FakeVerifier({})
    return SemanticCache(cfg, FakeEmbedder(V), FaissSqliteStore(cfg.semantic_db_path, DIM),
                         verifier if cfg.verifier_enabled else None)


def seed(cache: SemanticCache, query: str, ns: str = "ns") -> None:
    cache.add(ns, query, cache.embedder.embed([query])[0], f"ANSWER({query})", "m", "p", 1, 1, "cheap")


def test_paraphrase_hits(tmp_path):
    c = make_cache(tmp_path)
    seed(c, "What is the capital of France?")
    r = c.lookup("Which city is the capital of France?", "ns")
    assert r.hit is not None and r.hit.answer == "ANSWER(What is the capital of France?)"
    assert r.best.accepted and r.best.similarity > 0.95
    assert {"sem_embed", "sem_search", "sem_guards", "sem_verifier"} <= r.latency_ms.keys()


def test_entity_guard_blocks_before_verifier(tmp_path):
    v = FakeVerifier({}, default=0.99)  # a gullible verifier that would accept anything
    c = make_cache(tmp_path, verifier=v)
    seed(c, "What is the capital of France?")
    r = c.lookup("What is the capital of Germany?", "ns")
    assert r.hit is None
    assert r.candidates[0].rejected_by == "guard:entities"
    assert v.calls == []  # rejected pairs never reach the (expensive) verifier


def test_number_guard_blocks_version_change(tmp_path):
    c = make_cache(tmp_path, verifier=FakeVerifier({}, default=0.99))
    seed(c, "Python 3.10 features")
    r = c.lookup("Python 3.12 features", "ns")
    assert r.hit is None and r.best.rejected_by == "guard:numbers"


def test_verifier_threshold_logic(tmp_path):
    pair = ("Why do volcanoes erupt?", "How do volcanoes form?")
    for score, expect_hit in [(0.49, False), (0.5, True), (0.51, True)]:
        c = make_cache(tmp_path / str(score), verifier=FakeVerifier({pair: score}), verifier_threshold=0.5)
        seed(c, "How do volcanoes form?")
        r = c.lookup("Why do volcanoes erupt?", "ns")
        assert (r.hit is not None) is expect_hit, score
        assert r.best.verifier_score == score
        if not expect_hit:
            assert r.best.rejected_by == "verifier"


def test_similarity_threshold_filters_candidates(tmp_path):
    c = make_cache(tmp_path)
    seed(c, "What is the capital of France?")
    r = c.lookup("Tell me a joke", "ns")
    assert r.hit is None and r.candidates == []


def test_ablation_switches(tmp_path):
    # Cosine only: the near-miss gets served (this is the naive cache's failure mode).
    c = make_cache(tmp_path, guards_enabled=False, verifier_enabled=False)
    seed(c, "What is the capital of France?")
    assert c.lookup("What is the capital of Germany?", "ns").hit is not None


def test_namespaces_are_isolated(tmp_path):
    c = make_cache(tmp_path)
    seed(c, "What is the capital of France?", ns="pirate")
    assert c.lookup("What is the capital of France?", "lawyer").hit is None
    assert c.lookup("What is the capital of France?", "pirate").hit is not None


def test_verifier_required_when_enabled(tmp_path):
    cfg = CacheConfig(semantic_db_path=str(tmp_path / "x.db"))
    with pytest.raises(ValueError):
        SemanticCache(cfg, FakeEmbedder(V), FaissSqliteStore(cfg.semantic_db_path, DIM), None)


def test_store_persists_and_expires(tmp_path):
    now = [1000.0]
    db = tmp_path / "persist.db"
    store = FaissSqliteStore(db, DIM, clock=lambda: now[0])
    store.add("ns", "q1", unit(1), "a1", "m", "p", 1, 1, None, ttl_s=100)
    store.add("ns", "q2", unit(0, 1), "a2", "m", "p", 1, 1, None, ttl_s=10)
    # Restart: the FAISS index is rebuilt from SQLite.
    store2 = FaissSqliteStore(db, DIM, clock=lambda: now[0])
    assert len(store2) == 2
    assert store2.search("ns", unit(1), 1)[0][0].answer == "a1"
    now[0] += 50  # q2 expired, q1 still valid
    assert store2.search("ns", unit(0, 1), 2)[0][0].query == "q1"
    assert len(store2) == 1  # expired entry lazily removed
    assert len(FaissSqliteStore(db, DIM, clock=lambda: now[0])) == 1


# ---- API integration ------------------------------------------------------------------
@pytest.fixture
def sem_client(settings, app_config, fake_clients, fake_redis, repo, tmp_path):
    cache = make_cache(tmp_path)
    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=fake_redis, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=cache)
    with TestClient(app) as c:
        yield c


def test_api_semantic_hit_and_trap(sem_client, fake_clients):
    first = sem_client.post("/v1/chat/completions", json=chat_body("What is the capital of France?"), headers=AUTH).json()
    assert first["routewise"]["cache_status"]["status"] == "miss"

    para = sem_client.post("/v1/chat/completions", json=chat_body("Which city is the capital of France?"), headers=AUTH).json()
    meta = para["routewise"]
    assert meta["cache_status"]["status"] == "semantic_hit"
    assert meta["matched_query"] == "What is the capital of France?"
    assert meta["similarity"] > 0.95 and meta["verifier_score"] is not None
    assert meta["estimated_cost_usd"] == 0
    assert para["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]

    trap = sem_client.post("/v1/chat/completions", json=chat_body("What is the capital of Germany?"), headers=AUTH).json()
    tm = trap["routewise"]
    assert tm["cache_status"] == {"status": "miss", "reason": "semantic_rejected:guard:entities"}
    assert tm["matched_query"] == "What is the capital of France?"  # shown for debugging/audit
    assert len(fake_clients["groq"].calls) == 2  # France + Germany; the paraphrase was free


def test_api_multi_turn_skips_semantic(sem_client):
    sem_client.post("/v1/chat/completions", json=chat_body("What is the capital of France?"), headers=AUTH)
    body = {"model": "auto", "temperature": 0, "messages": [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "Which city is the capital of France?"}]}
    r = sem_client.post("/v1/chat/completions", json=body, headers=AUTH).json()
    assert r["routewise"]["cache_status"]["status"] == "miss"
    assert r["routewise"]["matched_query"] is None


def test_verifier_load_failure_disables_semantic_cache(monkeypatch, tmp_path, caplog):
    class Boom:
        def __init__(self, *a, **k):
            raise OSError("model download failed")

    monkeypatch.setattr(app_module, "SentenceTransformerEmbedder", lambda *a, **k: FakeEmbedder(V))
    monkeypatch.setattr(app_module, "CrossEncoderVerifier", Boom)
    cfg = CacheConfig(semantic_db_path=str(tmp_path / "s.db"))
    assert build_semantic_cache(cfg) is None
    assert "semantic cache DISABLED" in caplog.text


def test_lookup_exception_degrades_to_miss(settings, app_config, fake_clients, fake_redis, repo):
    class Exploding:
        def lookup(self, *a):
            raise RuntimeError("faiss exploded")

    app = create_app(settings, app_config=app_config, clients=fake_clients, redis=fake_redis, repo=repo,
                     router=LengthHeuristicRouter(), semantic_cache=Exploding())
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json=chat_body(), headers=AUTH)
        assert r.status_code == 200 and r.json()["routewise"]["cache_status"]["status"] == "miss"
