"""The demo trap examples with the REAL embedding model and cross-encoder.

Slow (loads ~400 MB of models). Run with: pytest -m slow
"""

from __future__ import annotations

import pytest

from routewise.cache.models import CrossEncoderVerifier, SentenceTransformerEmbedder
from routewise.cache.semantic import SemanticCache
from routewise.cache.store import FaissSqliteStore
from routewise.core.config import CacheConfig

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def models():
    cfg = CacheConfig()
    return SentenceTransformerEmbedder(cfg.embedding_model), CrossEncoderVerifier(cfg.verifier_model)


def _cache(models, tmp_path, **kw) -> SemanticCache:
    emb, ver = models
    cfg = CacheConfig(semantic_db_path=str(tmp_path / "real.db"), similarity_threshold=0.6, **kw)
    return SemanticCache(cfg, emb, FaissSqliteStore(cfg.semantic_db_path, emb.dim), ver)


def _seed(cache, q):
    cache.add("ns", q, cache.embedder.embed([q])[0], f"A:{q}", "m", "p", 1, 1, None)


@pytest.mark.parametrize("cached,incoming", [
    ("What is the capital of India?", "What is the capital of Indonesia?"),
    ("Python 3.10 features", "Python 3.12 features"),
    ("convert 5 miles to km", "convert 5 km to miles"),
    ("Is coffee bad for health?", "Is coffee not bad for health?"),
])
def test_traps_are_rejected(models, tmp_path, cached, incoming):
    c = _cache(models, tmp_path)
    _seed(c, cached)
    r = c.lookup(incoming, "ns")
    assert r.hit is None, r.candidates


@pytest.mark.parametrize("cached,incoming", [
    ("What is the capital of India?", "Which city is the capital of India?"),
    ("How do I learn Python quickly?", "What is the fastest way to learn Python?"),
])
def test_paraphrases_hit(models, tmp_path, cached, incoming):
    c = _cache(models, tmp_path)
    _seed(c, cached)
    r = c.lookup(incoming, "ns")
    assert r.hit is not None, r.candidates
