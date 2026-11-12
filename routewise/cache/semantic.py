"""Verified semantic cache: recall (embeddings) -> rule guards -> cross-encoder verifier.

Stage 1 is tuned for RECALL (loose cosine threshold, top-k): it's cheap and its job is to
not miss real paraphrases. Stages 2 and 3 are tuned for PRECISION: they only veto. This
"retrieve then re-rank/verify" split is the same pattern used in search engines and RAG.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from routewise.api.schemas import ChatCompletionRequest
from routewise.cache.guards import run_guards
from routewise.cache.models import Embedder, Verifier
from routewise.cache.store import CacheEntry, SemanticCacheStore
from routewise.core.config import CacheConfig


def semantic_namespace(req: ChatCompletionRequest) -> str:
    """Everything except the user's question that changes what a good answer is."""
    system = [m.text().strip() for m in req.messages if m.role in ("system", "developer")]
    # No max_tokens (see exact.py): the pipeline checks the cached answer fits the caller's limit.
    payload = {"system": system, "model": req.model, "stop": req.stop,
               "response_format": req.response_format}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


@dataclass
class CandidateCheck:
    query: str
    similarity: float
    guards: dict[str, bool] = field(default_factory=dict)
    verifier_score: float | None = None
    accepted: bool = False
    rejected_by: str | None = None  # "guard:<name>" | "verifier"


@dataclass
class SemanticLookup:
    hit: CacheEntry | None
    embedding: np.ndarray
    candidates: list[CandidateCheck]
    latency_ms: dict[str, float]

    @property
    def best(self) -> CandidateCheck | None:
        """The accepted candidate, or else the most similar one considered (for debugging)."""
        accepted = [c for c in self.candidates if c.accepted]
        return accepted[0] if accepted else (self.candidates[0] if self.candidates else None)

    def summary(self) -> dict[str, Any]:
        return {"candidates": [asdict(c) for c in self.candidates]}


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 3)


class SemanticCache:
    def __init__(self, cfg: CacheConfig, embedder: Embedder, store: SemanticCacheStore,
                 verifier: Verifier | None) -> None:
        if cfg.verifier_enabled and verifier is None:
            raise ValueError("verifier_enabled but no verifier given")
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.verifier = verifier

    def lookup(self, query: str, namespace: str) -> SemanticLookup:
        lat: dict[str, float] = {}
        t = time.perf_counter()
        emb = self.embedder.embed([query])[0]
        lat["sem_embed"] = _ms(t)

        t = time.perf_counter()
        found = self.store.search(namespace, emb, self.cfg.semantic_top_k)
        lat["sem_search"] = _ms(t)
        # Stage 1: loose cosine threshold (recall).
        found = [(e, s) for e, s in found if s >= self.cfg.similarity_threshold]
        checks = [CandidateCheck(query=e.query, similarity=round(s, 4)) for e, s in found]
        if not found:
            return SemanticLookup(None, emb, checks, lat)

        # Stage 2: rule guards (precision).
        survivors: list[int] = []
        if self.cfg.guards_enabled:
            t = time.perf_counter()
            for i, (entry, _) in enumerate(found):
                report = run_guards(query, entry.query, self.cfg.guards)
                checks[i].guards = report.results
                if report.passed:
                    survivors.append(i)
                else:
                    checks[i].rejected_by = "guard:" + report.failed[0]
            lat["sem_guards"] = _ms(t)
        else:
            survivors = list(range(len(found)))

        # Stage 3: cross-encoder verifier (precision), batched over surviving candidates.
        if survivors and self.cfg.verifier_enabled and self.verifier is not None:
            t = time.perf_counter()
            scores = self.verifier.score([(query, found[i][0].query) for i in survivors])
            lat["sem_verifier"] = _ms(t)
            passing = []
            for i, s in zip(survivors, scores):
                checks[i].verifier_score = round(s, 4)
                if s >= self.cfg.verifier_threshold:
                    passing.append(i)
                else:
                    checks[i].rejected_by = "verifier"
            survivors = passing

        if not survivors:
            return SemanticLookup(None, emb, checks, lat)
        # Candidates are in descending similarity order, so the first survivor is the best.
        winner = survivors[0]
        checks[winner].accepted = True
        return SemanticLookup(found[winner][0], emb, checks, lat)

    def add(self, namespace: str, query: str, embedding: np.ndarray, answer: str, model: str, provider: str,
            prompt_tokens: int, completion_tokens: int, route: str | None) -> None:
        self.store.add(namespace, query, embedding, answer, model, provider, prompt_tokens, completion_tokens,
                       route, self.cfg.semantic_ttl_s)
