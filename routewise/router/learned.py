"""The trained router used by the gateway at request time.

The artifact (joblib) holds a fitted sklearn-compatible pipeline that outputs
P(cheap model is good enough), the decision threshold chosen on the validation split,
and metadata. Route to cheap when P >= threshold, else strong.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib

from routewise.cache.models import Embedder
from routewise.router.baselines import RouteDecision
from routewise.router.features import RouterFeaturizer

log = logging.getLogger(__name__)


class LearnedRouter:
    def __init__(self, artifact: dict[str, Any], embedder: Embedder | None) -> None:
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.name = artifact.get("model_name", "learned")
        self.metadata = artifact.get("metadata", {})
        self.featurizer = RouterFeaturizer(embedder, use_embedding=artifact.get("use_embedding", True))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None) -> "LearnedRouter":
        return cls(joblib.load(path), embedder)

    def prob_cheap_ok(self, query: str) -> float:
        x = self.featurizer.transform([query])
        return float(self.model.predict_proba(x)[0, 1])

    def decide(self, query: str) -> RouteDecision:
        t = time.perf_counter()
        p = self.prob_cheap_ok(query)
        tier = "cheap" if p >= self.threshold else "strong"
        log.debug("router %.1f ms p=%.3f", (time.perf_counter() - t) * 1000, p)
        return RouteDecision(tier, round(p, 4), f"learned:{self.name}")
