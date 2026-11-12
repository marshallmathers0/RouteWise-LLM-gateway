"""Simple routing policies. Used as evaluation baselines (Phase 5) and as the gateway's
fallback router until a trained router artifact exists."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class RouteDecision:
    tier: str  # "cheap" | "strong"
    prob_cheap_ok: float | None  # P(cheap model is good enough); None for rule-based policies
    reason: str


class Router(Protocol):
    def decide(self, query: str) -> RouteDecision: ...


class AlwaysRouter:
    def __init__(self, tier: str) -> None:
        self.tier = tier

    def decide(self, query: str) -> RouteDecision:
        return RouteDecision(self.tier, None, f"always_{self.tier}")


_HARD_HINTS = re.compile(r"```|\bdef |\bclass |\bprove\b|\bderive\b|\bstep by step\b|[=+\-*/^]\s*\d", re.I)


class LengthHeuristicRouter:
    """The classic rule-based router: long or code/math-looking queries go to strong."""

    def __init__(self, max_cheap_words: int = 40) -> None:
        self.max_cheap_words = max_cheap_words

    def decide(self, query: str) -> RouteDecision:
        if len(query.split()) > self.max_cheap_words:
            return RouteDecision("strong", None, "heuristic:long_query")
        if _HARD_HINTS.search(query):
            return RouteDecision("strong", None, "heuristic:code_or_math")
        return RouteDecision("cheap", None, "heuristic:short_query")
