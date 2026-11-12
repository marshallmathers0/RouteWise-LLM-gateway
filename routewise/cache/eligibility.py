"""Decide whether a request may use the cache at all.

A cached answer is only correct if the same question would get (essentially) the same
answer now. That fails for: creative sampling (high temperature), questions about the
present moment, and callers who explicitly ask for a fresh answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from routewise.api.schemas import ChatCompletionRequest

# Words/phrases whose answer depends on *when* you ask. Deliberately conservative:
# a false "time-sensitive" just costs a cache miss, a false "not time-sensitive" can serve
# a stale answer.
_TIME_SENSITIVE = re.compile(
    r"\b("
    r"today|tonight|tomorrow|yesterday|now|right now|currently|current|latest|recent|recently|"
    r"this (week|month|year|morning|evening)|last (week|month|year|night)|next (week|month|year)|"
    r"news|breaking|live|real[- ]time|weather|forecast|stock price|share price|exchange rate|"
    r"score|as of|up[- ]to[- ]date|nowadays|these days|trending|"
    # Prices and availability change over time ("iphone 17 price" was cached in the live demo).
    r"prices?|pricing|costs?|how much (is|are|does|do)|discounts?|deals?|on sale|in stock|availability|"
    # Events and releases that move: schedules, results, rankings, upcoming launches.
    r"release date|upcoming|schedule|fixtures?|who won|winner|standings|rankings?|election|"
    r"this season|still (open|available|alive)"
    r")\b",
    re.IGNORECASE,
)

OPT_OUT_HEADER = "x-routewise-cache"  # value "off"/"no"/"false"/"bypass" disables cache


@dataclass
class Eligibility:
    exact: bool
    semantic: bool
    reason: str | None = None  # why caching was bypassed (when exact is False)


def is_time_sensitive(text: str) -> bool:
    return bool(_TIME_SENSITIVE.search(text))


def check_eligibility(
    req: ChatCompletionRequest, headers: dict[str, str], max_temperature: float
) -> Eligibility:
    opt = headers.get(OPT_OUT_HEADER, "").strip().lower()
    cache_control = headers.get("cache-control", "").lower()
    if opt in {"off", "no", "false", "bypass", "0"} or "no-cache" in cache_control or "no-store" in cache_control:
        return Eligibility(False, False, "client_opt_out")
    if req.n is not None and req.n > 1:
        return Eligibility(False, False, "n_gt_1")
    # temperature=None means the provider default (usually 1.0): treat as non-deterministic.
    temperature = 1.0 if req.temperature is None else req.temperature
    if temperature > max_temperature:
        return Eligibility(False, False, f"temperature>{max_temperature}")
    if is_time_sensitive(req.last_user_text()):
        return Eligibility(False, False, "time_sensitive")
    # Exact cache is safe for multi-turn because the key hashes the whole conversation.
    # Semantic cache is single-turn only: "what about the second one?" means nothing
    # without its context, and embedding the whole conversation rarely matches anyway.
    if not req.is_single_turn():
        return Eligibility(True, False, "multi_turn")
    return Eligibility(True, True, None)
