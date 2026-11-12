"""Estimated cost from the configurable price table.

All costs are ESTIMATES: free-tier calls cost nothing, so we price them at the list price
of an equivalent paid model (see config/routewise.yaml).
"""

from __future__ import annotations

from routewise.core.config import ModelPrice


def estimate_cost_usd(
    pricing: dict[str, ModelPrice], model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    price = pricing.get(model)
    if price is None:
        return 0.0  # unknown model: report 0 rather than guess
    return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000
