"""Request log table (Postgres in Docker, SQLite locally / on HF Spaces)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    # Never store the raw client key: only a short hash prefix that identifies it.
    api_key_id: Mapped[str] = mapped_column(String(16), index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    requested_model: Mapped[str] = mapped_column(String(128))
    route: Mapped[str | None] = mapped_column(String(16), nullable=True)
    route_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    router_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)

    cache_status: Mapped[str] = mapped_column(String(16), index=True)
    cache_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    matched_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    verifier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    guard_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[dict] = mapped_column(JSON, default=dict)  # per-stage breakdown
    # Both are ESTIMATES from the price table. The counterfactual is what the same tokens
    # would have cost on the strong model (0 actual spend on a cache hit vs this).
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    counterfactual_strong_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    status_code: Mapped[int] = mapped_column(Integer, default=200)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
