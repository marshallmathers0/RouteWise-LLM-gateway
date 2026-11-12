"""Configuration: secrets from environment (pydantic-settings), behaviour from YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str


class TierTarget(BaseModel):
    provider: str
    model: str
    # Extra provider-specific request fields, e.g. {"reasoning_effort": "low"} for gpt-oss.
    params: dict[str, Any] = Field(default_factory=dict)


class ModelPrice(BaseModel):
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens


class ReliabilityConfig(BaseModel):
    request_timeout_s: float = 30.0
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    breaker_failure_threshold: int = 5
    breaker_cooldown_s: float = 30.0
    # Cost guard: a FALLBACK model may cost at most this multiple of the always-strong
    # reference model (counterfactual_strong_model), per input AND per output token.
    # None = availability over cost (use every fallback whatever it costs).
    fallback_price_cap_ratio: float | None = 1.0


class RateLimitConfig(BaseModel):
    enabled: bool = True
    capacity: int = 30
    refill_per_second: float = 0.5


class CacheConfig(BaseModel):
    exact_enabled: bool = True
    exact_ttl_s: int = 86400
    max_cacheable_temperature: float = 0.3

    # Semantic cache. Each stage can be switched off independently (ablation study).
    semantic_enabled: bool = True
    semantic_db_path: str = "data/semantic_cache.db"
    semantic_ttl_s: int = 604800
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    semantic_top_k: int = 5
    similarity_threshold: float = 0.799  # tuned in Phase 3 (reports/cache_eval.md)
    guards_enabled: bool = True
    guards: list[str] = Field(default_factory=lambda: ["numbers", "negation", "entities", "word_order"])
    verifier_enabled: bool = True
    verifier_model: str = "cross-encoder/quora-distilroberta-base"
    verifier_threshold: float = 0.904


class RouterConfig(BaseModel):
    artifact_path: str = "artifacts/router.joblib"


class LoggingConfig(BaseModel):
    log_prompts: bool = True
    prompt_max_chars: int = 500


class AppConfig(BaseModel):
    """Everything in config/routewise.yaml."""

    providers: dict[str, ProviderConfig]
    tiers: dict[str, list[TierTarget]]
    pricing: dict[str, ModelPrice]
    counterfactual_strong_model: str
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def tier_of_model(self, model: str) -> str | None:
        """The tier an explicitly requested model belongs to.

        A model can appear in several chains (gpt-oss-120b is the strong primary AND a cheap
        fallback), so the tier where it is the PRIMARY choice wins; otherwise the first chain
        that contains it.
        """
        for tier, targets in self.tiers.items():
            if targets and targets[0].model == model:
                return tier
        for tier, targets in self.tiers.items():
            if any(t.model == model for t in targets):
                return tier
        return None


class Settings(BaseSettings):
    """Environment-driven settings (secrets and infrastructure URLs)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    routewise_config: str = str(PROJECT_ROOT / "config" / "routewise.yaml")
    # Comma-separated list of client API keys accepted by the gateway.
    routewise_api_keys: str = "rw-dev-key"
    # Optional. Unset = no Redis: exact cache off, rate-limit/breaker state in memory.
    redis_url: str | None = None
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'routewise.db').as_posix()}"

    @property
    def api_keys(self) -> set[str]:
        return {k.strip() for k in self.routewise_api_keys.split(",") if k.strip()}

    def load_app_config(self) -> AppConfig:
        with open(self.routewise_config, encoding="utf-8") as f:
            return AppConfig.model_validate(yaml.safe_load(f))


def provider_api_key(provider: ProviderConfig) -> str | None:
    # Read at call time (not import time) so tests and reloaded .env values are honoured.
    value = os.environ.get(provider.api_key_env)
    return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
