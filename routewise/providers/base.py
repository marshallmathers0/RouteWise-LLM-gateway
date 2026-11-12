"""Provider-agnostic LLM client interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMRequest:
    """What we send to a provider: already-normalized OpenAI-style messages."""

    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    # Provider-specific extra body fields (from the tier target's `params` in config).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    model: str
    provider: str
    # True if the provider didn't report usage and we estimated tokens from characters.
    tokens_estimated: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ProviderError(Exception):
    """A failed provider call.

    `retryable` drives the reliability layer: rate limits (429), server errors (5xx),
    timeouts and connection errors are transient and worth retrying; 4xx client errors
    (bad request, bad key, unknown model) will fail the same way again, so we don't retry
    them on the same provider — but we can still fall back to the next provider.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class LLMClient(Protocol):
    name: str

    def is_configured(self) -> bool:
        """False if e.g. the API key is missing; such providers are skipped in fallback chains."""
        ...

    async def chat(self, model: str, request: LLMRequest, timeout_s: float) -> LLMResponse: ...

    async def aclose(self) -> None: ...
