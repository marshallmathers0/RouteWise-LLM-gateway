"""One client for every OpenAI-compatible provider (Groq, Gemini, OpenRouter, Ollama, ...).

We call the HTTP API directly with httpx instead of using each vendor's SDK so that
*our* reliability layer (timeouts, retries, fallback, circuit breaker) is the only one in
play — SDKs have their own hidden retry logic that would distort measurements.
"""

from __future__ import annotations

from typing import Any

import httpx

from routewise.core.config import ProviderConfig, provider_api_key
from routewise.providers.base import LLMRequest, LLMResponse, ProviderError

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _estimate_tokens(text: str) -> int:
    # ~4 characters per token for English is the usual rule of thumb.
    return max(1, len(text) // 4)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-date form: ignore, fall back to exponential backoff


class OpenAICompatClient:
    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self._http = http_client or httpx.AsyncClient()

    def _api_key(self) -> str | None:
        return provider_api_key(self.config)

    def is_configured(self) -> bool:
        return self._api_key() is not None

    async def chat(self, model: str, request: LLMRequest, timeout_s: float) -> LLMResponse:
        api_key = self._api_key()
        if api_key is None:
            raise ProviderError(
                f"{self.config.api_key_env} not set", provider=self.name, retryable=False
            )

        body: dict[str, Any] = {"model": model, "messages": request.messages}
        for key in ("temperature", "top_p", "stop"):
            value = getattr(request, key)
            if value is not None:
                body[key] = value
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        body.update(request.extra)

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        try:
            resp = await self._http.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout_s,
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"timeout: {e}", provider=self.name, retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"connection error: {e}", provider=self.name, retryable=True) from e

        if resp.status_code != 200:
            raise ProviderError(
                f"HTTP {resp.status_code}: {resp.text[:300]}",
                provider=self.name,
                status_code=resp.status_code,
                retryable=resp.status_code in RETRYABLE_STATUS,
                retry_after_s=_parse_retry_after(resp.headers.get("retry-after")),
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            # Malformed 200s happen (esp. free tiers under load); treat as transient.
            raise ProviderError(
                f"malformed response: {resp.text[:300]}", provider=self.name, retryable=True
            ) from e

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        estimated = prompt_tokens is None or completion_tokens is None
        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens(" ".join(str(m.get("content", "")) for m in request.messages))
        if completion_tokens is None:
            completion_tokens = _estimate_tokens(content)

        return LLMResponse(
            content=content,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            finish_reason=choice.get("finish_reason"),
            model=model,
            provider=self.name,
            tokens_estimated=estimated,
            raw=data,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
