"""Instant fake provider, for load-testing the gateway's OWN overhead.

Enable with ROUTEWISE_MOCK_PROVIDERS=1 (optionally ROUTEWISE_MOCK_DELAY_MS=<ms> to
simulate model latency). Never enable in production: answers are canned.
"""

from __future__ import annotations

import asyncio
import hashlib

from routewise.providers.base import LLMRequest, LLMResponse


class MockLLMClient:
    def __init__(self, name: str, delay_ms: float = 0.0) -> None:
        self.name = name
        self.delay_s = delay_ms / 1000

    def is_configured(self) -> bool:
        return True

    async def chat(self, model: str, request: LLMRequest, timeout_s: float) -> LLMResponse:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        text = str(request.messages[-1].get("content", ""))
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        return LLMResponse(f"[mock {self.name}/{model}] answer {digest}", prompt_tokens=max(1, len(text) // 4),
                           completion_tokens=50, finish_reason="stop", model=model, provider=self.name)

    async def aclose(self) -> None:
        pass
