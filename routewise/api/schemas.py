"""OpenAI-compatible request/response schemas plus RouteWise metadata.

Only the subset of the Chat Completions API that RouteWise needs is modelled strictly;
unknown fields are accepted (extra="allow") so standard SDK calls never get rejected.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None

    def text(self) -> str:
        """Flatten content to plain text (we only route/cache text)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text or "" for p in self.content if p.type == "text")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    stream: bool | None = False
    user: str | None = None
    # Passed through to the provider (e.g. {"type": "json_object"} for JSON mode).
    response_format: dict[str, Any] | None = None

    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.text()
        return ""

    def is_single_turn(self) -> bool:
        """One user message, optionally preceded by system/developer prompts."""
        non_system = [m for m in self.messages if m.role not in ("system", "developer")]
        return len(non_system) == 1 and non_system[0].role == "user"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"


class CacheStatus(BaseModel):
    status: Literal["miss", "exact_hit", "semantic_hit", "bypassed"] = "miss"
    reason: str | None = None


class RouteWiseMeta(BaseModel):
    """Extra metadata RouteWise attaches to every response (ignored by OpenAI SDKs)."""

    request_id: str
    route: str | None = None  # "cheap" | "strong" | None for cache hits
    route_reason: str | None = None
    router_prob: float | None = None  # P(cheap model is good enough)
    cache_status: CacheStatus = Field(default_factory=CacheStatus)
    matched_query: str | None = None
    similarity: float | None = None
    verifier_score: float | None = None
    guard_results: dict[str, Any] | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    cost_is_estimate: bool = True
    provider_used: str | None = None
    model_used: str | None = None
    fallback_used: bool = False


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-rw-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    routewise: RouteWiseMeta


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
