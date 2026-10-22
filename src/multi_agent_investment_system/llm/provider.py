"""Provider-neutral request and response contracts for future model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A role-tagged text message independent of any model provider."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Input to a model adapter, with an optional logical task role."""

    messages: tuple[ChatMessage, ...]
    logical_role: str
    output_schema: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Model text and optional non-secret identifiers for traceability."""

    content: str
    provider_id: str | None = None
    model_id: str | None = None


class ModelProvider(Protocol):
    """Async interface implemented by a future replaceable model adapter."""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for a validated request."""
