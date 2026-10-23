"""Provider-neutral request and response contracts for future model adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from multi_agent_investment_system.core.exceptions import StockInvestmentError

MessageRole = Literal["system", "user", "assistant"]


class ModelOutputRejectionCategory(StrEnum):
    """Fixed safe categories for rejected structured model output."""

    INVALID_RESPONSE = "invalid_response"
    INVALID_UTF8 = "invalid_utf8"
    RESPONSE_TOO_LARGE = "response_too_large"
    INVALID_JSON = "invalid_json"
    UNEXPECTED_SHAPE = "unexpected_shape"
    DISALLOWED_ACTION = "disallowed_action"
    TASK_ID_MISMATCH = "task_id_mismatch"
    INVALID_RATIONALE = "invalid_rationale"
    COMPLETION_CONDITION_MISMATCH = "completion_condition_mismatch"


_SAFE_DIAGNOSTIC_VALUE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SAFE_EXCEPTION_CLASS = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_SENSITIVE_DIAGNOSTIC_SEGMENTS = frozenset(
    {
        "access",
        "access_key",
        "api",
        "api_key",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "key",
        "password",
        "prompt",
        "response",
        "secret",
        "token",
    }
)


def _safe_diagnostic_value(value: object) -> str | None:
    if not isinstance(value, str) or not _SAFE_DIAGNOSTIC_VALUE.fullmatch(value):
        return None
    if set(value.split("_")) & _SENSITIVE_DIAGNOSTIC_SEGMENTS:
        return None
    return value


@dataclass(frozen=True, slots=True)
class ModelProviderDiagnostic:
    """Allowlisted provider failure facts safe to include in workflow reports."""

    exception_class: str
    http_status: int | None = None
    provider_error_code: str | None = None
    provider_error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exception_class, str) or not (
            _SAFE_EXCEPTION_CLASS.fullmatch(self.exception_class)
        ):
            raise ValueError("Model provider exception class is not a safe identifier")
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise ValueError("Model provider HTTP status is invalid")
        for value in (self.provider_error_code, self.provider_error_type):
            if value is not None and (
                _safe_diagnostic_value(value) != value or "_" not in value
            ):
                raise ValueError("Model provider error identifier is not safe")

    @classmethod
    def from_exception_class(cls, exception_class: str) -> ModelProviderDiagnostic:
        """Create a class-only diagnostic without retaining an exception."""

        safe_class = (
            exception_class
            if isinstance(exception_class, str)
            and _SAFE_EXCEPTION_CLASS.fullmatch(exception_class)
            else "UnknownException"
        )
        return cls(exception_class=safe_class)

    def to_dict(self) -> dict[str, str | int]:
        """Return only allowlisted populated diagnostic fields."""

        result: dict[str, str | int] = {"exception_class": self.exception_class}
        if self.http_status is not None:
            result["http_status"] = self.http_status
        if self.provider_error_code is not None:
            result["provider_error_code"] = self.provider_error_code
        if self.provider_error_type is not None:
            result["provider_error_type"] = self.provider_error_type
        return result


class ModelProviderError(StockInvestmentError):
    """Sanitized model-provider failure with safe diagnostics only."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        diagnostic: ModelProviderDiagnostic,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.diagnostic = diagnostic


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
