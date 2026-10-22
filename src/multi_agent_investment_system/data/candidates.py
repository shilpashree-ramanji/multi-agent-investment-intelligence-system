"""Provider-neutral models for stocks returned by a screening source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from multi_agent_investment_system.core.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class ScreeningMetric:
    """A source-reported metric without inferred units or interpretation."""

    name: str
    value: object
    unit: str | None = None
    period: str | None = None
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataValidationError("Screening metric name must not be empty")


@dataclass(frozen=True, slots=True)
class ScreenedCandidate:
    """A source-returned candidate with only observed screening context."""

    exchange: str
    trading_symbol: str
    source: str
    captured_at: datetime
    company_name: str | None = None
    screen_id: str | None = None
    screen_query: str | None = None
    metrics: tuple[ScreeningMetric, ...] = ()
    applied_criteria: tuple[Mapping[str, object], ...] | None = None
    qualification_details: Mapping[str, object] | None = None
    source_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for field_name in ("exchange", "trading_symbol", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(
                    f"Candidate {field_name.replace('_', ' ')} must not be empty"
                )

        if not isinstance(self.captured_at, datetime):
            raise DataValidationError("Candidate capture timestamp must be a datetime")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise DataValidationError(
                "Candidate capture timestamp must be timezone-aware"
            )

        for field_name in ("company_name", "screen_id", "screen_query"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataValidationError(
                    f"Candidate {field_name.replace('_', ' ')} must be non-empty "
                    "when provided"
                )
