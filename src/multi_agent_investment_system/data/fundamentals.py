"""Provider-neutral models for source-reported company financials."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from multi_agent_investment_system.core.exceptions import DataValidationError


class FinancialStatementBasis(StrEnum):
    """Basis stated by the financial source, when it is known."""

    STANDALONE = "standalone"
    CONSOLIDATED = "consolidated"


class FundamentalValueStatus(StrEnum):
    """Whether a metric value was reported, absent, or explicitly unavailable."""

    REPORTED = "reported"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FundamentalMetric:
    """One source-reported financial line item, without derived values."""

    name: str
    status: FundamentalValueStatus
    value: object | None = None
    currency: str | None = None
    unit: str | None = None
    provenance: Mapping[str, object] | None = None
    source_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataValidationError("Fundamental metric name must not be empty")
        if not isinstance(self.status, FundamentalValueStatus):
            raise DataValidationError("Fundamental metric status is invalid")

        if self.status is FundamentalValueStatus.REPORTED and self.value is None:
            raise DataValidationError("Reported fundamental metric must have a value")
        if (
            self.status is not FundamentalValueStatus.REPORTED
            and self.value is not None
        ):
            raise DataValidationError(
                "Missing or unavailable fundamental metrics must not have a value"
            )

        for field_name in ("currency", "unit"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataValidationError(
                    f"Fundamental metric {field_name} must be non-empty when provided"
                )

        if any(not isinstance(reference, str) or not reference.strip()
               for reference in self.source_references):
            raise DataValidationError(
                "Fundamental metric source references must be non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class CompanyFundamentalsPeriod:
    """Financial metrics for one listed company and source reporting period.

    ``exchange`` and ``trading_symbol`` use the same identity convention as
    ``ScreenedCandidate``. Period labels, dates, basis, currency, and source
    references remain optional when the source does not provide them.
    """

    exchange: str
    trading_symbol: str
    source: str
    retrieved_at: datetime
    metrics: tuple[FundamentalMetric, ...]
    reporting_period: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    basis: FinancialStatementBasis | None = None
    company_name: str | None = None
    source_references: tuple[str, ...] = ()
    source_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for field_name in ("exchange", "trading_symbol", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(
                    f"Fundamentals {field_name.replace('_', ' ')} must not be empty"
                )

        if not isinstance(self.retrieved_at, datetime):
            raise DataValidationError("Fundamentals retrieval time must be a datetime")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise DataValidationError(
                "Fundamentals retrieval timestamp must be timezone-aware"
            )

        if self.period_start is not None and not isinstance(self.period_start, date):
            raise DataValidationError("Fundamentals period start must be a date")
        if self.period_end is not None and not isinstance(self.period_end, date):
            raise DataValidationError("Fundamentals period end must be a date")
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_start > self.period_end
        ):
            raise DataValidationError(
                "Fundamentals period start must not be after period end"
            )

        for field_name in ("reporting_period", "company_name"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataValidationError(
                    f"Fundamentals {field_name.replace('_', ' ')} must be non-empty "
                    "when provided"
                )

        if self.basis is not None and not isinstance(
            self.basis, FinancialStatementBasis
        ):
            raise DataValidationError("Fundamentals statement basis is invalid")
        if any(not isinstance(metric, FundamentalMetric) for metric in self.metrics):
            raise DataValidationError(
                "Fundamentals metrics must contain FundamentalMetric records"
            )
        if any(not isinstance(reference, str) or not reference.strip()
               for reference in self.source_references):
            raise DataValidationError(
                "Fundamentals source references must be non-empty strings"
            )
