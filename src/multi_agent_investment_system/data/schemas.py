"""Foundational market-data schema with explicit timestamp validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from numbers import Real

from multi_agent_investment_system.core.exceptions import DataValidationError


class CandleIssueCode(StrEnum):
    """Stable codes describing daily-candle validation failures."""

    MISSING_FIELD = "missing_field"
    MISSING_VALUE = "missing_value"
    NON_NUMERIC = "non_numeric"
    NON_FINITE = "non_finite"
    NON_POSITIVE_PRICE = "non_positive_price"
    NEGATIVE_VOLUME = "negative_volume"
    HIGH_BELOW_LOW = "high_below_low"
    HIGH_BELOW_OPEN_OR_CLOSE = "high_below_open_or_close"
    LOW_ABOVE_OPEN_OR_CLOSE = "low_above_open_or_close"


@dataclass(frozen=True, slots=True)
class CandleValidationIssue:
    """One deterministic validation finding for a daily OHLCV record."""

    code: CandleIssueCode
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class DailyOHLCVValidationResult:
    """Validation outcome, retaining existing symbol and observation context."""

    is_complete: bool
    is_valid: bool
    issues: tuple[CandleValidationIssue, ...]
    symbol: str | None = None
    observed_at: datetime | None = None


_PRICE_FIELDS = ("open", "high", "low", "close")
_REQUIRED_CANDLE_FIELDS = (*_PRICE_FIELDS, "volume")


def validate_daily_ohlcv(
    record: Mapping[str, object],
) -> DailyOHLCVValidationResult:
    """Validate a daily OHLCV mapping without filling missing values.

    A missing field, ``None``, or NaN makes the record incomplete. Present but
    invalid values make it invalid. Existing ``symbol`` and ``observed_at``
    context is copied to the result when it has the expected type; no retrieval
    timestamp or source identifier is inferred.
    """

    issues: list[CandleValidationIssue] = []
    values: dict[str, Decimal] = {}
    missing = False

    for field in _REQUIRED_CANDLE_FIELDS:
        if field not in record:
            missing = True
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.MISSING_FIELD,
                    field,
                    f"Required candle field '{field}' is missing.",
                )
            )
            continue

        value = record[field]
        if value is None:
            missing = True
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.MISSING_VALUE,
                    field,
                    f"Required candle field '{field}' has no value.",
                )
            )
            continue

        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.NON_NUMERIC,
                    field,
                    f"Candle field '{field}' must be numeric.",
                )
            )
            continue

        try:
            numeric_value = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError):
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.NON_NUMERIC,
                    field,
                    f"Candle field '{field}' must be numeric.",
                )
            )
            continue

        if numeric_value.is_nan():
            missing = True
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.MISSING_VALUE,
                    field,
                    f"Required candle field '{field}' has no value.",
                )
            )
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.NON_FINITE,
                    field,
                    f"Candle field '{field}' must be finite.",
                )
            )
            continue

        if not numeric_value.is_finite():
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.NON_FINITE,
                    field,
                    f"Candle field '{field}' must be finite.",
                )
            )
            continue

        values[field] = numeric_value
        if field in _PRICE_FIELDS and numeric_value <= 0:
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.NON_POSITIVE_PRICE,
                    field,
                    f"Candle price '{field}' must be greater than zero.",
                )
            )
        elif field == "volume" and numeric_value < 0:
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.NEGATIVE_VOLUME,
                    field,
                    "Candle volume must not be negative.",
                )
            )

    relationships_can_be_checked = all(
        field in values and values[field] > 0 for field in _PRICE_FIELDS
    )
    if relationships_can_be_checked:
        open_price, high, low, close = (values[field] for field in _PRICE_FIELDS)
        if high < low:
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.HIGH_BELOW_LOW,
                    "high",
                    "High must be greater than or equal to Low.",
                )
            )
        if high < max(open_price, close):
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.HIGH_BELOW_OPEN_OR_CLOSE,
                    "high",
                    "High must be greater than or equal to Open and Close.",
                )
            )
        if low > min(open_price, close):
            issues.append(
                CandleValidationIssue(
                    CandleIssueCode.LOW_ABOVE_OPEN_OR_CLOSE,
                    "low",
                    "Low must be less than or equal to Open and Close.",
                )
            )

    symbol = record.get("symbol")
    observed_at = record.get("observed_at")
    return DailyOHLCVValidationResult(
        is_complete=not missing,
        is_valid=not issues,
        issues=tuple(issues),
        symbol=symbol if isinstance(symbol, str) else None,
        observed_at=observed_at if isinstance(observed_at, datetime) else None,
    )


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """A market observation retaining decimal precision and timestamp zone."""

    symbol: str
    observed_at: datetime
    close: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise DataValidationError("Market observation symbol must not be empty")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise DataValidationError(
                "Market observation timestamp must be timezone-aware"
            )
        if not self.close.is_finite() or self.close < 0:
            raise DataValidationError(
                "Market observation close must be finite and non-negative"
            )
        if not self.currency.strip():
            raise DataValidationError("Market observation currency must not be empty")
