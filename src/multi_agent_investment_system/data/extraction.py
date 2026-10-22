"""Provider-independent preparation and classification of daily candles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Real

from multi_agent_investment_system.data.schemas import (
    DailyOHLCVValidationResult,
    validate_daily_ohlcv,
)

_NUMERIC_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class ClassifiedDailyCandle:
    """A normalized input record paired with its existing validation result."""

    record: Mapping[str, object]
    validation: DailyOHLCVValidationResult


@dataclass(frozen=True, slots=True)
class DailyCandleBatchResult:
    """Disjoint, ordered groups of valid, incomplete, and invalid records."""

    valid_records: tuple[ClassifiedDailyCandle, ...]
    incomplete_records: tuple[ClassifiedDailyCandle, ...]
    invalid_records: tuple[ClassifiedDailyCandle, ...]


def _normalize_record(record: Mapping[str, object]) -> dict[str, object]:
    """Copy a canonical candle mapping and decimalize supported numeric fields."""

    normalized = dict(record)
    for field in _NUMERIC_FIELDS:
        value = normalized.get(field)
        if isinstance(value, bool) or isinstance(value, Decimal):
            continue
        if not isinstance(value, (Real, str)):
            continue
        try:
            normalized[field] = Decimal(str(value))
        except (InvalidOperation, ValueError):
            # Leave conversion failures for the shared validator to report.
            pass
    return normalized


def normalize_and_validate_daily_candles(
    records: Iterable[Mapping[str, object]],
) -> DailyCandleBatchResult:
    """Normalize canonical daily candle mappings and classify every record.

    Inputs use the existing validator's lowercase OHLCV field names. Other
    fields, including available symbol, observation time, and source metadata,
    are copied unchanged. Records missing any required value are classified as
    incomplete; complete records that fail validation are classified invalid.
    Retrieval is intentionally outside this interface.
    """

    valid: list[ClassifiedDailyCandle] = []
    incomplete: list[ClassifiedDailyCandle] = []
    invalid: list[ClassifiedDailyCandle] = []

    for record in records:
        normalized = _normalize_record(record)
        validation = validate_daily_ohlcv(normalized)
        classified = ClassifiedDailyCandle(normalized, validation)

        if not validation.is_complete:
            incomplete.append(classified)
        elif validation.is_valid:
            valid.append(classified)
        else:
            invalid.append(classified)

    return DailyCandleBatchResult(
        valid_records=tuple(valid),
        incomplete_records=tuple(incomplete),
        invalid_records=tuple(invalid),
    )
