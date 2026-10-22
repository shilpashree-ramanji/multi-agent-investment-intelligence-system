"""Parse generic CSV daily candles and pass them through shared validation.

The supported CSV header is case-sensitive and requires ``symbol``,
``observed_at``, ``open``, ``high``, ``low``, ``close``, and ``volume``; their
column order is unrestricted. Observation times must use timezone-aware ISO
8601 text. Blank OHLCV cells and short rows become missing values for shared
validation. Additional columns are retained as source metadata. This parser
performs no downloading, authentication, or storage.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.extraction import (
    DailyCandleBatchResult,
    normalize_and_validate_daily_candles,
)

_REQUIRED_HEADERS = (
    "symbol",
    "observed_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_RESERVED_HEADERS = {"source_file", "source_row", "observed_at_raw"}


class FileRowIssueCode(StrEnum):
    """Stable reasons why a CSV row has incomplete non-price structure."""

    EMPTY_ROW = "empty_row"
    EXTRA_COLUMNS = "extra_columns"
    MISSING_SYMBOL = "missing_symbol"
    MISSING_OBSERVED_AT = "missing_observed_at"
    INVALID_OBSERVED_AT = "invalid_observed_at"
    MALFORMED_CSV = "malformed_csv"


@dataclass(frozen=True, slots=True)
class MarketDataFileRowIssue:
    """A row-level parse finding, retaining its source values when available."""

    row_number: int
    code: FileRowIssueCode
    message: str
    raw_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DailyOHLCVFileIngestionResult:
    """Prepared candle classifications and file parsing issues."""

    source_file: str
    batch: DailyCandleBatchResult
    row_issues: tuple[MarketDataFileRowIssue, ...]
    is_empty_file: bool


def parse_daily_ohlcv_csv(path: str | Path) -> DailyOHLCVFileIngestionResult:
    """Parse a generic CSV file, normalize its candles, and validate OHLCV.

    Blank required candle cells become ``None`` and are classified by the
    shared validator as incomplete. Rows with extra columns are retained in
    ``row_issues`` rather than interpreted as candles. Invalid observation
    timestamps and missing symbols are likewise reported as row issues; their
    candle fields still pass through OHLCV validation. The returned
    ``source_file`` and each candle's ``source_file`` use the input basename.

    An empty file succeeds with an empty batch and ``is_empty_file=True``.
    Header-only files also yield an empty batch but are not byte-empty. File
    access and invalid-header failures raise errors rather than becoming empty
    results.
    """

    source_path = Path(path)
    records: list[Mapping[str, object]] = []
    row_issues: list[MarketDataFileRowIssue] = []

    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, strict=True)
        headers = next(reader, None)
        if headers is None:
            return DailyOHLCVFileIngestionResult(
                source_file=source_path.name,
                batch=normalize_and_validate_daily_candles(()),
                row_issues=(),
                is_empty_file=True,
            )

        normalized_headers = [header.strip() for header in headers]
        if len(set(normalized_headers)) != len(normalized_headers):
            raise DataValidationError("CSV header names must be unique")
        if _RESERVED_HEADERS.intersection(normalized_headers):
            raise DataValidationError("CSV header uses a reserved ingestion field")
        missing_headers = [
            header for header in _REQUIRED_HEADERS if header not in normalized_headers
        ]
        if missing_headers:
            missing = ", ".join(missing_headers)
            raise DataValidationError(f"CSV is missing required headers: {missing}")

        try:
            for row in reader:
                row_number = reader.line_num
                if not row:
                    row_issues.append(
                        MarketDataFileRowIssue(
                            row_number,
                            FileRowIssueCode.EMPTY_ROW,
                            "Blank CSV row contains no candle.",
                        )
                    )
                    continue
                if len(row) > len(normalized_headers):
                    row_issues.append(
                        MarketDataFileRowIssue(
                            row_number,
                            FileRowIssueCode.EXTRA_COLUMNS,
                            "CSV row has more values than the header.",
                            tuple(row),
                        )
                    )
                    continue

                padded_row: list[str | None] = [*row]
                padded_row.extend([None] * (len(normalized_headers) - len(row)))
                source_values = dict(zip(normalized_headers, padded_row, strict=True))
                record: dict[str, object] = {
                    "source_file": source_path.name,
                    "source_row": row_number,
                }

                for header, value in source_values.items():
                    if header in _REQUIRED_HEADERS:
                        continue
                    if value is not None:
                        record[header] = value

                symbol = source_values["symbol"]
                if symbol is None or not symbol.strip():
                    record["symbol"] = None
                    row_issues.append(
                        MarketDataFileRowIssue(
                            row_number,
                            FileRowIssueCode.MISSING_SYMBOL,
                            "CSV row has no symbol.",
                            tuple(row),
                        )
                    )
                else:
                    record["symbol"] = symbol.strip()

                observation_text = source_values["observed_at"]
                if observation_text is None or not observation_text.strip():
                    record["observed_at"] = None
                    row_issues.append(
                        MarketDataFileRowIssue(
                            row_number,
                            FileRowIssueCode.MISSING_OBSERVED_AT,
                            "CSV row has no observation timestamp.",
                            tuple(row),
                        )
                    )
                else:
                    try:
                        observed_at = datetime.fromisoformat(observation_text.strip())
                    except ValueError:
                        observed_at = None
                    if (
                        observed_at is None
                        or observed_at.tzinfo is None
                        or observed_at.utcoffset() is None
                    ):
                        record["observed_at"] = None
                        record["observed_at_raw"] = observation_text
                        row_issues.append(
                            MarketDataFileRowIssue(
                                row_number,
                                FileRowIssueCode.INVALID_OBSERVED_AT,
                                "Timestamp must be timezone-aware ISO 8601.",
                                tuple(row),
                            )
                        )
                    else:
                        record["observed_at"] = observed_at

                for field in _REQUIRED_HEADERS[2:]:
                    value = source_values[field]
                    record[field] = (
                        value if value is not None and value.strip() else None
                    )
                records.append(record)
        except csv.Error as exc:
            row_issues.append(
                MarketDataFileRowIssue(
                    reader.line_num + 1,
                    FileRowIssueCode.MALFORMED_CSV,
                    f"Malformed CSV input: {exc}",
                )
            )

    return DailyOHLCVFileIngestionResult(
        source_file=source_path.name,
        batch=normalize_and_validate_daily_candles(records),
        row_issues=tuple(row_issues),
        is_empty_file=False,
    )
