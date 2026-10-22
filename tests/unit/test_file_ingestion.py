"""Tests for generic CSV market-data file parsing."""

from datetime import datetime, timedelta, timezone

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.file_ingestion import (
    FileRowIssueCode,
    parse_daily_ohlcv_csv,
)
from multi_agent_investment_system.data.schemas import CandleIssueCode

_CSV_HEADER = "symbol,observed_at,open,high,low,close,volume,source_note\n"
_VALID_ROW = "RELIANCE,2026-10-20T15:30:00+05:30,100,105,99,103,1000,filing-ref\n"


def test_valid_csv_is_normalized_validated_and_provenance_is_preserved(
    tmp_path,
) -> None:
    path = tmp_path / "nse_daily.csv"
    path.write_text(_CSV_HEADER + _VALID_ROW, encoding="utf-8")

    result = parse_daily_ohlcv_csv(path)

    assert result.source_file == "nse_daily.csv"
    assert not result.is_empty_file
    assert result.row_issues == ()
    assert result.batch.incomplete_records == ()
    assert result.batch.invalid_records == ()
    candle = result.batch.valid_records[0].record
    assert candle["symbol"] == "RELIANCE"
    assert candle["observed_at"] == datetime(
        2026,
        10,
        20,
        15,
        30,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )
    assert candle["source_file"] == "nse_daily.csv"
    assert candle["source_row"] == 2
    assert candle["source_note"] == "filing-ref"


def test_missing_csv_cells_are_classified_incomplete(tmp_path) -> None:
    path = tmp_path / "missing_close.csv"
    path.write_text(
        _CSV_HEADER + "RELIANCE,2026-10-20T15:30:00+05:30,100,105,99,,1000,\n",
        encoding="utf-8",
    )

    result = parse_daily_ohlcv_csv(path)

    assert len(result.batch.incomplete_records) == 1
    assert result.batch.incomplete_records[0].validation.is_complete is False
    assert result.batch.incomplete_records[0].validation.issues[0].code is (
        CandleIssueCode.MISSING_VALUE
    )
    assert result.batch.incomplete_records[0].record["close"] is None


def test_invalid_numbers_use_existing_validation_results(tmp_path) -> None:
    path = tmp_path / "invalid_price.csv"
    path.write_text(
        _CSV_HEADER + "RELIANCE,2026-10-20T15:30:00+05:30,100,95,99,103,-1,source\n",
        encoding="utf-8",
    )

    result = parse_daily_ohlcv_csv(path)

    assert len(result.batch.invalid_records) == 1
    assert [
        issue.code for issue in result.batch.invalid_records[0].validation.issues
    ] == [
        CandleIssueCode.NEGATIVE_VOLUME,
        CandleIssueCode.HIGH_BELOW_LOW,
        CandleIssueCode.HIGH_BELOW_OPEN_OR_CLOSE,
    ]


def test_non_numeric_cell_is_classified_by_shared_validator(tmp_path) -> None:
    path = tmp_path / "invalid_number.csv"
    path.write_text(
        _CSV_HEADER
        + "RELIANCE,2026-10-20T15:30:00+05:30,100,not-a-number,99,103,1000,source\n",
        encoding="utf-8",
    )

    result = parse_daily_ohlcv_csv(path)

    assert len(result.batch.invalid_records) == 1
    issue = result.batch.invalid_records[0].validation.issues[0]
    assert issue.code is CandleIssueCode.NON_NUMERIC
    assert issue.field == "high"


def test_short_row_is_padded_as_missing_fields(tmp_path) -> None:
    path = tmp_path / "short_row.csv"
    path.write_text(
        _CSV_HEADER + "RELIANCE,2026-10-20T15:30:00+05:30,100,105,99,103\n",
        encoding="utf-8",
    )

    result = parse_daily_ohlcv_csv(path)

    assert len(result.batch.incomplete_records) == 1
    issue = result.batch.incomplete_records[0].validation.issues[0]
    assert issue.code is CandleIssueCode.MISSING_VALUE
    assert issue.field == "volume"


def test_malformed_row_is_reported_and_later_rows_are_processed(tmp_path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(
        _CSV_HEADER
        + _VALID_ROW
        + "TCS,2026-10-20T15:30:00+05:30,1,2,1,2,5,extra,overflow\n",
        encoding="utf-8",
    )

    result = parse_daily_ohlcv_csv(path)

    assert len(result.batch.valid_records) == 1
    assert result.row_issues[0].code is FileRowIssueCode.EXTRA_COLUMNS
    assert result.row_issues[0].row_number == 3
    assert result.row_issues[0].raw_values[-1] == "overflow"


def test_malformed_observation_time_is_reported_without_fabrication(tmp_path) -> None:
    path = tmp_path / "bad_timestamp.csv"
    path.write_text(
        _CSV_HEADER + "RELIANCE,not-a-time,100,105,99,103,1000,source\n",
        encoding="utf-8",
    )

    result = parse_daily_ohlcv_csv(path)

    assert result.row_issues[0].code is FileRowIssueCode.INVALID_OBSERVED_AT
    record = result.batch.valid_records[0].record
    assert record["observed_at"] is None
    assert record["observed_at_raw"] == "not-a-time"


def test_empty_file_returns_empty_batch(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    result = parse_daily_ohlcv_csv(path)

    assert result.is_empty_file
    assert result.batch.valid_records == ()
    assert result.batch.incomplete_records == ()
    assert result.batch.invalid_records == ()
    assert result.row_issues == ()


def test_missing_required_header_is_an_explicit_error(tmp_path) -> None:
    path = tmp_path / "missing_header.csv"
    path.write_text("symbol,open,high,low,close,volume\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="observed_at"):
        parse_daily_ohlcv_csv(path)
