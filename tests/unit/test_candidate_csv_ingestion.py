"""Tests for validated candidate-screen CSV ingestion."""

from datetime import timedelta
from decimal import Decimal

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.candidate_csv_ingestion import (
    CandidateCsvIssueCode,
    CandidateCsvIssueSeverity,
    ingest_candidate_csv,
)

_HEADER = (
    "source_rank,company_name,market_cap_crore,roce_pct,sales_growth_5y_pct,"
    "profit_growth_5y_pct,source,extracted_at\n"
)
_ROW = "1,Example Ltd,1200,18,12,15,Screener.in,2026-10-23T14:10:19+05:30\n"


def test_valid_import_preserves_metrics_identity_source_and_provenance(
    tmp_path,
) -> None:
    path = tmp_path / "candidate_list.csv"
    path.write_text(_HEADER + _ROW, encoding="utf-8")

    result = ingest_candidate_csv(path)

    assert result.source_file == "candidate_list.csv"
    assert result.source_identifier is None
    assert len(result.rows) == 1
    candidate = result.rows[0]
    assert candidate.source_identifier == "Screener.in"
    assert candidate.row_number == 2
    assert candidate.original_rank == "1"
    assert candidate.company_name == "Example Ltd"
    assert candidate.extracted_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert candidate.original_values == tuple(_ROW.rstrip("\n").split(","))
    assert candidate.numeric_values["market_cap_crore"] == Decimal("1200")
    assert candidate.numeric_values["roce_pct"] == Decimal("18")
    assert result.screening_limitations == (
        "Debt-to-equity data is absent, so this candidate screen criterion "
        "cannot be evaluated.",
        "Exchange and trading-symbol identifiers are absent; company names "
        "are not resolved to listings.",
    )
    assert result.can_evaluate_all_screen_criteria is False


def test_caller_source_identifier_is_preserved(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(_HEADER + _ROW, encoding="utf-8")

    result = ingest_candidate_csv(path, source_identifier="screen-export-2026-10-23")

    assert result.source_identifier == "screen-export-2026-10-23"
    assert result.rows[0].source_identifier == "screen-export-2026-10-23"


def test_missing_required_header_returns_actionable_error(tmp_path) -> None:
    path = tmp_path / "missing_company.csv"
    path.write_text("source_rank,market_cap_crore\n1,1000\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="company_name"):
        ingest_candidate_csv(path)


def test_duplicate_headers_are_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate_headers.csv"
    path.write_text("company_name, Company_Name\nExample,Example\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="headers must be unique"):
        ingest_candidate_csv(path)


def test_invalid_numeric_value_is_reported_and_row_is_retained(tmp_path) -> None:
    path = tmp_path / "invalid_numeric.csv"
    path.write_text(_HEADER + _ROW.replace(",1200,", ",unknown,"), encoding="utf-8")

    result = ingest_candidate_csv(path)

    assert len(result.rows) == 1
    candidate = result.rows[0]
    assert candidate.values["market_cap_crore"] == "unknown"
    assert candidate.numeric_values["market_cap_crore"] is None
    assert candidate.issues[0].code is CandidateCsvIssueCode.INVALID_NUMERIC_VALUE
    assert candidate.issues[0].severity is CandidateCsvIssueSeverity.ERROR


def test_blank_company_name_is_reported_without_dropping_row(tmp_path) -> None:
    path = tmp_path / "blank_name.csv"
    path.write_text(_HEADER + _ROW.replace("1,Example Ltd,", "1,,"), encoding="utf-8")

    result = ingest_candidate_csv(path)

    assert len(result.rows) == 1
    assert result.rows[0].company_name is None
    assert result.rows[0].issues[0].code is CandidateCsvIssueCode.BLANK_COMPANY_NAME


def test_duplicate_names_are_warned_and_both_rows_are_retained(tmp_path) -> None:
    path = tmp_path / "duplicates.csv"
    path.write_text(
        _HEADER + _ROW + _ROW.replace("1,Example", "2, example"),
        encoding="utf-8",
    )

    result = ingest_candidate_csv(path)

    assert len(result.rows) == 2
    assert [row.row_number for row in result.rows] == [2, 3]
    assert all(
        any(
            issue.code is CandidateCsvIssueCode.DUPLICATE_RECORD
            for issue in row.issues
        )
        for row in result.rows
    )
    assert all(
        issue.severity is CandidateCsvIssueSeverity.WARNING
        for row in result.rows
        for issue in row.issues
        if issue.code is CandidateCsvIssueCode.DUPLICATE_RECORD
    )


def test_missing_cells_remain_explicit_and_missing_screen_inputs_are_reported(
    tmp_path,
) -> None:
    path = tmp_path / "missing_values.csv"
    path.write_text(_HEADER + "1,Example Ltd,,,,,Screener.in,\n", encoding="utf-8")

    result = ingest_candidate_csv(path)

    candidate = result.rows[0]
    assert candidate.values["market_cap_crore"] == ""
    assert candidate.numeric_values["market_cap_crore"] is None
    assert candidate.extracted_at is None
    assert any(
        limitation.startswith("Debt-to-equity data is absent")
        for limitation in result.screening_limitations
    )
    assert any(
        limitation.startswith("Market capitalization is missing or invalid")
        for limitation in result.screening_limitations
    )


def test_malformed_csv_is_reported_and_prior_rows_are_retained(tmp_path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(_HEADER + _ROW + '2,"unterminated,1200\n', encoding="utf-8")

    result = ingest_candidate_csv(path)

    assert len(result.rows) == 1
    assert result.rows[0].company_name == "Example Ltd"
    assert any(
        issue.code is CandidateCsvIssueCode.MALFORMED_CSV for issue in result.issues
    )


def test_extra_values_are_reported_and_preserved(tmp_path) -> None:
    path = tmp_path / "extra_values.csv"
    path.write_text(_HEADER + _ROW.rstrip("\n") + ",extra\n", encoding="utf-8")

    result = ingest_candidate_csv(path)

    assert result.rows[0].extra_values == ("extra",)
    assert result.rows[0].issues[0].code is CandidateCsvIssueCode.EXTRA_COLUMNS


def test_repeat_imports_are_deterministic_and_do_not_mutate_input(tmp_path) -> None:
    path = tmp_path / "repeat.csv"
    path.write_text(_HEADER + _ROW, encoding="utf-8")

    first = ingest_candidate_csv(path)
    second = ingest_candidate_csv(path)

    assert first == second


def test_invalid_timestamp_is_warned_and_raw_value_is_retained(tmp_path) -> None:
    path = tmp_path / "bad_timestamp.csv"
    path.write_text(_HEADER + _ROW.replace("+05:30", ""), encoding="utf-8")

    result = ingest_candidate_csv(path)

    candidate = result.rows[0]
    assert candidate.extracted_at is None
    assert candidate.values["extracted_at"] == "2026-10-23T14:10:19"
    assert any(
        issue.code is CandidateCsvIssueCode.INVALID_EXTRACTION_TIMESTAMP
        for issue in candidate.issues
    )
