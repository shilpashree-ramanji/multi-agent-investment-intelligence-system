"""Tests for the provider-independent initial candidate-screen evaluator."""

import csv

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.candidate_csv_ingestion import (
    ingest_candidate_csv,
)
from multi_agent_investment_system.data.candidate_screening import (
    CandidateScreeningResult,
    InitialScreenCriterion,
    MetricUnit,
    ScreeningOutcome,
    evaluate_initial_candidate_screen,
)

_HEADERS = (
    "company_name",
    "exchange",
    "trading_symbol",
    "sales_growth_5y_pct",
    "profit_growth_5y_pct",
    "roce_pct",
    "debt_to_equity",
    "market_cap_crore",
    "source",
    "source_rank",
    "extracted_at",
)
_BASE_VALUES = {
    "company_name": "Example Ltd",
    "exchange": "NSE",
    "trading_symbol": "EXAMPLE",
    "sales_growth_5y_pct": "12",
    "profit_growth_5y_pct": "15",
    "roce_pct": "18",
    "debt_to_equity": "0.4",
    "market_cap_crore": "1200",
    "source": "Screener.in",
    "source_rank": "7",
    "extracted_at": "2026-10-23T14:10:19+05:30",
}
_THRESHOLDS = {
    "sales_growth_5y_pct": "10",
    "profit_growth_5y_pct": "10",
    "roce_pct": "15",
    "debt_to_equity": "0.5",
    "market_cap_crore": "1000",
}


def _import_rows(tmp_path, rows, *, headers=_HEADERS):
    path = tmp_path / "candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return ingest_candidate_csv(path)


def _evaluations_by_field(row_result):
    return {evaluation.input_field: evaluation for evaluation in row_result.criteria}


def test_all_five_criteria_pass_with_explicit_units_and_traceability(tmp_path) -> None:
    imported = _import_rows(tmp_path, [_BASE_VALUES])

    result = evaluate_initial_candidate_screen(imported)

    assert isinstance(result, CandidateScreeningResult)
    assert result.rows[0].outcome is ScreeningOutcome.PASS
    assert len(result.rows[0].criteria) == 5
    assert result.rows[0].company_name == "Example Ltd"
    assert result.rows[0].exchange == "NSE"
    assert result.rows[0].trading_symbol == "EXAMPLE"
    assert result.rows[0].source_file == "candidates.csv"
    assert result.rows[0].source_identifier == "Screener.in"
    assert result.rows[0].source_row_number == 2
    assert result.rows[0].original_rank == "7"
    assert result.rows[0].original_headers == _HEADERS
    assert result.rows[0].original_values == tuple(
        _BASE_VALUES[field] for field in _HEADERS
    )
    assert result.rows[0].input_values["debt_to_equity"] == "0.4"
    evaluations = _evaluations_by_field(result.rows[0])
    assert evaluations["sales_growth_5y_pct"].expected_unit is (
        MetricUnit.PERCENT_POINTS
    )
    assert evaluations["debt_to_equity"].expected_unit is MetricUnit.RATIO
    assert evaluations["market_cap_crore"].expected_unit is MetricUnit.INR_CRORE
    assert not hasattr(result.rows[0], "recommendation")
    assert not hasattr(result, "recommendation")


@pytest.mark.parametrize(
    ("field_name", "criterion"),
    [
        ("sales_growth_5y_pct", InitialScreenCriterion.SALES_GROWTH_5Y),
        ("profit_growth_5y_pct", InitialScreenCriterion.PROFIT_GROWTH_5Y),
        ("roce_pct", InitialScreenCriterion.ROCE),
        ("debt_to_equity", InitialScreenCriterion.DEBT_TO_EQUITY),
        ("market_cap_crore", InitialScreenCriterion.MARKET_CAPITALIZATION),
    ],
)
def test_value_equal_to_each_strict_threshold_fails(
    tmp_path, field_name, criterion
) -> None:
    values = {**_BASE_VALUES, field_name: _THRESHOLDS[field_name]}

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [values]))

    evaluation = _evaluations_by_field(result.rows[0])[field_name]
    assert evaluation.criterion is criterion
    assert evaluation.outcome is ScreeningOutcome.FAIL
    assert result.rows[0].outcome is ScreeningOutcome.FAIL


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sales_growth_5y_pct", "10.01"),
        ("profit_growth_5y_pct", "10.01"),
        ("roce_pct", "15.01"),
        ("debt_to_equity", "0.49"),
        ("market_cap_crore", "1000.01"),
    ],
)
def test_values_beyond_each_threshold_pass(tmp_path, field_name, value) -> None:
    values = {**_BASE_VALUES, field_name: value}

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [values]))

    assert _evaluations_by_field(result.rows[0])[field_name].outcome is (
        ScreeningOutcome.PASS
    )


def test_missing_metric_is_unknown_and_missing_data_aggregates_to_unknown(
    tmp_path,
) -> None:
    values = {**_BASE_VALUES, "debt_to_equity": ""}

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [values]))

    evaluations = _evaluations_by_field(result.rows[0])
    assert evaluations["debt_to_equity"].outcome is ScreeningOutcome.UNKNOWN
    assert evaluations["debt_to_equity"].observed_value is None
    assert result.rows[0].outcome is ScreeningOutcome.UNKNOWN


def test_invalid_metric_is_unknown_even_when_other_metrics_pass(tmp_path) -> None:
    values = {**_BASE_VALUES, "roce_pct": "not-a-number"}

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [values]))

    evaluations = _evaluations_by_field(result.rows[0])
    assert evaluations["roce_pct"].outcome is ScreeningOutcome.UNKNOWN
    assert "not a valid finite number" in evaluations["roce_pct"].explanation
    assert result.rows[0].outcome is ScreeningOutcome.UNKNOWN


def test_fail_takes_priority_over_unknown_in_overall_aggregation(tmp_path) -> None:
    values = {
        **_BASE_VALUES,
        "sales_growth_5y_pct": "10",
        "debt_to_equity": "",
    }

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [values]))

    assert result.rows[0].outcome is ScreeningOutcome.FAIL
    assert _evaluations_by_field(result.rows[0])["sales_growth_5y_pct"].outcome is (
        ScreeningOutcome.FAIL
    )
    assert _evaluations_by_field(result.rows[0])["debt_to_equity"].outcome is (
        ScreeningOutcome.UNKNOWN
    )


def test_duplicate_listing_identity_makes_all_criteria_unknown(tmp_path) -> None:
    first = dict(_BASE_VALUES)
    second = {**_BASE_VALUES, "company_name": "Different Example Ltd"}

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [first, second]))

    assert len(result.rows) == 2
    assert all(row.outcome is ScreeningOutcome.UNKNOWN for row in result.rows)
    assert all(
        all(item.outcome is ScreeningOutcome.UNKNOWN for item in row.criteria)
        for row in result.rows
    )
    assert all(
        "identity is ambiguous" in row.identity_explanation for row in result.rows
    )


def test_duplicate_company_name_makes_rows_ambiguous_without_dropping_them(
    tmp_path,
) -> None:
    first = dict(_BASE_VALUES)
    second = {**_BASE_VALUES, "trading_symbol": "EXAMPLE-B"}

    result = evaluate_initial_candidate_screen(_import_rows(tmp_path, [first, second]))

    assert len(result.rows) == 2
    assert all(row.outcome is ScreeningOutcome.UNKNOWN for row in result.rows)
    assert all("ambiguous" in row.identity_explanation for row in result.rows)


def test_missing_listing_identity_is_not_inferred_from_company_name(tmp_path) -> None:
    headers = tuple(
        header for header in _HEADERS if header not in {"exchange", "trading_symbol"}
    )
    values = {key: value for key, value in _BASE_VALUES.items() if key in headers}
    values["debt_to_equity"] = ""

    result = evaluate_initial_candidate_screen(
        _import_rows(tmp_path, [values], headers=headers)
    )

    row_result = result.rows[0]
    assert row_result.exchange is None
    assert row_result.trading_symbol is None
    assert row_result.outcome is ScreeningOutcome.UNKNOWN
    evaluations = _evaluations_by_field(row_result)
    assert evaluations["sales_growth_5y_pct"].outcome is ScreeningOutcome.PASS
    assert evaluations["debt_to_equity"].outcome is ScreeningOutcome.UNKNOWN
    assert "no symbol is inferred" in row_result.identity_explanation
    assert any(
        "identifiers are absent" in item for item in result.screening_limitations
    )


def test_incompatible_metric_unit_header_produces_unknown(tmp_path) -> None:
    headers = tuple(
        "sales_growth_5y_ratio" if field == "sales_growth_5y_pct" else field
        for field in _HEADERS
    )
    values = {
        ("sales_growth_5y_ratio" if key == "sales_growth_5y_pct" else key): value
        for key, value in _BASE_VALUES.items()
    }

    result = evaluate_initial_candidate_screen(
        _import_rows(tmp_path, [values], headers=headers)
    )

    evaluation = _evaluations_by_field(result.rows[0])["sales_growth_5y_pct"]
    assert evaluation.outcome is ScreeningOutcome.UNKNOWN
    assert "percentage points" in evaluation.explanation
    assert "column is absent or ambiguous" in evaluation.explanation


def test_current_csv_schema_cannot_fully_screen_rows(tmp_path) -> None:
    headers = tuple(field for field in _HEADERS if field != "debt_to_equity")
    values = {key: value for key, value in _BASE_VALUES.items() if key in headers}

    result = evaluate_initial_candidate_screen(
        _import_rows(tmp_path, [values], headers=headers)
    )

    assert (
        _evaluations_by_field(result.rows[0])["debt_to_equity"].outcome
        is ScreeningOutcome.UNKNOWN
    )
    assert result.rows[0].outcome is ScreeningOutcome.UNKNOWN
    assert any("Debt-to-equity" in item for item in result.screening_limitations)


def test_evaluator_requires_existing_csv_import_result() -> None:
    with pytest.raises(DataValidationError, match="CSV import result"):
        evaluate_initial_candidate_screen(object())  # type: ignore[arg-type]
