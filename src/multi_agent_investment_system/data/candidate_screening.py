"""Deterministic initial candidate screening over validated CSV imports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.candidate_csv_ingestion import (
    CandidateCsvImportResult,
    CandidateCsvIssue,
    CandidateCsvIssueCode,
    ImportedCandidateCsvRow,
)


class InitialScreenCriterion(StrEnum):
    """The five approved initial candidate-screen criteria."""

    SALES_GROWTH_5Y = "sales_growth_5y_pct"
    PROFIT_GROWTH_5Y = "profit_growth_5y_pct"
    ROCE = "roce_pct"
    DEBT_TO_EQUITY = "debt_to_equity"
    MARKET_CAPITALIZATION = "market_cap_crore"


class MetricUnit(StrEnum):
    """Units fixed by the existing candidate CSV column conventions."""

    PERCENT_POINTS = "percentage points"
    RATIO = "ratio"
    INR_CRORE = "INR crore"


class ScreeningOutcome(StrEnum):
    """Criterion and candidate-screen outcomes, separate from recommendations."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class _CriterionDefinition:
    criterion: InitialScreenCriterion
    threshold: Decimal
    unit: MetricUnit
    comparison: str


_CRITERIA = (
    _CriterionDefinition(
        InitialScreenCriterion.SALES_GROWTH_5Y,
        Decimal("10"),
        MetricUnit.PERCENT_POINTS,
        ">",
    ),
    _CriterionDefinition(
        InitialScreenCriterion.PROFIT_GROWTH_5Y,
        Decimal("10"),
        MetricUnit.PERCENT_POINTS,
        ">",
    ),
    _CriterionDefinition(
        InitialScreenCriterion.ROCE,
        Decimal("15"),
        MetricUnit.PERCENT_POINTS,
        ">",
    ),
    _CriterionDefinition(
        InitialScreenCriterion.DEBT_TO_EQUITY,
        Decimal("0.5"),
        MetricUnit.RATIO,
        "<",
    ),
    _CriterionDefinition(
        InitialScreenCriterion.MARKET_CAPITALIZATION,
        Decimal("1000"),
        MetricUnit.INR_CRORE,
        ">",
    ),
)

_AMBIGUOUS_ROW_ISSUES = frozenset(
    {
        CandidateCsvIssueCode.EMPTY_ROW,
        CandidateCsvIssueCode.EXTRA_COLUMNS,
        CandidateCsvIssueCode.BLANK_COMPANY_NAME,
        CandidateCsvIssueCode.DUPLICATE_RECORD,
    }
)


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    """One metric comparison with its unit, source value and explanation."""

    criterion: InitialScreenCriterion
    input_field: str
    expected_unit: MetricUnit
    threshold: Decimal
    observed_value: Decimal | None
    outcome: ScreeningOutcome
    explanation: str


@dataclass(frozen=True, slots=True)
class CandidateScreeningRowResult:
    """Initial-screen result and source identity/provenance for one input row."""

    source_file: str
    source_identifier: str | None
    source_row_number: int
    original_rank: str | None
    extracted_at: datetime | None
    company_name: str | None
    exchange: str | None
    trading_symbol: str | None
    identity_explanation: str
    original_headers: tuple[str, ...]
    original_values: tuple[str, ...]
    extra_values: tuple[str, ...]
    input_values: Mapping[str, str]
    criteria: tuple[CriterionEvaluation, ...]
    outcome: ScreeningOutcome


@dataclass(frozen=True, slots=True)
class CandidateScreeningResult:
    """Screening outcomes only; this type contains no investment action."""

    source_file: str
    source_identifier: str | None
    rows: tuple[CandidateScreeningRowResult, ...]
    screening_limitations: tuple[str, ...]
    import_issues: tuple[CandidateCsvIssue, ...]


def _normalized_headers(import_result: CandidateCsvImportResult) -> tuple[str, ...]:
    return tuple(
        "_".join(header.strip().casefold().split())
        for header in import_result.headers
    )


def _candidate_identity(
    row: ImportedCandidateCsvRow,
) -> tuple[str | None, str | None]:
    exchange = row.values.get("exchange", "").strip() or None
    trading_symbol = row.values.get("trading_symbol", "").strip() or None
    return exchange, trading_symbol


def _row_identity_status(
    row: ImportedCandidateCsvRow,
    duplicate_listing_identities: frozenset[tuple[str, str]],
) -> tuple[bool, str | None, str | None, str]:
    exchange, trading_symbol = _candidate_identity(row)
    if row.company_name is None:
        return (
            True,
            exchange,
            trading_symbol,
            "Source company name is missing; identity is ambiguous.",
        )
    if any(issue.code in _AMBIGUOUS_ROW_ISSUES for issue in row.issues):
        return (
            True,
            exchange,
            trading_symbol,
            "The source row is malformed, duplicated, or has no reliable company "
            "name, so identity is ambiguous.",
        )
    if exchange is not None and trading_symbol is not None:
        if any(character.isspace() for character in trading_symbol):
            return (
                True,
                exchange,
                trading_symbol,
                "Trading symbol contains whitespace and is ambiguous.",
            )
        key = (exchange.casefold(), trading_symbol.casefold())
        if key in duplicate_listing_identities:
            return (
                True,
                exchange,
                trading_symbol,
                "Exchange and trading symbol repeat in this import; "
                "identity is ambiguous.",
            )

    if exchange is None or trading_symbol is None:
        return (
            False,
            exchange,
            trading_symbol,
            "Reliable exchange and trading-symbol identifiers are absent; "
            "company name is retained and no symbol is inferred.",
        )
    return False, exchange, trading_symbol, ""


def _evaluate_criterion(
    row: ImportedCandidateCsvRow,
    definition: _CriterionDefinition,
    headers: tuple[str, ...],
    *,
    identity_is_ambiguous: bool,
) -> CriterionEvaluation:
    field_name = definition.criterion.value
    unknown_reason: str | None = None
    observed_value: Decimal | None = None

    if identity_is_ambiguous:
        unknown_reason = "Candidate listing identity is missing or ambiguous."
    elif headers.count(field_name) != 1:
        unknown_reason = (
            f"Expected exactly one '{field_name}' column with unit "
            f"{definition.unit.value}; the input column is absent or ambiguous."
        )
    elif any(
        issue.code is CandidateCsvIssueCode.INVALID_NUMERIC_VALUE
        and issue.field == field_name
        for issue in row.issues
    ):
        unknown_reason = "The source value is not a valid finite number."
    else:
        candidate_value = row.numeric_values.get(field_name)
        if not isinstance(candidate_value, Decimal) or not candidate_value.is_finite():
            unknown_reason = "The source value is missing, unavailable, or invalid."
        else:
            observed_value = candidate_value

    if unknown_reason is not None:
        return CriterionEvaluation(
            criterion=definition.criterion,
            input_field=field_name,
            expected_unit=definition.unit,
            threshold=definition.threshold,
            observed_value=None,
            outcome=ScreeningOutcome.UNKNOWN,
            explanation=unknown_reason,
        )

    assert observed_value is not None
    passes = (
        observed_value > definition.threshold
        if definition.comparison == ">"
        else observed_value < definition.threshold
    )
    outcome = ScreeningOutcome.PASS if passes else ScreeningOutcome.FAIL
    explanation = (
        f"{observed_value} {definition.unit.value} {definition.comparison} "
        f"{definition.threshold} {definition.unit.value}: {outcome.value.upper()}."
    )
    return CriterionEvaluation(
        criterion=definition.criterion,
        input_field=field_name,
        expected_unit=definition.unit,
        threshold=definition.threshold,
        observed_value=observed_value,
        outcome=outcome,
        explanation=explanation,
    )


def _aggregate_outcomes(
    evaluations: tuple[CriterionEvaluation, ...],
) -> ScreeningOutcome:
    outcomes = {evaluation.outcome for evaluation in evaluations}
    if ScreeningOutcome.FAIL in outcomes:
        return ScreeningOutcome.FAIL
    if ScreeningOutcome.UNKNOWN in outcomes or len(evaluations) != len(_CRITERIA):
        return ScreeningOutcome.UNKNOWN
    return ScreeningOutcome.PASS


def evaluate_initial_candidate_screen(
    import_result: CandidateCsvImportResult,
) -> CandidateScreeningResult:
    """Evaluate the five approved screen criteria over an existing CSV import.

    Only canonical CSV columns define units: growth and ROCE ``*_pct`` values
    are percentage points, debt-to-equity is a ratio, and market capitalization
    is INR crore. No unit conversion, identity resolution, or missing-data
    imputation is performed.
    """

    if not isinstance(import_result, CandidateCsvImportResult):
        raise DataValidationError("Candidate screen input must be a CSV import result")

    headers = _normalized_headers(import_result)
    listing_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(import_result.rows):
        exchange, trading_symbol = _candidate_identity(row)
        if exchange and trading_symbol:
            listing_groups[(exchange.casefold(), trading_symbol.casefold())].append(
                index
            )
    duplicate_listing_identities = frozenset(
        identity for identity, indexes in listing_groups.items() if len(indexes) > 1
    )

    results: list[CandidateScreeningRowResult] = []
    for row in import_result.rows:
        identity_is_ambiguous, exchange, trading_symbol, identity_reason = (
            _row_identity_status(row, duplicate_listing_identities)
        )
        evaluations = tuple(
            _evaluate_criterion(
                row,
                definition,
                headers,
                identity_is_ambiguous=identity_is_ambiguous,
            )
            for definition in _CRITERIA
        )
        outcome = _aggregate_outcomes(evaluations)
        results.append(
            CandidateScreeningRowResult(
                source_file=row.source_file,
                source_identifier=row.source_identifier,
                source_row_number=row.row_number,
                original_rank=row.original_rank,
                extracted_at=row.extracted_at,
                company_name=row.company_name,
                exchange=exchange,
                trading_symbol=trading_symbol,
                identity_explanation=identity_reason,
                original_headers=row.original_headers,
                original_values=row.original_values,
                extra_values=row.extra_values,
                input_values=row.values,
                criteria=evaluations,
                outcome=outcome,
            )
        )
    return CandidateScreeningResult(
        source_file=import_result.source_file,
        source_identifier=import_result.source_identifier,
        rows=tuple(results),
        screening_limitations=import_result.screening_limitations,
        import_issues=import_result.issues,
    )
