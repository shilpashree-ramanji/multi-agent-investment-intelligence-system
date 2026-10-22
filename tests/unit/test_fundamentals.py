"""Tests for provider-neutral company fundamentals records."""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.fundamentals import (
    CompanyFundamentalsPeriod,
    FinancialStatementBasis,
    FundamentalMetric,
    FundamentalValueStatus,
)


def _fundamentals(**overrides: object) -> CompanyFundamentalsPeriod:
    values: dict[str, object] = {
        "exchange": "NSE",
        "trading_symbol": "RELIANCE",
        "source": "example-source",
        "retrieved_at": datetime(2026, 10, 22, 12, tzinfo=UTC),
        "metrics": (),
    }
    values.update(overrides)
    return CompanyFundamentalsPeriod(**values)  # type: ignore[arg-type]


def test_fundamentals_preserve_reporting_period_basis_and_provenance() -> None:
    retrieved_at = datetime(
        2026, 10, 22, 17, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    provenance = {"source_label": "Revenue from operations"}
    metric = FundamentalMetric(
        name="Revenue",
        status=FundamentalValueStatus.REPORTED,
        value="1250000000",
        currency="INR",
        unit="source-reported rupees",
        provenance=provenance,
        source_references=("report-page-12",),
    )
    source_metadata = {"document_id": "report-2025-26"}

    record = _fundamentals(
        retrieved_at=retrieved_at,
        reporting_period="FY 2025-26",
        period_start=date(2025, 4, 1),
        period_end=date(2026, 3, 31),
        basis=FinancialStatementBasis.CONSOLIDATED,
        company_name="Example Limited",
        metrics=(metric,),
        source_references=("report-url-or-document-id",),
        source_metadata=source_metadata,
    )

    assert record.exchange == "NSE"
    assert record.trading_symbol == "RELIANCE"
    assert record.reporting_period == "FY 2025-26"
    assert record.period_start == date(2025, 4, 1)
    assert record.period_end == date(2026, 3, 31)
    assert record.basis is FinancialStatementBasis.CONSOLIDATED
    assert record.retrieved_at is retrieved_at
    assert record.metrics[0].value == "1250000000"
    assert record.metrics[0].currency == "INR"
    assert record.metrics[0].unit == "source-reported rupees"
    assert record.metrics[0].provenance is provenance
    assert record.metrics[0].source_references == ("report-page-12",)
    assert record.source_references == ("report-url-or-document-id",)
    assert record.source_metadata is source_metadata


def test_fundamentals_preserve_company_and_filing_revision_identity() -> None:
    published_at = datetime(2026, 5, 1, 9, tzinfo=UTC)
    record = _fundamentals(
        company_id="INE002A01018",
        company_id_type="ISIN",
        source_record_id="filing-revision-2",
        published_at=published_at,
        revision_of="filing-revision-1",
    )

    assert record.company_id == "INE002A01018"
    assert record.company_id_type == "ISIN"
    assert record.source_record_id == "filing-revision-2"
    assert record.published_at is published_at
    assert record.revision_of == "filing-revision-1"


def test_standalone_basis_is_preserved() -> None:
    record = _fundamentals(basis=FinancialStatementBasis.STANDALONE)

    assert record.basis is FinancialStatementBasis.STANDALONE


@pytest.mark.parametrize(
    ("status", "value"),
    [
        (FundamentalValueStatus.REPORTED, None),
        (FundamentalValueStatus.MISSING, "must-not-be-attached"),
        (FundamentalValueStatus.UNAVAILABLE, 42),
    ],
)
def test_metric_status_must_match_value_presence(
    status: FundamentalValueStatus, value: object | None
) -> None:
    with pytest.raises(DataValidationError):
        FundamentalMetric(name="Revenue", status=status, value=value)


def test_missing_and_unavailable_are_distinct_and_not_imputed() -> None:
    missing = FundamentalMetric(
        name="Operating cash flow", status=FundamentalValueStatus.MISSING
    )
    unavailable = FundamentalMetric(
        name="Free cash flow", status=FundamentalValueStatus.UNAVAILABLE
    )

    assert missing.status is FundamentalValueStatus.MISSING
    assert missing.value is None
    assert unavailable.status is FundamentalValueStatus.UNAVAILABLE
    assert unavailable.value is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"exchange": ""},
        {"trading_symbol": " "},
        {"source": ""},
        {"retrieved_at": datetime(2026, 10, 22, 12)},
        {"period_start": date(2026, 4, 1), "period_end": date(2026, 3, 31)},
        {"metrics": ("Revenue",)},
        {"published_at": datetime(2026, 10, 22, 12)},
        {"revision_of": " "},
    ],
)
def test_invalid_fundamentals_records_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(DataValidationError):
        _fundamentals(**overrides)


def test_reported_zero_is_preserved_as_a_value() -> None:
    metric = FundamentalMetric(
        name="Borrowings",
        status=FundamentalValueStatus.REPORTED,
        value=0,
        currency="INR",
        unit="source-reported rupees",
    )

    assert metric.value == 0
    assert metric.status is FundamentalValueStatus.REPORTED
