"""Tests for the provider-neutral fundamentals ingestion boundary."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    FundamentalsProviderError,
)
from multi_agent_investment_system.data.fundamentals import (
    CompanyFundamentalsPeriod,
    FinancialStatementBasis,
    FundamentalMetric,
    FundamentalValueStatus,
)
from multi_agent_investment_system.data.fundamentals_ingestion import (
    CompanyFundamentalsIngestionService,
    FundamentalsIssueCode,
)


def _period(
    *,
    symbol: str = "RELIANCE",
    period: str | None = "FY 2025-26",
    source_record_id: str | None = None,
    revision_of: str | None = None,
    metric: FundamentalMetric | None = None,
) -> CompanyFundamentalsPeriod:
    return CompanyFundamentalsPeriod(
        exchange="NSE",
        trading_symbol=symbol,
        source="synthetic-fixture",
        retrieved_at=datetime(2026, 10, 22, 12, tzinfo=UTC),
        reporting_period=period,
        basis=FinancialStatementBasis.CONSOLIDATED,
        source_record_id=source_record_id,
        revision_of=revision_of,
        metrics=() if metric is None else (metric,),
    )


class FakeFundamentalsProvider:
    def __init__(
        self,
        records: Sequence[CompanyFundamentalsPeriod] = (),
        error: Exception | None = None,
    ) -> None:
        self.records = records
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def get_company_fundamentals(
        self, exchange: str, trading_symbol: str
    ) -> Sequence[CompanyFundamentalsPeriod]:
        self.calls.append((exchange, trading_symbol))
        if self.error is not None:
            raise self.error
        return self.records


def test_ingestion_preserves_records_and_reports_missing_expected_periods() -> None:
    source_metadata = {"document": "synthetic filing reference"}
    record = _period(
        metric=FundamentalMetric(
            name="Revenue",
            status=FundamentalValueStatus.REPORTED,
            value="1250",
            currency="INR",
            unit="crore",
            provenance=source_metadata,
            source_references=("synthetic-page-1",),
        )
    )
    provider = FakeFundamentalsProvider((record,))

    result = asyncio.run(
        CompanyFundamentalsIngestionService(provider).ingest(
            "NSE",
            "RELIANCE",
            expected_reporting_periods=("FY 2025-26", "Q1 FY 2026-27"),
        )
    )

    assert result.records == (record,)
    assert result.valid_records == (record,)
    assert result.invalid_records == ()
    assert result.missing_reporting_periods == ("Q1 FY 2026-27",)
    assert result.issues[-1].code is FundamentalsIssueCode.MISSING_REPORTING_PERIOD
    assert result.records[0].metrics[0].value == "1250"
    assert result.records[0].metrics[0].unit == "crore"
    assert result.records[0].metrics[0].provenance is source_metadata
    assert provider.calls == [("NSE", "RELIANCE")]


def test_empty_success_is_distinct_and_returns_explicit_period_gaps() -> None:
    provider = FakeFundamentalsProvider()

    result = asyncio.run(
        CompanyFundamentalsIngestionService(provider).ingest(
            "NSE", "RELIANCE", expected_reporting_periods=("FY 2025-26",)
        )
    )

    assert result.records == ()
    assert result.missing_reporting_periods == ("FY 2025-26",)


def test_identity_mismatch_is_flagged_and_record_is_retained() -> None:
    mismatched = _period(symbol="TCS")

    result = asyncio.run(
        CompanyFundamentalsIngestionService(
            FakeFundamentalsProvider((mismatched,))
        ).ingest("NSE", "RELIANCE")
    )

    assert result.records == (mismatched,)
    assert result.valid_records == ()
    assert result.invalid_records == (mismatched,)
    assert result.issues[0].code is FundamentalsIssueCode.IDENTITY_MISMATCH


def test_duplicates_revisions_and_unit_differences_are_reported() -> None:
    first = _period(
        source_record_id="filing-1",
        metric=FundamentalMetric(
            "Revenue", FundamentalValueStatus.REPORTED, "100", "INR", "crore"
        ),
    )
    revision = _period(
        source_record_id="filing-2",
        revision_of="filing-1",
        metric=FundamentalMetric(
            "Revenue", FundamentalValueStatus.REPORTED, "1000000000", "INR", "rupee"
        ),
    )
    exact_duplicate = _period(
        source_record_id="filing-1",
        metric=FundamentalMetric(
            "Revenue", FundamentalValueStatus.REPORTED, "100", "INR", "crore"
        ),
    )

    result = asyncio.run(
        CompanyFundamentalsIngestionService(
            FakeFundamentalsProvider((first, revision, exact_duplicate))
        ).ingest("NSE", "RELIANCE")
    )

    codes = {issue.code for issue in result.issues}
    assert result.records == (first, revision, exact_duplicate)
    assert len(result.valid_records) == 3
    assert FundamentalsIssueCode.MULTIPLE_STATEMENTS_FOR_PERIOD in codes
    assert FundamentalsIssueCode.DUPLICATE_SOURCE_RECORD in codes
    assert FundamentalsIssueCode.REVISED_STATEMENT in codes
    assert FundamentalsIssueCode.INCONSISTENT_UNITS in codes


def test_provider_failure_is_sanitized_and_not_treated_as_empty() -> None:
    provider = FakeFundamentalsProvider(error=RuntimeError("raw provider detail"))

    with pytest.raises(FundamentalsProviderError) as exc_info:
        asyncio.run(
            CompanyFundamentalsIngestionService(provider).ingest("NSE", "RELIANCE")
        )

    assert str(exc_info.value) == "Company fundamentals retrieval failed"
    assert "raw provider detail" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("exchange", "symbol"),
    [("", "RELIANCE"), ("NSE", " ")],
)
def test_invalid_request_is_rejected_before_provider_call(
    exchange: str, symbol: str
) -> None:
    provider = FakeFundamentalsProvider()

    with pytest.raises(DataValidationError):
        asyncio.run(
            CompanyFundamentalsIngestionService(provider).ingest(exchange, symbol)
        )

    assert provider.calls == []


def test_expected_periods_must_be_a_sequence_not_one_string() -> None:
    provider = FakeFundamentalsProvider()

    with pytest.raises(DataValidationError, match="sequence of labels"):
        asyncio.run(
            CompanyFundamentalsIngestionService(provider).ingest(
                "NSE", "RELIANCE", expected_reporting_periods="FY 2025-26"
            )
        )

    assert provider.calls == []


def test_non_model_provider_record_is_rejected() -> None:
    provider = FakeFundamentalsProvider()
    provider.records = ("malformed",)  # type: ignore[assignment]

    with pytest.raises(DataValidationError, match="malformed statement"):
        asyncio.run(
            CompanyFundamentalsIngestionService(provider).ingest("NSE", "RELIANCE")
        )
