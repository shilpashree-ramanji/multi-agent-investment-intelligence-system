"""In-memory ingestion and validation boundary for company fundamentals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    FundamentalsProviderError,
)
from multi_agent_investment_system.data.fundamentals import (
    CompanyFundamentalsPeriod,
    FundamentalValueStatus,
)
from multi_agent_investment_system.tools.fundamentals import (
    CompanyFundamentalsProvider,
)


class FundamentalsIssueCode(StrEnum):
    """Stable validation and reconciliation issue codes."""

    IDENTITY_MISMATCH = "identity_mismatch"
    DUPLICATE_SOURCE_RECORD = "duplicate_source_record"
    MULTIPLE_STATEMENTS_FOR_PERIOD = "multiple_statements_for_period"
    REVISED_STATEMENT = "revised_statement"
    INCONSISTENT_UNITS = "inconsistent_units"
    MISSING_REPORTING_PERIOD = "missing_reporting_period"


@dataclass(frozen=True, slots=True)
class FundamentalsIssue:
    """A deterministic finding that does not discard its source records."""

    code: FundamentalsIssueCode
    record_indexes: tuple[int, ...] = ()
    reporting_period: str | None = None
    metric_name: str | None = None


@dataclass(frozen=True, slots=True)
class FundamentalsRecordAssessment:
    """One returned statement and whether its requested identity matches."""

    record: CompanyFundamentalsPeriod
    is_valid: bool
    issues: tuple[FundamentalsIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class FundamentalsIngestionResult:
    """Complete in-memory result; statement records are never deduplicated."""

    exchange: str
    trading_symbol: str
    assessments: tuple[FundamentalsRecordAssessment, ...]
    issues: tuple[FundamentalsIssue, ...]
    missing_reporting_periods: tuple[str, ...]

    @property
    def records(self) -> tuple[CompanyFundamentalsPeriod, ...]:
        """Return every provider record, including duplicates and revisions."""

        return tuple(assessment.record for assessment in self.assessments)

    @property
    def valid_records(self) -> tuple[CompanyFundamentalsPeriod, ...]:
        """Return records whose exchange and symbol match the request."""

        return tuple(
            assessment.record
            for assessment in self.assessments
            if assessment.is_valid
        )

    @property
    def invalid_records(self) -> tuple[CompanyFundamentalsPeriod, ...]:
        """Return mismatched records without silently dropping them."""

        return tuple(
            assessment.record
            for assessment in self.assessments
            if not assessment.is_valid
        )


class CompanyFundamentalsIngestionService:
    """Fetch and assess normalized statements without storage or derivation.

    Duplicate source records and revised statements remain separate in the
    result. Units are compared only to flag differences; no unit conversion is
    performed. Missing periods are reported only against explicit caller
    expectations because reporting cadence cannot be inferred safely here.
    """

    def __init__(self, provider: CompanyFundamentalsProvider) -> None:
        self._provider = provider

    async def ingest(
        self,
        exchange: str,
        trading_symbol: str,
        *,
        expected_reporting_periods: Sequence[str] = (),
    ) -> FundamentalsIngestionResult:
        """Retrieve and validate statements for one exchange listing."""

        if not isinstance(exchange, str) or not exchange.strip():
            raise DataValidationError("Fundamentals exchange must not be empty")
        if not isinstance(trading_symbol, str) or not trading_symbol.strip():
            raise DataValidationError("Fundamentals symbol must not be empty")
        if isinstance(expected_reporting_periods, (str, bytes)):
            raise DataValidationError(
                "Expected reporting periods must be a sequence of labels"
            )
        if any(
            not isinstance(period, str) or not period.strip()
            for period in expected_reporting_periods
        ):
            raise DataValidationError(
                "Expected reporting periods must be non-empty strings"
            )

        try:
            records = await self._provider.get_company_fundamentals(
                exchange, trading_symbol
            )
        except Exception:
            raise FundamentalsProviderError(
                "Company fundamentals retrieval failed"
            ) from None

        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise DataValidationError(
                "Fundamentals provider returned malformed records"
            )
        if any(not isinstance(record, CompanyFundamentalsPeriod) for record in records):
            raise DataValidationError(
                "Fundamentals provider returned a malformed statement"
            )

        requested_identity = (exchange.strip().upper(), trading_symbol.strip().upper())
        statement_groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
        source_record_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        metric_units: dict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
        assessments: list[FundamentalsRecordAssessment] = []
        issues: list[FundamentalsIssue] = []

        for index, record in enumerate(records):
            identity = (
                record.exchange.strip().upper(),
                record.trading_symbol.strip().upper(),
            )
            record_issues: list[FundamentalsIssue] = []
            is_valid = identity == requested_identity
            if not is_valid:
                issue = FundamentalsIssue(
                    FundamentalsIssueCode.IDENTITY_MISMATCH,
                    record_indexes=(index,),
                    reporting_period=record.reporting_period,
                )
                record_issues.append(issue)
                issues.append(issue)

            if record.reporting_period is not None:
                statement_groups[
                    (
                        identity,
                        record.reporting_period,
                        record.period_start,
                        record.period_end,
                        record.basis,
                        record.source,
                    )
                ].append(index)
            if record.source_record_id is not None:
                source_record_groups[(record.source, record.source_record_id)].append(
                    index
                )
            if record.revision_of is not None:
                issue = FundamentalsIssue(
                    FundamentalsIssueCode.REVISED_STATEMENT,
                    record_indexes=(index,),
                    reporting_period=record.reporting_period,
                )
                record_issues.append(issue)
                issues.append(issue)
            for metric in record.metrics:
                if metric.status is FundamentalValueStatus.REPORTED:
                    metric_units[metric.name.casefold()].add(
                        (metric.currency, metric.unit)
                    )
            assessments.append(
                FundamentalsRecordAssessment(
                    record=record,
                    is_valid=is_valid,
                    issues=tuple(record_issues),
                )
            )

        for indexes in statement_groups.values():
            if len(indexes) > 1:
                issue = FundamentalsIssue(
                    FundamentalsIssueCode.MULTIPLE_STATEMENTS_FOR_PERIOD,
                    record_indexes=tuple(indexes),
                    reporting_period=records[indexes[0]].reporting_period,
                )
                issues.append(issue)
        for indexes in source_record_groups.values():
            if len(indexes) > 1:
                issue = FundamentalsIssue(
                    FundamentalsIssueCode.DUPLICATE_SOURCE_RECORD,
                    record_indexes=tuple(indexes),
                    reporting_period=records[indexes[0]].reporting_period,
                )
                issues.append(issue)
        for metric_name, units in metric_units.items():
            if len(units) > 1:
                issue = FundamentalsIssue(
                    FundamentalsIssueCode.INCONSISTENT_UNITS,
                    metric_name=metric_name,
                )
                issues.append(issue)

        available_periods = {
            record.reporting_period
            for record in records
            if record.reporting_period is not None
        }
        missing_periods = tuple(
            period
            for period in dict.fromkeys(expected_reporting_periods)
            if period not in available_periods
        )
        issues.extend(
            FundamentalsIssue(
                FundamentalsIssueCode.MISSING_REPORTING_PERIOD,
                reporting_period=period,
            )
            for period in missing_periods
        )

        return FundamentalsIngestionResult(
            exchange=exchange,
            trading_symbol=trading_symbol,
            assessments=tuple(assessments),
            issues=tuple(issues),
            missing_reporting_periods=missing_periods,
        )
