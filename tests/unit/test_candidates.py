"""Tests for provider-neutral screened-candidate contracts."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.candidates import (
    ScreenedCandidate,
    ScreeningMetric,
)
from multi_agent_investment_system.tools.candidate_screening import (
    CandidateScreeningProvider,
)


class EmptyScreeningProvider:
    async def get_candidates(self, screen: str) -> Sequence[ScreenedCandidate]:
        assert screen == "caller-defined-screen"
        return ()


def _candidate(**overrides: object) -> ScreenedCandidate:
    values: dict[str, object] = {
        "exchange": "NSE",
        "trading_symbol": "RELIANCE",
        "source": "example-provider",
        "captured_at": datetime(2026, 10, 22, 12, tzinfo=UTC),
    }
    values.update(overrides)
    return ScreenedCandidate(**values)  # type: ignore[arg-type]


def test_candidate_preserves_required_identifiers_and_aware_timestamp() -> None:
    captured_at = datetime(
        2026, 10, 22, 17, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )

    candidate = _candidate(captured_at=captured_at)

    assert candidate.exchange == "NSE"
    assert candidate.trading_symbol == "RELIANCE"
    assert candidate.source == "example-provider"
    assert candidate.captured_at is captured_at
    assert candidate.captured_at.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("field", ["exchange", "trading_symbol", "source"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_candidate_requires_nonempty_identifiers(field: str, value: object) -> None:
    with pytest.raises(DataValidationError, match="must not be empty"):
        _candidate(**{field: value})


@pytest.mark.parametrize(
    "captured_at", [None, datetime(2026, 10, 22, 12)]
)
def test_candidate_requires_timezone_aware_capture_timestamp(
    captured_at: datetime | None,
) -> None:
    with pytest.raises(DataValidationError, match="capture timestamp"):
        _candidate(captured_at=captured_at)


def test_optional_metrics_and_source_details_are_preserved() -> None:
    provenance = {"source_column": "Market Cap", "raw_value": "₹ 10 Cr"}
    metric = ScreeningMetric(
        name="Market Cap",
        value="₹ 10 Cr",
        unit="as reported",
        period=None,
        provenance=provenance,
    )
    criteria = ({"source_text": "source-reported criterion"},)
    qualification = {"status": "qualified as reported"}
    source_metadata = {"page": 2}

    candidate = _candidate(
        company_name="Example Limited",
        screen_id="source-screen-42",
        screen_query="source-defined query",
        metrics=(metric,),
        applied_criteria=criteria,
        qualification_details=qualification,
        source_metadata=source_metadata,
    )

    assert candidate.company_name == "Example Limited"
    assert candidate.screen_id == "source-screen-42"
    assert candidate.screen_query == "source-defined query"
    assert candidate.metrics == (metric,)
    assert candidate.metrics[0].value == "₹ 10 Cr"
    assert candidate.metrics[0].period is None
    assert candidate.metrics[0].provenance is provenance
    assert candidate.applied_criteria is criteria
    assert candidate.qualification_details is qualification
    assert candidate.source_metadata is source_metadata


def test_unavailable_optional_information_remains_unset() -> None:
    candidate = _candidate()

    assert candidate.company_name is None
    assert candidate.screen_id is None
    assert candidate.screen_query is None
    assert candidate.metrics == ()
    assert candidate.applied_criteria is None
    assert candidate.qualification_details is None
    assert candidate.source_metadata is None


def test_screening_provider_can_return_successful_empty_result() -> None:
    provider: CandidateScreeningProvider = EmptyScreeningProvider()

    result = asyncio.run(provider.get_candidates("caller-defined-screen"))

    assert result == ()
