"""Tests for provider-independent NSE equity universe selection."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from multi_agent_investment_system.core.exceptions import (
    CandidateUniverseProviderError,
    DataValidationError,
)
from multi_agent_investment_system.data.candidate_selection import (
    CandidateCriterion,
    CandidateSelectionService,
    select_candidates,
)
from multi_agent_investment_system.data.candidates import (
    NSEEquityUniverseMember,
    UniverseEligibilityStatus,
)
from multi_agent_investment_system.tools.equity_universe import (
    NSEEquityUniverseProvider,
)


def _member(**overrides: object) -> NSEEquityUniverseMember:
    values: dict[str, object] = {
        "exchange": "NSE",
        "trading_symbol": "RELIANCE",
        "source": "fixture-source",
        "eligibility_status": UniverseEligibilityStatus.ELIGIBLE,
        "last_updated_at": datetime(2026, 10, 22, 10, tzinfo=UTC),
        "company_id": "INE002A01018",
        "company_id_type": "ISIN",
        "company_name": "Reliance Industries Limited",
        "provenance": {"record_id": "fixture-1"},
    }
    values.update(overrides)
    return NSEEquityUniverseMember(**values)  # type: ignore[arg-type]


class FakeNSEEquityUniverseProvider:
    def __init__(
        self,
        members: Sequence[NSEEquityUniverseMember] = (),
        error: Exception | None = None,
    ) -> None:
        self.members = members
        self.error = error
        self.calls = 0

    async def refresh_universe(self) -> Sequence[NSEEquityUniverseMember]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.members


def test_universe_member_preserves_identity_eligibility_and_provenance() -> None:
    member = _member()

    assert member.exchange == "NSE"
    assert member.trading_symbol == "RELIANCE"
    assert member.company_id == "INE002A01018"
    assert member.company_name == "Reliance Industries Limited"
    assert member.eligibility_status is UniverseEligibilityStatus.ELIGIBLE
    assert member.last_updated_at.tzinfo is UTC
    assert member.provenance == {"record_id": "fixture-1"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"trading_symbol": ""},
        {"trading_symbol": "BAD SYMBOL"},
        {"exchange": " "},
        {"source": ""},
        {"last_updated_at": datetime(2026, 10, 22, 10)},
        {"eligibility_status": "eligible"},
    ],
)
def test_malformed_universe_member_is_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(DataValidationError):
        _member(**overrides)


def test_duplicate_symbols_are_all_excluded() -> None:
    first = _member()
    second = _member(provenance={"record_id": "fixture-2"})

    result = select_candidates((first, second))

    assert result.selected_candidates == ()
    assert len(result.exclusions) == 2
    assert all(item.reasons == ("duplicate_symbol",) for item in result.exclusions)


def test_conflicting_company_identities_make_duplicate_symbol_ambiguous() -> None:
    first = _member()
    second = _member(company_id="DIFFERENT-ID")

    result = select_candidates((first, second))

    assert result.selected_candidates == ()
    assert all(item.reasons == ("ambiguous_symbol",) for item in result.exclusions)


def test_inactive_non_nse_and_criterion_exclusions_are_explained() -> None:
    eligible = _member()
    inactive = _member(
        trading_symbol="INACTIVE",
        eligibility_status=UniverseEligibilityStatus.INACTIVE,
    )
    foreign_exchange = _member(exchange="BSE", trading_symbol="OTHER")
    criterion = CandidateCriterion(
        name="caller_rule",
        predicate=lambda candidate: candidate.trading_symbol != "RELIANCE",
        exclusion_reason="caller_rule_rejected",
    )

    result = select_candidates((eligible, inactive, foreign_exchange), (criterion,))

    assert result.universe_count == 3
    assert result.selected_candidates == ()
    assert result.exclusions[0].reasons == (
        "criterion:caller_rule:caller_rule_rejected",
    )
    assert result.exclusions[1].reasons == ("eligibility_inactive",)
    assert result.exclusions[2].reasons == ("non_nse_exchange",)


def test_empty_universe_is_a_valid_empty_selection() -> None:
    result = select_candidates(())

    assert result.universe_count == 0
    assert result.selected_candidates == ()
    assert result.exclusions == ()


def test_refresh_accepts_successful_empty_provider_result() -> None:
    provider: NSEEquityUniverseProvider = FakeNSEEquityUniverseProvider()
    service = CandidateSelectionService(provider)

    result = asyncio.run(service.refresh())

    assert result.universe_count == 0
    assert result.selected_candidates == ()


def test_refresh_sanitizes_provider_failure_and_does_not_return_empty_success() -> None:
    provider: NSEEquityUniverseProvider = FakeNSEEquityUniverseProvider(
        error=RuntimeError("sensitive upstream response")
    )
    service = CandidateSelectionService(provider)

    with pytest.raises(CandidateUniverseProviderError) as error:
        asyncio.run(service.refresh())

    assert str(error.value) == "NSE equity universe refresh failed"
    assert error.value.__cause__ is None


def test_refresh_rejects_untyped_malformed_provider_records() -> None:
    provider = FakeNSEEquityUniverseProvider(members=(object(),))  # type: ignore[arg-type]

    with pytest.raises(DataValidationError, match="malformed candidate record"):
        asyncio.run(CandidateSelectionService(provider).refresh())
