"""Deterministic selection over a provider-supplied equity universe."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from multi_agent_investment_system.core.exceptions import (
    CandidateUniverseProviderError,
    DataValidationError,
)
from multi_agent_investment_system.data.candidates import (
    NSEEquityUniverseMember,
    UniverseEligibilityStatus,
)
from multi_agent_investment_system.tools.equity_universe import (
    NSEEquityUniverseProvider,
)


@dataclass(frozen=True, slots=True)
class CandidateCriterion:
    """Caller-supplied pure predicate and its deterministic exclusion explanation."""

    name: str
    predicate: Callable[[NSEEquityUniverseMember], bool]
    exclusion_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataValidationError("Candidate criterion name must not be empty")
        if not callable(self.predicate):
            raise DataValidationError("Candidate criterion must be callable")
        if (
            not isinstance(self.exclusion_reason, str)
            or not self.exclusion_reason.strip()
        ):
            raise DataValidationError("Candidate exclusion reason must not be empty")


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    """A universe member excluded with stable, explainable reason codes."""

    candidate: NSEEquityUniverseMember
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateSelectionResult:
    """Snapshot selection output with exclusions retained for auditability."""

    universe_count: int
    selected_candidates: tuple[NSEEquityUniverseMember, ...]
    exclusions: tuple[CandidateExclusion, ...]


def select_candidates(
    universe: Sequence[NSEEquityUniverseMember],
    criteria: Sequence[CandidateCriterion] = (),
) -> CandidateSelectionResult:
    """Select eligible NSE listings without inferring missing market data."""

    if any(not isinstance(member, NSEEquityUniverseMember) for member in universe):
        raise DataValidationError("Universe contains a malformed candidate record")

    symbol_groups: dict[tuple[str, str], list[int]] = {}
    for index, member in enumerate(universe):
        key = (member.exchange.strip().upper(), member.trading_symbol.strip().upper())
        symbol_groups.setdefault(key, []).append(index)

    duplicate_reasons: dict[int, str] = {}
    for indexes in symbol_groups.values():
        if len(indexes) < 2:
            continue
        company_identities = set()
        for index in indexes:
            member = universe[index]
            if member.company_id is not None:
                company_identities.add(
                    ("id", member.company_id_type, member.company_id)
                )
            elif member.company_name is not None:
                company_identities.add(("name", member.company_name.casefold()))
        reason = (
            "ambiguous_symbol"
            if len(company_identities) > 1
            else "duplicate_symbol"
        )
        for index in indexes:
            duplicate_reasons[index] = reason

    selected: list[NSEEquityUniverseMember] = []
    exclusions: list[CandidateExclusion] = []
    for index, member in enumerate(universe):
        reasons: list[str] = []
        if member.exchange.strip().upper() != "NSE":
            reasons.append("non_nse_exchange")
        if member.eligibility_status != UniverseEligibilityStatus.ELIGIBLE:
            reasons.append(f"eligibility_{member.eligibility_status.value}")
        duplicate_reason = duplicate_reasons.get(index)
        if duplicate_reason is not None:
            reasons.append(duplicate_reason)

        if not reasons:
            for criterion in criteria:
                try:
                    accepted = criterion.predicate(member)
                except Exception:
                    raise DataValidationError(
                        "Candidate screening criterion evaluation failed"
                    ) from None
                if not isinstance(accepted, bool):
                    raise DataValidationError(
                        "Candidate screening criterion must return a boolean"
                    )
                if not accepted:
                    reasons.append(
                        f"criterion:{criterion.name}:{criterion.exclusion_reason}"
                    )

        if reasons:
            exclusions.append(CandidateExclusion(member, tuple(reasons)))
        else:
            selected.append(member)

    return CandidateSelectionResult(
        universe_count=len(universe),
        selected_candidates=tuple(selected),
        exclusions=tuple(exclusions),
    )


class CandidateSelectionService:
    """Refresh an authorized provider snapshot and select explainable candidates."""

    def __init__(self, provider: NSEEquityUniverseProvider) -> None:
        self._provider = provider

    async def refresh(
        self, criteria: Sequence[CandidateCriterion] = ()
    ) -> CandidateSelectionResult:
        """Fetch a current universe snapshot and apply deterministic criteria."""

        try:
            universe = await self._provider.refresh_universe()
        except Exception:
            raise CandidateUniverseProviderError(
                "NSE equity universe refresh failed"
            ) from None
        if not isinstance(universe, Sequence) or isinstance(universe, (str, bytes)):
            raise DataValidationError("Universe provider returned malformed records")
        return select_candidates(universe, criteria)
