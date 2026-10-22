"""Allocation assessment contract without weights or recommendation rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    """A position supplied to a future portfolio assessment."""

    asset_id: str
    market_value: Decimal
    currency: str


@dataclass(frozen=True, slots=True)
class AllocationRequest:
    """Portfolio holdings and a proposed amount for deterministic assessment."""

    portfolio_id: str
    positions: tuple[PortfolioPosition, ...]
    proposed_asset_id: str
    proposed_amount: Decimal
    currency: str


@dataclass(frozen=True, slots=True)
class AllocationAssessment:
    """Assessment outcome; it does not itself authorize an investment action."""

    passed: bool
    reasons: tuple[str, ...] = ()


class AllocationPolicy(Protocol):
    """Interface for a separately specified portfolio allocation policy."""

    def assess(self, request: AllocationRequest) -> AllocationAssessment:
        """Assess a proposed amount without inventing allocation thresholds."""
