"""Deterministic risk evaluation contract; no thresholds are invented here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Validated numeric facts and configured limits for deterministic checks."""

    facts: Mapping[str, Decimal]
    limits: Mapping[str, Decimal]


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Result emitted by deterministic risk logic."""

    rule_id: str
    passed: bool
    reasons: tuple[str, ...] = ()


class RiskRule(Protocol):
    """Pure, deterministic rule interface that model output cannot override."""

    def evaluate(self, context: RiskContext) -> RiskResult:
        """Evaluate facts against explicit configured limits."""
