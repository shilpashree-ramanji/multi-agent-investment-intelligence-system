"""Typed decision records and a persistence-neutral journal interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from multi_agent_investment_system.core.types import EvidenceId, RunId


class DecisionKind(StrEnum):
    """Kinds of reviewable journal entries."""

    PROPOSED = "proposed"
    SKIPPED_OPPORTUNITY = "skipped_opportunity"


class HumanApproval(StrEnum):
    """Human review state attached to a proposed decision."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Auditable proposal or skipped-opportunity record."""

    record_id: str
    run_id: RunId
    kind: DecisionKind
    rationale: str
    evidence_ids: tuple[EvidenceId, ...]
    created_at: datetime
    human_approval: HumanApproval = HumanApproval.PENDING


class DecisionJournal(Protocol):
    """Persistence-neutral contract for recording decision history."""

    def record(self, entry: DecisionRecord) -> None:
        """Record a proposed decision or skipped opportunity."""
