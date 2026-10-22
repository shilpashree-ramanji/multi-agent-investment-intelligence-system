"""Shared identifiers and role/status labels used across package boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType

RunId = NewType("RunId", str)
EvidenceId = NewType("EvidenceId", str)


class AgentRole(StrEnum):
    """High-level agent responsibilities; these do not define strategies."""

    DISCOVERY = "discovery"
    RESEARCH = "research"
    STRATEGY = "strategy"
    REVIEW = "review"
    RISK = "risk"


class WorkflowStatus(StrEnum):
    """Generic lifecycle values for future workflow implementations."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
