"""Framework-neutral contract for future stateful research workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from multi_agent_investment_system.core.types import RunId, WorkflowStatus


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    """Minimal input envelope for a future workflow implementation."""

    run_id: RunId
    inputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Workflow state and human-review requirement, without a decision rule."""

    run_id: RunId
    status: WorkflowStatus
    output_references: tuple[str, ...] = ()
    human_approval_required: bool = True


class ResearchWorkflow(Protocol):
    """Async workflow boundary that can later be implemented with LangGraph."""

    async def run(self, request: WorkflowRequest) -> WorkflowResult:
        """Run a research workflow and return its traceable state."""
