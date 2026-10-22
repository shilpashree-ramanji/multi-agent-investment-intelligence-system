"""Model-selection contract; no routing policy or provider is selected yet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from multi_agent_investment_system.llm.provider import ModelProvider


@dataclass(frozen=True, slots=True)
class RoutingRequest:
    """Requirements a future model router may use to select an adapter."""

    logical_role: str
    task_complexity: str
    required_capabilities: frozenset[str]


class ModelRouter(Protocol):
    """Select a configured provider adapter using evaluated routing policy."""

    def resolve(self, request: RoutingRequest) -> ModelProvider:
        """Return an adapter meeting the requested role and capabilities."""
