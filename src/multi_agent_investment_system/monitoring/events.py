"""Basic structured events without an observability backend dependency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from multi_agent_investment_system.core.types import RunId


@dataclass(frozen=True, slots=True)
class MonitoringEvent:
    """Named event with run correlation and non-secret string attributes."""

    name: str
    occurred_at: datetime
    run_id: RunId | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
