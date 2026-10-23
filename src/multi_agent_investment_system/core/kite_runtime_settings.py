"""Non-secret runtime settings for the opt-in Kite ingestion workflow."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from multi_agent_investment_system.core.configuration import (
    MarketDataDatabaseSettings,
)
from multi_agent_investment_system.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class KiteWorkflowRuntimeSettings:
    """Explicit non-secret settings; date bounds have no strategy default."""

    database_path: Path
    maximum_calendar_days: int

    def __post_init__(self) -> None:
        if not isinstance(self.database_path, Path):
            raise ConfigurationError("Kite database path must be a Path")
        if (
            isinstance(self.maximum_calendar_days, bool)
            or not isinstance(self.maximum_calendar_days, int)
            or self.maximum_calendar_days < 1
        ):
            raise ConfigurationError(
                "Kite maximum calendar days must be a positive integer"
            )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> KiteWorkflowRuntimeSettings:
        """Read non-secret workflow settings; require an explicit range bound."""

        values = os.environ if environ is None else environ
        days_text = values.get("KITE_INGESTION_MAXIMUM_CALENDAR_DAYS", "").strip()
        if not days_text:
            raise ConfigurationError(
                "KITE_INGESTION_MAXIMUM_CALENDAR_DAYS must be configured"
            )
        try:
            days = int(days_text)
        except ValueError:
            raise ConfigurationError(
                "KITE_INGESTION_MAXIMUM_CALENDAR_DAYS must be a positive integer"
            ) from None
        return cls(
            database_path=MarketDataDatabaseSettings.from_environment(values).path,
            maximum_calendar_days=days,
        )
