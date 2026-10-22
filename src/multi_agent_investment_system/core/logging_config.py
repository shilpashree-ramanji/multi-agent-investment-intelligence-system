"""Small standard-library logging setup that does not expose configuration."""

from __future__ import annotations

import logging

from multi_agent_investment_system.core.exceptions import ConfigurationError


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging using a standard Python logging level."""
    normalized_level = level.upper()
    numeric_level = logging.getLevelNamesMapping().get(normalized_level)
    if not isinstance(numeric_level, int):
        raise ConfigurationError("Logging level must be a standard Python level")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
