"""Local command-line entry point; this module makes no external calls."""

from __future__ import annotations

import logging

from multi_agent_investment_system.core.logging_config import configure_logging


def main() -> int:
    """Report that the application package is initialized."""
    configure_logging()
    logging.getLogger(__name__).info(
        "Multi-Agent Investment System initialized"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
