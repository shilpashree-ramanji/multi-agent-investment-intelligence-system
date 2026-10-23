"""CLI entry point; external calls require explicit live mode confirmation."""

from __future__ import annotations

from multi_agent_investment_system.orchestration.kite_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
