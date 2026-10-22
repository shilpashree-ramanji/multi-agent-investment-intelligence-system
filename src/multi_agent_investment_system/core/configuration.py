"""Secure environment-backed configuration primitives."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from multi_agent_investment_system.core.exceptions import ConfigurationError

_DEFAULT_MARKET_DATA_DATABASE_PATH = Path(".local/market_data.sqlite3")


@dataclass(frozen=True, slots=True)
class KiteConnectCredentials:
    """Kite Connect credentials loaded from environment variables.

    Secret fields are omitted from the generated representation to reduce
    accidental exposure in logs and diagnostic output. This object does not
    perform authentication or token generation.
    """

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> KiteConnectCredentials | None:
        """Load both Kite credentials, or return ``None`` if neither is set.

        A partially configured pair raises a generic error that never includes
        credential values. ``environ`` is injectable for isolated tests.
        """

        values = os.environ if environ is None else environ
        api_key = values.get("KITE_API_KEY")
        api_secret = values.get("KITE_API_SECRET")

        key_is_set = api_key is not None and bool(api_key.strip())
        secret_is_set = api_secret is not None and bool(api_secret.strip())
        if not key_is_set and not secret_is_set:
            return None
        if not key_is_set or not secret_is_set:
            raise ConfigurationError(
                "Kite Connect API key and API secret must be configured together"
            )

        return cls(api_key=api_key, api_secret=api_secret)


@dataclass(frozen=True, slots=True)
class MarketDataDatabaseSettings:
    """Local SQLite path for validated market-data persistence."""

    path: Path = _DEFAULT_MARKET_DATA_DATABASE_PATH

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> MarketDataDatabaseSettings:
        """Load the database path or use the safe project-local default."""

        values = os.environ if environ is None else environ
        configured_path = values.get("MARKET_DATA_DATABASE_PATH", "").strip()
        return cls(
            path=Path(configured_path).expanduser()
            if configured_path
            else _DEFAULT_MARKET_DATA_DATABASE_PATH
        )
