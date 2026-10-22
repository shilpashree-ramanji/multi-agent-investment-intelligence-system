"""SQLite persistence for validated daily OHLCV candles."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    StorageError,
)
from multi_agent_investment_system.data.extraction import ClassifiedDailyCandle
from multi_agent_investment_system.data.schemas import validate_daily_ohlcv

_CANDLE_FIELDS = frozenset(
    {"symbol", "observed_at", "open", "high", "low", "close", "volume"}
)
_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "api_key",
        "api_secret",
        "access_token",
        "request_token",
        "authorization",
        "apikey",
        "apisecret",
        "accesstoken",
        "requesttoken",
    }
)


def _json_default(value: object) -> str:
    if isinstance(value, (Decimal, datetime, date)):
        return str(value) if isinstance(value, Decimal) else value.isoformat()
    raise TypeError


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            compact_key = normalized_key.replace("_", "")
            if (
                normalized_key in _SENSITIVE_METADATA_KEYS
                or compact_key in _SENSITIVE_METADATA_KEYS
            ):
                return True
            if _contains_sensitive_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


class SQLiteDailyCandleStore:
    """Store validated candles, updating an existing symbol/date row on conflict.

    The current market-data contract is NSE-focused, so ``(symbol,
    trading_date)`` is the instrument/day key. Each upsert replaces OHLCV and
    provider metadata with values from the latest validated retrieval. Decimal
    values are stored as text to avoid SQLite floating-point coercion.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path).expanduser()
        self.initialize()

    @property
    def database_path(self) -> Path:
        """Return the configured local database path."""

        return self._database_path

    def initialize(self) -> None:
        """Create the candle table if it does not already exist."""

        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_ohlcv (
                        symbol TEXT NOT NULL,
                        trading_date TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        open TEXT NOT NULL,
                        high TEXT NOT NULL,
                        low TEXT NOT NULL,
                        close TEXT NOT NULL,
                        volume TEXT NOT NULL,
                        source TEXT,
                        retrieved_at TEXT,
                        source_metadata_json TEXT NOT NULL,
                        PRIMARY KEY (symbol, trading_date)
                    )
                    """
                )
        except sqlite3.Error:
            raise StorageError(
                "Could not initialize local market-data storage"
            ) from None
        finally:
            connection.close()

    def upsert_validated(
        self, candles: Sequence[ClassifiedDailyCandle]
    ) -> int:
        """Insert validated candles or update existing rows for the same day."""

        rows = [self._prepare_row(candle) for candle in candles]
        if not rows:
            return 0

        connection = self._connect()
        try:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO daily_ohlcv (
                        symbol, trading_date, observed_at, open, high, low, close,
                        volume, source, retrieved_at, source_metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, trading_date) DO UPDATE SET
                        observed_at = excluded.observed_at,
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        source = excluded.source,
                        retrieved_at = excluded.retrieved_at,
                        source_metadata_json = excluded.source_metadata_json
                    """,
                    rows,
                )
            return len(rows)
        except sqlite3.Error:
            raise StorageError("Could not write local market data") from None
        finally:
            connection.close()

    def fetch_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> tuple[dict[str, object], ...]:
        """Fetch stored candles for an inclusive symbol and date range."""

        if not symbol.strip():
            raise DataValidationError("Market-data symbol must not be empty")
        if start > end:
            raise DataValidationError("Market-data start must not be after end")

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT symbol, observed_at, open, high, low, close, volume,
                       source, retrieved_at, source_metadata_json
                FROM daily_ohlcv
                WHERE symbol = ? AND trading_date >= ? AND trading_date <= ?
                ORDER BY trading_date
                """,
                (symbol, start.isoformat(), end.isoformat()),
            ).fetchall()
        except sqlite3.Error:
            raise StorageError("Could not read local market data") from None
        finally:
            connection.close()

        results: list[dict[str, object]] = []
        for row in rows:
            record = json.loads(row["source_metadata_json"])
            record.update(
                {
                    "symbol": row["symbol"],
                    "observed_at": datetime.fromisoformat(row["observed_at"]),
                    "open": Decimal(row["open"]),
                    "high": Decimal(row["high"]),
                    "low": Decimal(row["low"]),
                    "close": Decimal(row["close"]),
                    "volume": Decimal(row["volume"]),
                }
            )
            if row["source"] is not None:
                record["source"] = row["source"]
            if row["retrieved_at"] is not None:
                record["retrieved_at"] = row["retrieved_at"]
            results.append(record)
        return tuple(results)

    def _connect(self) -> sqlite3.Connection:
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._database_path, timeout=10)
            connection.row_factory = sqlite3.Row
            return connection
        except (OSError, sqlite3.Error):
            raise StorageError("Could not open local market-data storage") from None

    @staticmethod
    def _prepare_row(candle: ClassifiedDailyCandle) -> tuple[object, ...]:
        validation = validate_daily_ohlcv(candle.record)
        if not validation.is_complete or not validation.is_valid:
            raise DataValidationError(
                "Only complete, validated daily candles can be stored"
            )

        record = candle.record
        symbol = record.get("symbol")
        observed_at = record.get("observed_at")
        if not isinstance(symbol, str) or not symbol.strip():
            raise DataValidationError("Validated candle must include a symbol")
        if (
            not isinstance(observed_at, datetime)
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
        ):
            raise DataValidationError(
                "Stored candle must include a timezone-aware observation timestamp"
            )

        if _contains_sensitive_key(record):
            raise DataValidationError(
                "Provider metadata contains credential fields; storage was refused"
            )

        metadata = {
            key: value
            for key, value in record.items()
            if key not in _CANDLE_FIELDS
        }
        try:
            metadata_json = json.dumps(
                metadata,
                default=_json_default,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            raise DataValidationError(
                "Provider metadata cannot be represented in local storage"
            ) from None

        numeric_values = []
        for field in ("open", "high", "low", "close", "volume"):
            value = record.get(field)
            if isinstance(value, bool) or value is None:
                raise DataValidationError(
                    "Validated candle is missing a required numeric field"
                )
            try:
                numeric_values.append(Decimal(str(value)))
            except Exception:
                raise DataValidationError(
                    "Validated candle contains an unrepresentable numeric field"
                ) from None

        source = record.get("source")
        retrieved_at = record.get("retrieved_at")
        retrieved_at_text = None
        if isinstance(retrieved_at, (datetime, date)):
            retrieved_at_text = _json_default(retrieved_at)
        elif retrieved_at is not None:
            retrieved_at_text = str(retrieved_at)
        return (
            symbol.strip(),
            observed_at.date().isoformat(),
            observed_at.isoformat(),
            *(str(value) for value in numeric_values),
            source if isinstance(source, str) else None,
            retrieved_at_text,
            metadata_json,
        )
