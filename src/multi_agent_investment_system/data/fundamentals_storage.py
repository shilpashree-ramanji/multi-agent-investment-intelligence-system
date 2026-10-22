"""Append-only SQLite storage for validated company fundamentals statements."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    StorageError,
)
from multi_agent_investment_system.data.fundamentals import (
    CompanyFundamentalsPeriod,
    FinancialStatementBasis,
    FundamentalMetric,
    FundamentalValueStatus,
)


def _encode_value(value: object) -> dict[str, object]:
    """Encode supported values with explicit type tags for lossless readback."""

    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataValidationError("Fundamentals contain a non-finite value")
        return {"type": "float", "value": repr(value)}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DataValidationError("Fundamentals contain a non-finite value")
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DataValidationError(
                "Fundamentals mappings must use string keys for storage"
            )
        return {
            "type": "mapping",
            "value": [
                [key, _encode_value(value[key])] for key in sorted(value)
            ],
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "value": [_encode_value(item) for item in value]}
    raise DataValidationError("Fundamentals contain a value unsupported by storage")


def _decode_value(encoded: object) -> object:
    if not isinstance(encoded, dict) or not isinstance(encoded.get("type"), str):
        raise ValueError("Invalid encoded value")

    kind = encoded["type"]
    value = encoded.get("value")
    if kind == "none":
        return None
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "str" and isinstance(value, str):
        return value
    if kind == "int" and isinstance(value, str):
        return int(value)
    if kind == "float" and isinstance(value, str):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite stored value")
        return result
    if kind == "decimal" and isinstance(value, str):
        result = Decimal(value)
        if not result.is_finite():
            raise ValueError("Non-finite stored value")
        return result
    if kind == "datetime" and isinstance(value, str):
        return datetime.fromisoformat(value)
    if kind == "date" and isinstance(value, str):
        return date.fromisoformat(value)
    if kind == "mapping" and isinstance(value, list):
        result: dict[str, object] = {}
        for entry in value:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or entry[0] in result
            ):
                raise ValueError("Invalid stored mapping")
            result[entry[0]] = _decode_value(entry[1])
        return result
    if kind in {"tuple", "list"} and isinstance(value, list):
        items = [_decode_value(item) for item in value]
        return tuple(items) if kind == "tuple" else items
    raise ValueError("Unknown encoded value type")


def _record_payload(record: CompanyFundamentalsPeriod) -> dict[str, object]:
    return {
        "exchange": record.exchange,
        "trading_symbol": record.trading_symbol,
        "source": record.source,
        "retrieved_at": record.retrieved_at.isoformat(),
        "reporting_period": record.reporting_period,
        "period_start": record.period_start.isoformat()
        if record.period_start is not None
        else None,
        "period_end": record.period_end.isoformat()
        if record.period_end is not None
        else None,
        "basis": record.basis.value if record.basis is not None else None,
        "company_name": record.company_name,
        "company_id": record.company_id,
        "company_id_type": record.company_id_type,
        "source_record_id": record.source_record_id,
        "published_at": record.published_at.isoformat()
        if record.published_at is not None
        else None,
        "revision_of": record.revision_of,
        "source_references": list(record.source_references),
        "source_metadata": _encode_value(record.source_metadata),
        "metrics": [
            {
                "name": metric.name,
                "status": metric.status.value,
                "value": _encode_value(metric.value),
                "currency": metric.currency,
                "unit": metric.unit,
                "provenance": _encode_value(metric.provenance),
                "source_references": list(metric.source_references),
            }
            for metric in record.metrics
        ],
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record_identity(payload: Mapping[str, object]) -> str:
    """Hash source statement content, excluding the time it was retrieved."""

    identity_payload = dict(payload)
    identity_payload.pop("retrieved_at", None)
    return hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()


def _record_from_payload(payload: Mapping[str, object]) -> CompanyFundamentalsPeriod:
    try:
        metrics_value = payload["metrics"]
        if not isinstance(metrics_value, list):
            raise ValueError("Invalid stored metrics")
        metrics: list[FundamentalMetric] = []
        for item in metrics_value:
            if not isinstance(item, dict):
                raise ValueError("Invalid stored metric")
            provenance = _decode_value(item["provenance"])
            if provenance is not None and not isinstance(provenance, Mapping):
                raise ValueError("Invalid metric provenance")
            references = item["source_references"]
            if not isinstance(references, list):
                raise ValueError("Invalid metric references")
            metrics.append(
                FundamentalMetric(
                    name=item["name"],
                    status=FundamentalValueStatus(item["status"]),
                    value=_decode_value(item["value"]),
                    currency=item["currency"],
                    unit=item["unit"],
                    provenance=provenance,
                    source_references=tuple(references),
                )
            )

        metadata = _decode_value(payload["source_metadata"])
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("Invalid source metadata")
        period_start = payload["period_start"]
        period_end = payload["period_end"]
        published_at = payload["published_at"]
        references = payload["source_references"]
        if not isinstance(references, list):
            raise ValueError("Invalid statement references")

        return CompanyFundamentalsPeriod(
            exchange=payload["exchange"],
            trading_symbol=payload["trading_symbol"],
            source=payload["source"],
            retrieved_at=datetime.fromisoformat(payload["retrieved_at"]),
            metrics=tuple(metrics),
            reporting_period=payload["reporting_period"],
            period_start=date.fromisoformat(period_start)
            if isinstance(period_start, str)
            else None,
            period_end=date.fromisoformat(period_end)
            if isinstance(period_end, str)
            else None,
            basis=FinancialStatementBasis(payload["basis"])
            if payload["basis"] is not None
            else None,
            company_name=payload["company_name"],
            company_id=payload["company_id"],
            company_id_type=payload["company_id_type"],
            source_record_id=payload["source_record_id"],
            published_at=datetime.fromisoformat(published_at)
            if isinstance(published_at, str)
            else None,
            revision_of=payload["revision_of"],
            source_references=tuple(references),
            source_metadata=metadata,
        )
    except (KeyError, TypeError, ValueError, DataValidationError):
        raise StorageError("Stored company fundamentals are malformed") from None


class SQLiteCompanyFundamentalsStore:
    """Append-only storage for validated statement records and revisions.

    The deterministic key hashes canonical statement content while excluding
    only ``retrieved_at``. Exact re-ingestion is ignored. If the same source
    filing ID later has different content, the changed content receives a new
    key and both versions remain queryable. This store never updates or deletes
    an existing statement row.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path).expanduser()
        self.initialize()

    @property
    def database_path(self) -> Path:
        """Return the configured local database path."""

        return self._database_path

    def initialize(self) -> None:
        """Create the fundamentals tables and lookup index if needed."""

        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS company_fundamentals (
                        record_key TEXT PRIMARY KEY,
                        exchange TEXT NOT NULL,
                        trading_symbol TEXT NOT NULL,
                        company_id TEXT,
                        reporting_period TEXT,
                        basis TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_fundamentals_company_period
                    ON company_fundamentals (
                        exchange, trading_symbol, reporting_period, basis
                    )
                    """
                )
        except sqlite3.Error:
            raise StorageError(
                "Could not initialize local fundamentals storage"
            ) from None
        finally:
            connection.close()

    def insert_validated(
        self, records: Sequence[CompanyFundamentalsPeriod]
    ) -> int:
        """Insert new statement content, ignoring exact repeated ingestion.

        All records are revalidated and serialized before opening the write
        transaction. Any SQLite write failure rolls back the complete batch.
        The return value is the number of newly inserted statement versions.
        """

        rows = [self._prepare_row(record) for record in records]
        if not rows:
            return 0

        connection = self._connect()
        try:
            with connection:
                before = connection.total_changes
                connection.executemany(
                    """
                    INSERT INTO company_fundamentals (
                        record_key, exchange, trading_symbol, company_id,
                        reporting_period, basis, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_key) DO NOTHING
                    """,
                    rows,
                )
                return connection.total_changes - before
        except sqlite3.Error:
            raise StorageError("Could not write local company fundamentals") from None
        finally:
            connection.close()

    def fetch_company_periods(
        self,
        exchange: str,
        trading_symbol: str,
        reporting_period: str | None = None,
        basis: FinancialStatementBasis | None = None,
    ) -> tuple[CompanyFundamentalsPeriod, ...]:
        """Return all stored versions for a listing, optionally filtered.

        Results are in insertion order. No revision is selected as the only
        truth; callers receive every retained version for the requested period.
        """

        if not isinstance(exchange, str) or not exchange.strip():
            raise DataValidationError("Fundamentals exchange must not be empty")
        if not isinstance(trading_symbol, str) or not trading_symbol.strip():
            raise DataValidationError("Fundamentals symbol must not be empty")
        if reporting_period is not None and (
            not isinstance(reporting_period, str) or not reporting_period.strip()
        ):
            raise DataValidationError(
                "Fundamentals reporting period must be non-empty when provided"
            )
        if basis is not None and not isinstance(basis, FinancialStatementBasis):
            raise DataValidationError("Fundamentals statement basis is invalid")

        conditions = ["exchange = ?", "trading_symbol = ?"]
        parameters: list[str] = [exchange.strip(), trading_symbol.strip()]
        if reporting_period is not None:
            conditions.append("reporting_period = ?")
            parameters.append(reporting_period)
        if basis is not None:
            conditions.append("basis = ?")
            parameters.append(basis.value)

        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT payload_json FROM company_fundamentals WHERE "
                + " AND ".join(conditions)
                + " ORDER BY rowid",
                parameters,
            ).fetchall()
        except sqlite3.Error:
            raise StorageError("Could not read local company fundamentals") from None
        finally:
            connection.close()

        records: list[CompanyFundamentalsPeriod] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
                if not isinstance(payload, dict):
                    raise ValueError("Invalid statement payload")
            except (json.JSONDecodeError, TypeError, ValueError):
                raise StorageError(
                    "Stored company fundamentals are malformed"
                ) from None
            records.append(_record_from_payload(payload))
        return tuple(records)

    def count_records(self) -> int:
        """Return the total number of stored statement versions."""

        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS row_count FROM company_fundamentals"
            ).fetchone()
            return int(row["row_count"])
        except sqlite3.Error:
            raise StorageError("Could not read local company fundamentals") from None
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._database_path, timeout=10)
            connection.row_factory = sqlite3.Row
            return connection
        except (OSError, sqlite3.Error):
            raise StorageError("Could not open local fundamentals storage") from None

    @staticmethod
    def _prepare_row(record: CompanyFundamentalsPeriod) -> tuple[object, ...]:
        if not isinstance(record, CompanyFundamentalsPeriod):
            raise DataValidationError(
                "Only validated company fundamentals records can be stored"
            )
        record.__post_init__()
        for metric in record.metrics:
            metric.__post_init__()

        payload = _record_payload(record)
        payload_json = _canonical_json(payload)
        record_key = _record_identity(payload)
        return (
            record_key,
            record.exchange.strip(),
            record.trading_symbol.strip(),
            record.company_id,
            record.reporting_period,
            record.basis.value if record.basis is not None else None,
            payload_json,
        )
