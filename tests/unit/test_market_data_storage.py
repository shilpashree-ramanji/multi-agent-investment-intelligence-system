"""Tests for SQLite daily-candle storage and ingestion orchestration."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataProviderError,
    StorageError,
)
from multi_agent_investment_system.data.extraction import (
    normalize_and_validate_daily_candles,
)
from multi_agent_investment_system.data.ingestion import (
    DailyMarketDataIngestionService,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore


def _candle(
    *,
    symbol: str = "RELIANCE",
    observed_at: datetime | None = None,
    close: str = "103.25",
    volume: int = 1000,
    source: str = "fake-provider",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "observed_at": observed_at or datetime(2026, 10, 20, 10, tzinfo=UTC),
        "open": Decimal("100.00"),
        "high": Decimal("105.00"),
        "low": Decimal("99.00"),
        "close": Decimal(close),
        "volume": volume,
        "source": source,
        "retrieved_at": datetime(2026, 10, 21, 8, tzinfo=UTC),
        "instrument_token": 12345,
        "provider_context": {"segment": "NSE"},
    }


class FakeMarketDataProvider:
    def __init__(
        self,
        records: Sequence[Mapping[str, object]] = (),
        error: Exception | None = None,
    ) -> None:
        self.records = records
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        self.calls.append((symbol, start, end))
        if self.error is not None:
            raise self.error
        return self.records


def test_database_initialization_creates_daily_candle_table(tmp_path) -> None:
    database_path = tmp_path / "market.sqlite3"

    SQLiteDailyCandleStore(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'daily_ohlcv'"
        ).fetchone()
    assert row == ("daily_ohlcv",)


def test_insert_round_trip_preserves_decimal_timestamp_and_provenance(
    tmp_path,
) -> None:
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    classified = normalize_and_validate_daily_candles([_candle()]).valid_records

    assert store.upsert_validated(classified) == 1
    result = store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 20)
    )

    assert len(result) == 1
    assert result[0]["close"] == Decimal("103.25")
    assert result[0]["observed_at"] == datetime(2026, 10, 20, 10, tzinfo=UTC)
    assert result[0]["source"] == "fake-provider"
    assert result[0]["retrieved_at"] == "2026-10-21T08:00:00+00:00"
    assert result[0]["instrument_token"] == 12345
    assert result[0]["provider_context"] == {"segment": "NSE"}


def test_duplicate_symbol_and_trading_date_upserts_latest_validated_values(
    tmp_path,
) -> None:
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    first = normalize_and_validate_daily_candles([_candle()]).valid_records
    replacement = normalize_and_validate_daily_candles(
        [_candle(close="104.50", source="new-provider")]
    ).valid_records

    assert store.upsert_validated(first) == 1
    assert store.upsert_validated(replacement) == 1
    rows = store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 20)
    )

    assert len(rows) == 1
    assert rows[0]["close"] == Decimal("104.50")
    assert rows[0]["source"] == "new-provider"


def test_ingestion_uses_inclusive_range_and_stores_only_valid_records(
    tmp_path,
) -> None:
    valid_first = _candle(observed_at=datetime(2026, 10, 20, tzinfo=UTC))
    valid_second = _candle(
        observed_at=datetime(2026, 10, 21, tzinfo=UTC), close="104.00"
    )
    incomplete = _candle(observed_at=datetime(2026, 10, 20, 11, tzinfo=UTC))
    incomplete["close"] = None
    invalid = _candle(
        observed_at=datetime(2026, 10, 21, 11, tzinfo=UTC), volume=-1
    )
    provider = FakeMarketDataProvider(
        [valid_first, incomplete, invalid, valid_second]
    )
    extraction = DailyMarketDataExtractionService(provider)
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(extraction, store)

    result = asyncio.run(
        service.ingest("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))
    )

    assert provider.calls == [
        ("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))
    ]
    assert len(result.extraction_result.valid_records) == 2
    assert len(result.extraction_result.incomplete_records) == 1
    assert len(result.extraction_result.invalid_records) == 1
    assert result.upserted_count == 2
    assert result.requested_start == date(2026, 10, 20)
    assert result.requested_end == date(2026, 10, 21)
    assert result.retrieval_start == date(2026, 10, 20)
    assert result.fetched_count == 4
    assert result.valid_count == 2
    assert result.incomplete_count == 1
    assert result.invalid_count == 1
    assert result.stored_count == 2
    stored = store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 21)
    )
    assert [record["observed_at"].date() for record in stored] == [
        date(2026, 10, 20),
        date(2026, 10, 21),
    ]


def test_store_rejects_incomplete_or_invalid_classifications(tmp_path) -> None:
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    incomplete = _candle()
    incomplete["close"] = None
    invalid = _candle(volume=-1)
    classified = normalize_and_validate_daily_candles([incomplete, invalid])

    with pytest.raises(DataValidationError, match="Only complete, validated"):
        store.upsert_validated(
            (*classified.incomplete_records, *classified.invalid_records)
        )

    assert store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 20)
    ) == ()


def test_store_revalidates_record_after_classification(tmp_path) -> None:
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    classified = normalize_and_validate_daily_candles([_candle()]).valid_records
    classified[0].record["close"] = Decimal("1")

    with pytest.raises(DataValidationError, match="Only complete, validated"):
        store.upsert_validated(classified)

    assert store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 20)
    ) == ()


def test_ingestion_propagates_provider_failure_without_writing_rows(tmp_path) -> None:
    provider = FakeMarketDataProvider(
        error=RuntimeError("sensitive provider response detail")
    )
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    with pytest.raises(MarketDataProviderError) as error:
        asyncio.run(
            service.ingest("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))
        )

    assert str(error.value) == "Daily market-data retrieval failed"
    assert store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 21)
    ) == ()


def test_first_incremental_ingestion_uses_explicit_start_and_inclusive_end(
    tmp_path,
) -> None:
    start, end = date(2026, 10, 16), date(2026, 10, 20)
    provider = FakeMarketDataProvider(
        [
            _candle(observed_at=datetime(2026, 10, 16, tzinfo=UTC)),
            _candle(observed_at=datetime(2026, 10, 20, tzinfo=UTC)),
        ]
    )
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    result = asyncio.run(service.ingest_incremental("RELIANCE", start, end))

    assert provider.calls == [("RELIANCE", start, end)]
    assert result.retrieval_start == start
    assert result.fetched_count == 2
    assert result.valid_count == 2
    assert result.upserted_count == 2
    assert result.stored_count == 2


def test_incremental_ingestion_resumes_after_latest_calendar_date(tmp_path) -> None:
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    friday = date(2026, 10, 16)
    tuesday = date(2026, 10, 20)
    seed = normalize_and_validate_daily_candles(
        [
            _candle(observed_at=datetime(2026, 10, 16, tzinfo=UTC)),
            _candle(observed_at=datetime(2026, 10, 20, tzinfo=UTC)),
        ]
    ).valid_records
    store.upsert_validated(seed)
    provider = FakeMarketDataProvider(
        [_candle(observed_at=datetime(2026, 10, 22, tzinfo=UTC))]
    )
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    result = asyncio.run(
        service.ingest_incremental("RELIANCE", friday, date(2026, 10, 22))
    )

    assert provider.calls == [("RELIANCE", date(2026, 10, 21), date(2026, 10, 22))]
    assert result.requested_start == friday
    assert result.retrieval_start == date(2026, 10, 21)
    assert result.fetched_count == 1
    assert result.stored_count == 3
    assert tuesday < result.retrieval_start


def test_incremental_no_missing_range_skips_provider_and_reads_existing_rows(
    tmp_path,
) -> None:
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    candle = normalize_and_validate_daily_candles(
        [_candle(observed_at=datetime(2026, 10, 20, tzinfo=UTC))]
    ).valid_records
    store.upsert_validated(candle)
    provider = FakeMarketDataProvider([_candle()])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    result = asyncio.run(
        service.ingest_incremental(
            "RELIANCE", date(2026, 10, 16), date(2026, 10, 20)
        )
    )

    assert provider.calls == []
    assert result.retrieval_start is None
    assert result.fetched_count == 0
    assert result.upserted_count == 0
    assert result.stored_count == 1


def test_empty_provider_response_is_successful_and_reports_zero_fetched(
    tmp_path,
) -> None:
    start, end = date(2026, 10, 20), date(2026, 10, 22)
    provider = FakeMarketDataProvider()
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    result = asyncio.run(service.ingest_incremental("RELIANCE", start, end))

    assert provider.calls == [("RELIANCE", start, end)]
    assert result.fetched_count == 0
    assert result.valid_count == 0
    assert result.upserted_count == 0
    assert result.stored_count == 0


def test_repeated_ingestion_is_idempotent_and_readback_stays_unique(tmp_path) -> None:
    start = end = date(2026, 10, 20)
    provider = FakeMarketDataProvider(
        [_candle(observed_at=datetime(2026, 10, 20, tzinfo=UTC))]
    )
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    first = asyncio.run(service.ingest("RELIANCE", start, end))
    second = asyncio.run(service.ingest("RELIANCE", start, end))

    assert provider.calls == [("RELIANCE", start, end), ("RELIANCE", start, end)]
    assert first.upserted_count == second.upserted_count == 1
    assert first.stored_count == second.stored_count == 1


def test_explicit_reingestion_fills_a_stored_gap_without_duplicate_rows(
    tmp_path,
) -> None:
    start, end = date(2026, 10, 19), date(2026, 10, 21)
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    initial = FakeMarketDataProvider(
        [
            _candle(observed_at=datetime(2026, 10, 19, tzinfo=UTC)),
            _candle(observed_at=datetime(2026, 10, 21, tzinfo=UTC)),
        ]
    )
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(initial), store
    )
    first_result = asyncio.run(service.ingest("RELIANCE", start, end))

    backfill = FakeMarketDataProvider(
        [
            _candle(observed_at=datetime(2026, 10, 19, tzinfo=UTC)),
            _candle(observed_at=datetime(2026, 10, 20, tzinfo=UTC)),
            _candle(observed_at=datetime(2026, 10, 21, tzinfo=UTC)),
        ]
    )
    backfill_service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(backfill), store
    )
    second_result = asyncio.run(backfill_service.ingest("RELIANCE", start, end))

    assert first_result.stored_count == 2
    assert backfill.calls == [("RELIANCE", start, end)]
    assert second_result.upserted_count == 3
    assert second_result.stored_count == 3


def test_persistence_failure_raises_sanitized_error_without_success_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeMarketDataProvider([_candle()])
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")

    def fail_upsert(_candles) -> int:
        raise RuntimeError("sensitive database failure detail")

    monkeypatch.setattr(store, "upsert_validated", fail_upsert)
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    with pytest.raises(StorageError) as error:
        asyncio.run(
            service.ingest("RELIANCE", date(2026, 10, 20), date(2026, 10, 20))
        )

    assert str(error.value) == "Could not persist daily market data"


def test_readback_failure_does_not_return_successful_refresh(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeMarketDataProvider([_candle()])
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")

    def fail_readback(_symbol, _start, _end) -> int:
        raise RuntimeError("sensitive readback detail")

    monkeypatch.setattr(store, "count_daily_candles", fail_readback)
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(provider), store
    )

    with pytest.raises(StorageError) as error:
        asyncio.run(
            service.ingest("RELIANCE", date(2026, 10, 20), date(2026, 10, 20))
        )

    assert str(error.value) == "Could not verify persisted market data"


def test_store_refuses_credential_fields_in_provider_metadata(tmp_path) -> None:
    candle = _candle()
    candle["nested"] = {
        "access_token": "must-not-be-stored",
        "apiSecret": "also-must-not-be-stored",
    }
    classified = normalize_and_validate_daily_candles([candle]).valid_records
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")

    with pytest.raises(DataValidationError, match="credential fields"):
        store.upsert_validated(classified)

    assert store.fetch_daily_candles(
        "RELIANCE", date(2026, 10, 20), date(2026, 10, 20)
    ) == ()


def test_storage_errors_are_sanitized(tmp_path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("not a directory")

    with pytest.raises(StorageError, match="Could not open local market-data storage"):
        SQLiteDailyCandleStore(blocker / "market.sqlite3")
