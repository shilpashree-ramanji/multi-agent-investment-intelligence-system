"""Persistence orchestration for validated daily market data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataProviderError,
    StorageError,
)
from multi_agent_investment_system.data.extraction import (
    DailyCandleBatchResult,
    normalize_and_validate_daily_candles,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore


@dataclass(frozen=True, slots=True)
class DailyMarketDataIngestionResult:
    """Range, classification, persistence and readback counts for one refresh."""

    extraction_result: DailyCandleBatchResult
    upserted_count: int
    requested_start: date
    requested_end: date
    retrieval_start: date | None
    stored_count: int

    @property
    def fetched_count(self) -> int:
        """Return the number of provider records classified by extraction."""

        return (
            len(self.extraction_result.valid_records)
            + len(self.extraction_result.incomplete_records)
            + len(self.extraction_result.invalid_records)
        )

    @property
    def valid_count(self) -> int:
        return len(self.extraction_result.valid_records)

    @property
    def incomplete_count(self) -> int:
        return len(self.extraction_result.incomplete_records)

    @property
    def invalid_count(self) -> int:
        return len(self.extraction_result.invalid_records)


class DailyMarketDataIngestionService:
    """Connect existing retrieval/validation to local SQLite persistence."""

    def __init__(
        self,
        extraction_service: DailyMarketDataExtractionService,
        store: SQLiteDailyCandleStore,
    ) -> None:
        self._extraction_service = extraction_service
        self._store = store

    async def ingest(
        self, symbol: str, start: date, end: date
    ) -> DailyMarketDataIngestionResult:
        """Retrieve one inclusive date range and upsert only validated candles."""

        self._validate_request(symbol, start, end)
        return await self._ingest_range(symbol, start, end, start)

    async def ingest_incremental(
        self, symbol: str, initial_start: date, end: date
    ) -> DailyMarketDataIngestionResult:
        """Ingest from an explicit initial date, then resume after stored data.

        The provider receives inclusive calendar-date boundaries. No exchange
        calendar is inferred: weekends and holidays are left to the provider's
        successful response. This forward-only method does not discover gaps
        before the latest stored date; callers can use :meth:`ingest` with an
        explicit earlier range to backfill safely.
        """

        self._validate_request(symbol, initial_start, end)
        try:
            latest_date = await asyncio.to_thread(
                self._store.latest_trading_date, symbol
            )
        except DataValidationError:
            raise
        except Exception:
            raise StorageError("Could not determine market-data refresh date") from None

        retrieval_start = (
            initial_start
            if latest_date is None
            else max(initial_start, latest_date + timedelta(days=1))
        )
        if retrieval_start > end:
            retrieval_start = None
        return await self._ingest_range(
            symbol, initial_start, end, retrieval_start
        )

    @staticmethod
    def _validate_request(symbol: str, start: date, end: date) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise DataValidationError("Market-data symbol must not be empty")
        if not isinstance(start, date) or isinstance(start, datetime):
            raise DataValidationError("Market-data start must be a date")
        if not isinstance(end, date) or isinstance(end, datetime):
            raise DataValidationError("Market-data end must be a date")
        if start > end:
            raise DataValidationError("Market-data start must not be after end")

    async def _ingest_range(
        self,
        symbol: str,
        requested_start: date,
        requested_end: date,
        retrieval_start: date | None,
    ) -> DailyMarketDataIngestionResult:
        if retrieval_start is None:
            extraction_result = normalize_and_validate_daily_candles(())
            upserted_count = 0
        else:
            try:
                extraction_result = await self._extraction_service.extract(
                    symbol, retrieval_start, requested_end
                )
            except DataValidationError:
                raise
            except Exception:
                raise MarketDataProviderError(
                    "Daily market-data retrieval failed"
                ) from None

            try:
                upserted_count = await asyncio.to_thread(
                    self._store.upsert_validated,
                    extraction_result.valid_records,
                )
            except DataValidationError:
                raise
            except Exception:
                raise StorageError("Could not persist daily market data") from None

        try:
            stored_count = await asyncio.to_thread(
                self._store.count_daily_candles,
                symbol,
                requested_start,
                requested_end,
            )
        except DataValidationError:
            raise
        except Exception:
            raise StorageError("Could not verify persisted market data") from None

        return DailyMarketDataIngestionResult(
            extraction_result=extraction_result,
            upserted_count=upserted_count,
            requested_start=requested_start,
            requested_end=requested_end,
            retrieval_start=retrieval_start,
            stored_count=stored_count,
        )
