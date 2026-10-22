"""Persistence orchestration for validated daily market data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

from multi_agent_investment_system.data.extraction import DailyCandleBatchResult
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore


@dataclass(frozen=True, slots=True)
class DailyMarketDataIngestionResult:
    """Extraction classifications plus the number of rows inserted or updated."""

    extraction_result: DailyCandleBatchResult
    upserted_count: int


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

        extraction_result = await self._extraction_service.extract(symbol, start, end)
        upserted_count = await asyncio.to_thread(
            self._store.upsert_validated,
            extraction_result.valid_records,
        )
        return DailyMarketDataIngestionResult(extraction_result, upserted_count)
