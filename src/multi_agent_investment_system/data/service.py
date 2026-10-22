"""Provider-independent daily market-data extraction service."""

from __future__ import annotations

from datetime import date, datetime

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.extraction import (
    DailyCandleBatchResult,
    normalize_and_validate_daily_candles,
)
from multi_agent_investment_system.tools.market_data import MarketDataProvider


class DailyMarketDataExtractionService:
    """Retrieve daily candles and prepare them through the shared validator."""

    def __init__(self, provider: MarketDataProvider) -> None:
        self._provider = provider

    async def extract(
        self, symbol: str, start: date, end: date
    ) -> DailyCandleBatchResult:
        """Retrieve and classify candles for an inclusive date range.

        Provider exceptions are allowed to propagate so retrieval failures stay
        distinguishable from successful empty responses.
        """

        if not isinstance(symbol, str) or not symbol.strip():
            raise DataValidationError("Market-data symbol must not be empty")
        if not isinstance(start, date) or isinstance(start, datetime):
            raise DataValidationError("Market-data start must be a date")
        if not isinstance(end, date) or isinstance(end, datetime):
            raise DataValidationError("Market-data end must be a date")
        if start > end:
            raise DataValidationError("Market-data start must not be after end")

        records = await self._provider.get_daily_candles(symbol, start, end)
        return normalize_and_validate_daily_candles(records)
