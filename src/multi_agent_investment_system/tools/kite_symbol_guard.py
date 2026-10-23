"""Kite workflow guard for exact requested-symbol identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from multi_agent_investment_system.core.exceptions import (
    MarketDataErrorCategory,
    MarketDataProviderError,
)
from multi_agent_investment_system.tools.market_data import MarketDataProvider


class RequestedSymbolGuard:
    """Reject Kite workflow records not explicitly identified as requested symbol.

    This wrapper is scoped to the Kite workflow composition; general extraction
    and ingestion behavior remains provider-neutral and unchanged.
    """

    def __init__(self, provider: MarketDataProvider) -> None:
        self._provider = provider

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        records = await self._provider.get_daily_candles(symbol, start, end)
        for record in records:
            if not isinstance(record, Mapping) or record.get("symbol") != symbol:
                raise MarketDataProviderError(
                    "Kite market-data response contained a mismatched symbol",
                    category=MarketDataErrorCategory.INVALID_PROVIDER_RESPONSE,
                )
        return records
