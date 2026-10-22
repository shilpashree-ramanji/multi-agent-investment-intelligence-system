"""Yahoo Finance adapter for the provider-neutral daily-candle contract.

This is a prototype integration. Automated access and data-retention rights
remain subject to verification against the applicable terms.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from importlib import import_module
from typing import Any

from multi_agent_investment_system.tools.market_data import MarketDataProvider

_YAHOO_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YahooFinanceMarketDataProvider(MarketDataProvider):
    """Retrieve Yahoo Finance daily candles without preparing or storing them."""

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        """Fetch candles for inclusive dates and map available fields as-is."""

        try:
            yahoo = import_module("yfinance")
        except ImportError as exc:
            raise RuntimeError(
                "The Yahoo Finance adapter requires yfinance in the environment."
            ) from exc

        end_exclusive = end + timedelta(days=1)

        def fetch_history() -> Any:
            ticker = yahoo.Ticker(symbol)
            return ticker.history(
                start=start.isoformat(),
                end=end_exclusive.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=True,
            )

        history = await asyncio.to_thread(fetch_history)
        if history.empty:
            return ()

        records: list[Mapping[str, object]] = []
        for observation_time, row in history.iterrows():
            if isinstance(observation_time, datetime):
                observed_at = observation_time
            elif callable(getattr(observation_time, "to_pydatetime", None)):
                observed_at = observation_time.to_pydatetime()
            else:
                observed_at = None

            record: dict[str, object] = {"symbol": symbol}
            if observed_at is not None:
                record["observed_at"] = observed_at
            for yahoo_field, canonical_field in _YAHOO_FIELDS.items():
                if yahoo_field in history.columns:
                    record[canonical_field] = row[yahoo_field]
            records.append(record)

        return tuple(records)
