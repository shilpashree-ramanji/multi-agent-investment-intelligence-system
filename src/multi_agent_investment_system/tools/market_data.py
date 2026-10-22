"""Provider-neutral market-data interface with no network implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol


class MarketDataProvider(Protocol):
    """Retrieval-only contract for future daily market-data source adapters.

    Each returned mapping uses lowercase ``open``, ``high``, ``low``,
    ``close``, and ``volume`` keys. When available, ``symbol`` and a
    timezone-aware ``observed_at`` datetime identify the security and candle
    time. Adapters may include source-specific metadata; consumers should
    preserve it and pass records to the shared preparation function.

    ``start`` and ``end`` are inclusive calendar-date boundaries. An adapter
    must honor these interface bounds even if its native source uses different
    range semantics. A successful query with no matching candles returns an
    empty sequence. Retrieval failures must raise an exception; they must not
    be reported as an empty result.
    """

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        """Retrieve daily candle mappings without validating or storing them."""
