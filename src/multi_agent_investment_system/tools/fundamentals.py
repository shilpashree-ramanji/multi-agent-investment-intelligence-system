"""Provider-independent contract for company financial statements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from multi_agent_investment_system.data.fundamentals import CompanyFundamentalsPeriod


class CompanyFundamentalsProvider(Protocol):
    """Retrieve source-normalized financial statement periods for a listing.

    The provider owns source retrieval and mapping into the existing canonical
    model. It must preserve source-reported values, units, basis, periods, and
    available provenance without deriving missing fields. A successful lookup
    with no statements returns an empty sequence; retrieval failures raise an
    exception and must not be represented as an empty result.
    """

    async def get_company_fundamentals(
        self, exchange: str, trading_symbol: str
    ) -> Sequence[CompanyFundamentalsPeriod]:
        """Return available statement-period records for the requested listing."""
