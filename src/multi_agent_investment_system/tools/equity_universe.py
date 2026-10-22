"""Provider interface for refreshing the eligible NSE equity universe."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from multi_agent_investment_system.data.candidates import NSEEquityUniverseMember


class NSEEquityUniverseProvider(Protocol):
    """Retrieve a current, normalized NSE-equity universe snapshot.

    Implementations must use an authorized and maintainable source. This
    contract does not select a source or prescribe its retrieval mechanism.
    An empty sequence means a successful refresh with no eligible listings;
    failures must raise rather than return an empty sequence. Records retain
    source-reported eligibility and provenance.
    """

    async def refresh_universe(self) -> Sequence[NSEEquityUniverseMember]:
        """Fetch the latest available universe snapshot."""
