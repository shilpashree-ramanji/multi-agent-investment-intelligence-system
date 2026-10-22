"""Provider-neutral interface for retrieving screened stock candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from multi_agent_investment_system.data.candidates import ScreenedCandidate


class CandidateScreeningProvider(Protocol):
    """Retrieval-only contract, separate from daily market-data providers.

    ``screen`` is an opaque source-specific screen identity or query supplied
    by the caller. The provider returns normalized candidate records; it does
    not define or infer screening rules. A successful query with no matches
    returns an empty sequence. Retrieval errors must propagate as exceptions.
    """

    async def get_candidates(self, screen: str) -> Sequence[ScreenedCandidate]:
        """Return candidates and available source-reported screening context."""
