"""Retrieval interface; indexing and vector storage remain unimplemented."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from multi_agent_investment_system.rag.documents import DocumentChunk


class Retriever(Protocol):
    """Return source-linked chunks relevant to a query."""

    async def retrieve(self, query: str, *, limit: int) -> Sequence[DocumentChunk]:
        """Retrieve evidence without generating or interpreting an answer."""
