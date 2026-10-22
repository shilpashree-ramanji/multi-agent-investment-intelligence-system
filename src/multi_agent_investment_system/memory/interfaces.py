"""Provider-neutral key/value memory contract for future agent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """Serializable text record with an explicit namespace and key."""

    namespace: str
    key: str
    value: str


class MemoryStore(Protocol):
    """Minimal async contract; no persistence backend is selected."""

    async def get(self, namespace: str, key: str) -> MemoryRecord | None:
        """Return a stored record, if present."""

    async def put(self, record: MemoryRecord) -> None:
        """Store a record under its namespace and key."""
