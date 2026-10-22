"""Provider-independent source-document and chunk data structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Document:
    """Source content with provenance metadata for later evidence retrieval."""

    document_id: str
    source_uri: str
    content: str
    captured_at: datetime
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A text span linked to its source document."""

    chunk_id: str
    document_id: str
    content: str
    start_offset: int
    end_offset: int
