"""Core data structures shared by every stage of the pipeline.

Keeping them in one file means a module never has to import another
module's internals -- only this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source document, exactly as it was ingested."""

    id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GoldSpan:
    """Where the answer to a query lives inside a document.

    We label *source spans*, not chunks. Spans stay valid when the
    chunking strategy changes; chunk ids do not.
    Offsets are character offsets: start inclusive, end exclusive.
    """

    doc_id: str
    start: int
    end: int

    def overlaps(self, other_start: int, other_end: int) -> bool:
        # Half-open interval intersection.
        return other_start < self.end and self.start < other_end


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gold: list[GoldSpan]
    tags: list[str] = field(default_factory=list)  # for sliced analysis


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, with its position kept for span matching."""

    id: str
    doc_id: str
    text: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk plus the score some stage assigned to it."""

    chunk: Chunk
    score: float


def is_relevant(chunk: Chunk, gold: list[GoldSpan]) -> bool:
    """A chunk is relevant if it overlaps any gold span in its own document."""
    return any(
        g.doc_id == chunk.doc_id and g.overlaps(chunk.start, chunk.end)
        for g in gold
    )