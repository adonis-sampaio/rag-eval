"""Chunking-specific quality metrics.

These are NOT the retriever metrics (those live in evaluation/). These
answer one question only: *did the chunker leave the gold evidence
findable at all?* Whatever chunker scores lowest here puts a hard
ceiling on every downstream metric.
"""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import Chunk, GoldSpan, Query


def _covers(chunk: Chunk, span: GoldSpan) -> bool:
    return chunk.doc_id == span.doc_id and span.overlaps(chunk.start, chunk.end)


def chunking_recall(chunks: Sequence[Chunk], queries: Sequence[Query]) -> float:
    """Fraction of gold spans covered by at least one chunk.

    This is the *ceiling* on any end-to-end retrieval metric: if it is
    0.9, no retriever + ranker combination can exceed 0.9 recall.
    """
    total = sum(len(q.gold) for q in queries)
    if total == 0:
        return 0.0
    covered = sum(
        1
        for q in queries
        for span in q.gold
        if any(_covers(chunk, span) for chunk in chunks)
    )
    return covered / total


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    """Basic shape of the chunk set: count, length distribution."""
    if not chunks:
        return {"n": 0, "avg_len": 0.0, "min_len": 0, "max_len": 0}
    lengths = [c.length for c in chunks]
    return {
        "n": len(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }


def chunks_per_query(chunks: Sequence[Chunk], queries: Sequence[Query]) -> float:
    """Average number of chunks that overlap a query's gold spans.

    Low is good: the answer is concentrated. High means the answer is
    scattered, which taxes retrieval and wastes context budget.
    """
    if not queries:
        return 0.0
    counts: list[int] = []
    for q in queries:
        covered = {c.id for c in chunks if any(_covers(c, s) for s in q.gold)}
        counts.append(len(covered))
    return sum(counts) / len(counts)