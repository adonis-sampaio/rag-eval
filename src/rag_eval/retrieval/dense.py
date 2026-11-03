"""Dense retriever: cosine similarity over an embedding matrix.

The matrix is a plain list-of-lists. On real corpora you would swap
this for FAISS or a vector DB -- which is exactly the reason the
`Retriever` protocol exists. Nothing downstream of `retrieve()`
knows how the top-N was produced.
"""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import Chunk, ScoredChunk
from rag_eval.protocols import Embedder


class DenseRetriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.name = f"dense({embedder.name})"
        self._chunks: list[Chunk] = []
        self._matrix: list[list[float]] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        # One batched call: embedders (especially the real ones) like
        # batching. The whole corpus is small enough to embed in one go.
        self._matrix = self.embedder.embed([c.text for c in self._chunks])

    def retrieve(self, query: str, top_n: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        q_vec = self.embedder.embed([query])[0]
        scored = [
            ScoredChunk(chunk=c, score=_dot(q_vec, e))
            for c, e in zip(self._chunks, self._matrix)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_n]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    # Both sides are L2-normalized by the embedder, so dot == cosine.
    return sum(x * y for x, y in zip(a, b))