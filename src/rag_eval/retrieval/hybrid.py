"""Hybrid retriever using Reciprocal Rank Fusion (RRF).

RRF combines rankings by *rank*, not by score -- which is why it works
across retrievers whose scores are on completely different scales
(cosine similarity vs BM25). Each underlying retriever contributes
    1 / (rrf_k + rank)
to every chunk it retrieved, and the totals decide the final order.
rrf_k=60 is the value from the original RRF paper and behaves well
across domains.
"""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import Chunk, ScoredChunk
from rag_eval.protocols import Retriever


class HybridRetriever:
    def __init__(
        self, retrievers: Sequence[Retriever], rrf_k: int = 60
    ) -> None:
        if not retrievers:
            raise ValueError("need at least one underlying retriever")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        names = "+".join(r.name for r in self.retrievers)
        self.name = f"hybrid({names},rrf_k={rrf_k})"

    def index(self, chunks: Sequence[Chunk]) -> None:
        # Each underlying retriever builds its own index. Calling index()
        # here makes the hybrid interchangeable with any other retriever.
        for r in self.retrievers:
            r.index(chunks)

    def retrieve(self, query: str, top_n: int = 10) -> list[ScoredChunk]:
        # Ask each retriever for a wider pool than top_n -- the whole
        # point of fusion is to combine candidates, so over-fetch.
        pool_size = max(top_n * 2, top_n + 5)
        accumulator: dict[str, tuple[Chunk, float]] = {}

        for retriever in self.retrievers:
            for rank, scored in enumerate(
                retriever.retrieve(query, top_n=pool_size), start=1
            ):
                cid = scored.chunk.id
                contribution = 1.0 / (self.rrf_k + rank)
                if cid in accumulator:
                    chunk, total = accumulator[cid]
                    accumulator[cid] = (chunk, total + contribution)
                else:
                    accumulator[cid] = (scored.chunk, contribution)

        fused = [
            ScoredChunk(chunk=chunk, score=score)
            for chunk, score in accumulator.values()
        ]
        fused.sort(key=lambda sc: sc.score, reverse=True)
        return fused[:top_n]