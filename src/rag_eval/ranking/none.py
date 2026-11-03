"""Identity ranker: keep the retriever's order, just truncate.

This is the baseline. Every other ranker must beat it by enough to
justify its latency and its extra model weights. When the diagnostics
say "the ranker demoted the gold chunk", this is the run that
*didn't* demote it -- which is how you prove the ranker is the cause.
"""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import ScoredChunk


class NoneRanker:
    name = "none"

    def rank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int = 10
    ) -> list[ScoredChunk]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        return list(candidates[:top_k])