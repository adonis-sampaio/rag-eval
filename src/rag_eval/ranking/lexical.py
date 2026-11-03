"""Dependency-free reranker scoring query-term coverage.

Score = (unique query terms present in the chunk) / (unique query terms)

Interpretable, bounded to [0, 1], deterministic, no model weights. It
is not as good as a cross-encoder, but it is a real reranker -- it
reads the *pair* (query, chunk) rather than a pre-computed vector --
so it is the right fallback when a cross-encoder is unavailable and
the right sanity check when one is.

Ties are broken by chunk id, so results are reproducible across runs.
"""

from __future__ import annotations

import re
from typing import Sequence

from rag_eval.data.schema import ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class LexicalOverlapRanker:
    def __init__(self) -> None:
        self.name = "lexical_overlap"

    def rank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int = 10
    ) -> list[ScoredChunk]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        query_terms = _tokens(query)
        if not query_terms:
            return list(candidates[:top_k])

        rescored = [
            ScoredChunk(
                chunk=sc.chunk,
                score=len(query_terms & _tokens(sc.chunk.text)) / len(query_terms),
            )
            for sc in candidates
        ]
        # Stable sort with an explicit tie-breaker on chunk id.
        rescored.sort(key=lambda sc: (-sc.score, sc.chunk.id))
        return rescored[:top_k]