"""Cross-encoder reranker with an explicit fallback.

The cross-encoder reads (query, chunk) as a *pair*, so it can score
relevance directly instead of relying on two pre-computed vectors.
That is why it is stronger than dense retrieval and why it is slower.

Optional dependency. If sentence-transformers is not installed, the
constructor requires an explicit `fallback` ranker. This is on purpose:
silently falling back would make an evaluation run claim to measure a
cross-encoder while actually measuring lexical overlap, which is the
exact bug this project exists to prevent. The `name` attribute tells
the truth about which code path executed.
"""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import ScoredChunk
from rag_eval.protocols import Ranker


class CrossEncoderRanker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        fallback: Ranker | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.model_name = model_name

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            if fallback is None:
                raise ImportError(
                    "CrossEncoderRanker requires sentence-transformers. "
                    "Install with: pip install sentence-transformers "
                    "OR pass an explicit `fallback=` ranker."
                ) from exc
            self._model = None
            self._fallback = fallback
            self.name = f"cross_encoder({model_name})->fallback({fallback.name})"
            return

        self._model = CrossEncoder(model_name)
        self._fallback = None
        self.name = f"cross_encoder({model_name})"

    def rank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int = 10
    ) -> list[ScoredChunk]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not candidates:
            return []

        if self._model is None:
            assert self._fallback is not None
            # The fallback already sorts; re-slice defensively.
            return list(self._fallback.rank(query, candidates, top_k=top_k))

        pairs = [(query, sc.chunk.text) for sc in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        rescored = [
            ScoredChunk(chunk=sc.chunk, score=float(score))
            for sc, score in zip(candidates, scores)
        ]
        # Tie-break on chunk id for run-to-run reproducibility.
        rescored.sort(key=lambda sc: (-sc.score, sc.chunk.id))
        return rescored[:top_k]