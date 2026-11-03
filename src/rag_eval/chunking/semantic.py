"""Semantic chunker: group sentences whose embeddings are similar.

The chunker depends on an `Embedder`, which is
injected. That is deliberate: it means the chunker is testable today
with a stub embedder, and later you swap in a real one without
touching this file. This is the dependency-injection pattern the whole
project leans on.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from rag_eval.data.schema import Chunk, Document
from rag_eval.protocols import Embedder

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.5,
        max_size: int = 1024,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self.name = f"semantic(threshold={threshold},max_size={max_size})"

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        sentences = self._sentences(doc.text)
        if not sentences:
            return []

        embeddings = self.embedder.embed([text for _, _, text in sentences])

        # Greedy grouping: start a new group when either the semantic
        # similarity drops below threshold OR the group would exceed max_size.
        groups: list[list[tuple[int, int, str]]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            similarity = _cosine(prev_emb, curr_emb)
            current_len = sum(end - start for start, end, _ in groups[-1])
            next_len = sentences[i][1] - sentences[i][0]
            if similarity < self.threshold or current_len + next_len > self.max_size:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        return [
            Chunk(
                id=f"{doc.id}#{i}",
                doc_id=doc.id,
                text=doc.text[group[0][0] : group[-1][1]],
                start=group[0][0],
                end=group[-1][1],
            )
            for i, group in enumerate(groups)
        ]

    @staticmethod
    def _sentences(text: str) -> list[tuple[int, int, str]]:
        """Split into (start, end, text) sentence tuples, preserving offsets."""
        if not text.strip():
            return []
        bounds = [0]
        for match in _SENTENCE_BOUNDARY.finditer(text):
            bounds.append(match.end())
        bounds.append(len(text))
        return [
            (bounds[i], bounds[i + 1], text[bounds[i] : bounds[i + 1]])
            for i in range(len(bounds) - 1)
            if bounds[i] < bounds[i + 1]
        ]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1e-9
    db = math.sqrt(sum(x * x for x in b)) or 1e-9
    return num / (da * db)