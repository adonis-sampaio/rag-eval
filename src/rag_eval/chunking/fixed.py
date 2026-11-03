"""Fixed-size chunker with overlap. The baseline every other chunker beats."""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import Chunk, Document


class FixedChunker:
    """Split each document into windows of `size` characters, stepping by
    `size - overlap`. Overlap is what stops an answer that straddles a
    window boundary from being cut in half.

    This is deliberately the dumbest possible chunker. If a smarter
    chunker cannot beat it on your data, the smarter chunker is not
    earning its complexity.
    """

    def __init__(self, size: int = 256, overlap: int = 32) -> None:
        if not 0 <= overlap < size:
            raise ValueError(f"need 0 <= overlap < size, got {overlap=} {size=}")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed(size={size},overlap={overlap})"

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            start = 0
            i = 0
            while start < len(doc.text):
                end = min(start + self.size, len(doc.text))
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{i}",
                        doc_id=doc.id,
                        text=doc.text[start:end],
                        start=start,
                        end=end,
                    )
                )
                i += 1
                if end == len(doc.text):
                    break
                start = end - self.overlap
        return chunks