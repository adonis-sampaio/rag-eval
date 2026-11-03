"""Recursive chunker: split on the largest natural boundary that fits.

The idea, borrowed from LangChain's RecursiveCharacterTextSplitter, is
simple: try paragraph, then line, then sentence, then word, then hard
cut. Each recursive step only runs when the piece is still too big.
"""

from __future__ import annotations

from typing import Sequence

from rag_eval.data.schema import Chunk, Document


class RecursiveChunker:
    DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(
        self, size: int = 512, separators: Sequence[str] | None = None
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.separators = tuple(separators or self.DEFAULT_SEPARATORS)
        self.name = f"recursive(size={size})"

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = self._split(doc.text)
            cursor = 0
            for i, piece in enumerate(pieces):
                start = doc.text.find(piece, cursor)
                if start < 0:  # pieces are contiguous substrings; must find
                    raise RuntimeError(
                        f"piece not found in {doc.id} at offset {cursor}: {piece!r}"
                    )
                end = start + len(piece)
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{i}",
                        doc_id=doc.id,
                        text=piece,
                        start=start,
                        end=end,
                    )
                )
                cursor = end
        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.size:
            return [text]

        for sep in self.separators:
            if sep in text:
                pieces: list[str] = []
                parts = text.split(sep)
                for j, part in enumerate(parts):
                    # Reattach separator to every part but the last,
                    # so concatenating pieces reproduces the original text.
                    piece = part + sep if j < len(parts) - 1 else part
                    if not piece:
                        continue
                    if len(piece) <= self.size:
                        pieces.append(piece)
                    else:
                        pieces.extend(self._split(piece))
                return self._merge(pieces)

        # No separator applies (e.g. one giant token). Hard cut.
        return [text[i : i + self.size] for i in range(0, len(text), self.size)]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily glue small adjacent pieces back together up to `size`."""
        merged: list[str] = []
        buf = ""
        for piece in pieces:
            if len(buf) + len(piece) <= self.size:
                buf += piece
            else:
                if buf:
                    merged.append(buf)
                buf = piece
        if buf:
            merged.append(buf)
        return merged