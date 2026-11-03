"""Okapi BM25 lexical retriever.

The classic. It rewards term overlap and penalizes long documents, and
often beats dense retrieval on short queries with rare keywords --
which is exactly why the hybrid retriever exists.

Parameters follow the standard defaults from the original paper:
k1=1.5 (term-frequency saturation), b=0.75 (length normalization).
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from rag_eval.data.schema import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.name = f"bm25(k1={k1},b={b})"

        self._chunks: list[Chunk] = []
        self._term_freqs: list[dict[str, int]] = []
        self._doc_len: list[int] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_len: float = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._term_freqs = []
        self._doc_len = []
        self._doc_freq = {}

        for chunk in self._chunks:
            tokens = _tokenize(chunk.text)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._term_freqs.append(tf)
            self._doc_len.append(len(tokens))
            for term in tf:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        n = len(self._chunks)
        self._avg_len = sum(self._doc_len) / n if n else 0.0

    def retrieve(self, query: str, top_n: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_terms = _tokenize(query)
        n = len(self._chunks)

        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(self._chunks):
            tf = self._term_freqs[i]
            dl = self._doc_len[i]
            score = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (
                    1 - self.b + self.b * dl / (self._avg_len or 1.0)
                )
                score += idf * (f * (self.k1 + 1)) / denom
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_n]