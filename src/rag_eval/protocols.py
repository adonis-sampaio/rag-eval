"""The four interfaces every strategy implements.

The runner imports only these, never concrete classes. Adding a new
strategy = writing one class and registering it. No pipeline changes.

Note the split between `top_n` (candidate pool, pre-ranking) and
`top_k` (final list, post-ranking). That distinction is what lets the
diagnostics tell "never retrieved" apart from "retrieved and demoted".
"""

from __future__ import annotations

from typing import Protocol, Sequence

from rag_eval.data.schema import Chunk, Document, ScoredChunk


class Chunker(Protocol):
    name: str

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]: ...


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Retriever(Protocol):
    name: str

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def retrieve(self, query: str, top_n: int) -> list[ScoredChunk]: ...


class Ranker(Protocol):
    name: str

    def rank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...