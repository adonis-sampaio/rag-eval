"""Eyeball how each retriever behaves on the sample queries.

Marked [HIT] when the chunk overlaps a gold span. Use this to sanity-
check ranking changes; use the evaluation step for numbers.
"""

from __future__ import annotations

from rag_eval.chunking.recursive import RecursiveChunker
from rag_eval.data.loader import load_corpus, load_queries
from rag_eval.data.schema import is_relevant
from rag_eval.embedding.hashing import HashingEmbedder
from rag_eval.retrieval.bm25 import BM25Retriever
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.hybrid import HybridRetriever


def main() -> None:
    docs = load_corpus("data/sample/corpus.jsonl")
    queries = load_queries("data/sample/queries.jsonl", docs)
    chunks = RecursiveChunker(size=200).chunk(docs)

    embedder = HashingEmbedder(dim=256)
    retrievers = [
        DenseRetriever(embedder),
        BM25Retriever(),
        HybridRetriever([DenseRetriever(embedder), BM25Retriever()], rrf_k=60),
    ]
    for r in retrievers:
        r.index(chunks)

    for q in queries:
        print(f"\n{q.id}: {q.text}")
        for r in retrievers:
            print(f"  {r.name}")
            for rank, sc in enumerate(r.retrieve(q.text, top_n=3), start=1):
                tag = "HIT" if is_relevant(sc.chunk, q.gold) else "   "
                snippet = sc.chunk.text[:55].replace("\n", " ")
                print(
                    f"    {rank}. [{tag}] {sc.chunk.id:<6} "
                    f"score={sc.score:.3f}  {snippet}"
                )


if __name__ == "__main__":
    main()