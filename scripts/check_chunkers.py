"""Compare every chunker on the sample dataset.

Output columns:
  n       total chunks produced
  avg     average chunk length (chars)
  min/max shortest / longest chunk
  recall  fraction of gold spans that survived chunking

Recall is the column that matters. Everything else is context for
judging whether the chunker is doing sensible work.
"""

from __future__ import annotations

from rag_eval.chunking.fixed import FixedChunker
from rag_eval.chunking.recursive import RecursiveChunker
from rag_eval.chunking.semantic import SemanticChunker
from rag_eval.chunking.quality import chunk_stats, chunking_recall
from rag_eval.data.loader import load_corpus, load_queries


class StubEmbedder:
    """Trivial embedder so SemanticChunker can run before step 3.

    Character-frequency vector: cheap, deterministic, no dependencies.
    Not good -- just enough to exercise the grouping logic. Step 3
    replaces this with a real embedder.
    """

    name = "stub"
    dim = 26

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vec[ord(ch) - ord("a")] += 1
            out.append(vec)
        return out


def main() -> None:
    docs = load_corpus("data/sample/corpus.jsonl")
    queries = load_queries("data/sample/queries.jsonl", docs)

    chunkers = [
        FixedChunker(size=64, overlap=8),
        RecursiveChunker(size=80),
        SemanticChunker(StubEmbedder(), threshold=0.5, max_size=200),
    ]

    print(f"{'chunker':<40} {'n':>4} {'avg':>5} {'min':>5} {'max':>5} {'recall':>7}")
    print("-" * 78)
    for chunker in chunkers:
        chunks = chunker.chunk(docs)
        stats = chunk_stats(chunks)
        recall = chunking_recall(chunks, queries)
        print(
            f"{chunker.name:<40} "
            f"{stats['n']:>4} "
            f"{stats['avg_len']:>5.0f} "
            f"{stats['min_len']:>5} "
            f"{stats['max_len']:>5} "
            f"{recall:>7.2f}"
        )


if __name__ == "__main__":
    main()