"""Compare rankers on the sample dataset, and show the gold chunk's rank.

The columns `pre` and `post` are exactly what step 6's attribution
needs: if `pre` is good and `post` is missing or worse, the ranker
demoted the evidence. If `pre` is already missing, the ranker had
nothing to do with the failure.
"""

from __future__ import annotations

from rag_eval.chunking.recursive import RecursiveChunker
from rag_eval.data.loader import load_corpus, load_queries
from rag_eval.data.schema import ScoredChunk, is_relevant
from rag_eval.embedding.hashing import HashingEmbedder
from rag_eval.ranking.lexical import LexicalOverlapRanker
from rag_eval.ranking.none import NoneRanker
from rag_eval.retrieval.dense import DenseRetriever


def gold_rank(scored: list[ScoredChunk], gold) -> int | None:
    for rank, sc in enumerate(scored, start=1):
        if is_relevant(sc.chunk, gold):
            return rank
    return None


def main() -> None:
    docs = load_corpus("data/sample/corpus.jsonl")
    queries = load_queries("data/sample/queries.jsonl", docs)
    chunks = RecursiveChunker(size=200).chunk(docs)

    retriever = DenseRetriever(HashingEmbedder(dim=256))
    retriever.index(chunks)

    rankers = [NoneRanker(), LexicalOverlapRanker()]

    print(f"retriever: {retriever.name}")
    print(f"{'query':<6} {'ranker':<20} {'pre':>5} {'post':>5}  {'status'}")
    print("-" * 70)

    for q in queries:
        candidates = retriever.retrieve(q.text, top_n=10)
        pre = gold_rank(candidates, q.gold)
        for ranker in rankers:
            ranked = ranker.rank(q.text, candidates, top_k=5)
            post = gold_rank(ranked, q.gold)
            status = _classify(pre, post, top_k=5)
            pre_s = "-" if pre is None else str(pre)
            post_s = "-" if post is None else str(post)
            print(
                f"{q.id:<6} {ranker.name:<20} {pre_s:>5} {post_s:>5}  {status}"
            )


def _classify(pre: int | None, post: int | None, top_k: int) -> str:
    if pre is None:
        return "ranker innocent (retriever missed)"
    if post is None:
        return "ranker demoted evidence OUT of top-k"
    if post < pre:
        return "ranker promoted"
    if post > pre:
        return "ranker demoted"
    return "unchanged"


if __name__ == "__main__":
    main()