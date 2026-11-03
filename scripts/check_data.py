"""Sanity-check the dataset before running any experiment."""

from __future__ import annotations

from rag_eval.data.loader import load_corpus, load_queries, validate
from rag_eval.data.schema import Chunk, is_relevant


class WholeDocumentChunker:
    """Trivial chunker used only to prove span projection works.

    Real chunkers arrive in the next step. This one-liner keeps
    check_data.py self-contained so it runs after step 1 alone.
    """

    name = "whole-document"

    def chunk(self, documents):
        return [
            Chunk(id=d.id, doc_id=d.id, text=d.text, start=0, end=len(d.text))
            for d in documents
        ]


def main() -> None:
    docs = load_corpus("data/sample/corpus.jsonl")
    queries = load_queries("data/sample/queries.jsonl", docs)

    problems = validate(queries, docs)
    print(f"{len(docs)} documents, {len(queries)} queries")
    print("problems:", problems or "none")

    chunks = WholeDocumentChunker().chunk(docs)
    for q in queries:
        hits = [c.id for c in chunks if is_relevant(c, q.gold)]
        print(f"{q.id}: relevant chunks -> {hits}")


if __name__ == "__main__":
    main()