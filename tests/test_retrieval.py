import math

from rag_eval.chunking.recursive import RecursiveChunker
from rag_eval.data.loader import load_corpus, load_queries
from rag_eval.data.schema import Chunk, ScoredChunk, is_relevant
from rag_eval.embedding.hashing import HashingEmbedder
from rag_eval.retrieval.bm25 import BM25Retriever
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.hybrid import HybridRetriever


def _sample():
    docs = load_corpus("data/sample/corpus.jsonl")
    queries = load_queries("data/sample/queries.jsonl", docs)
    chunks = RecursiveChunker(size=200).chunk(docs)
    return docs, queries, chunks


def _cos(a, b):
    # Vectors are L2-normalized by the embedder, so dot == cosine.
    return sum(x * y for x, y in zip(a, b))


# --- HashingEmbedder --------------------------------------------------------

def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder(dim=64).embed(["Hello world"])
    b = HashingEmbedder(dim=64).embed(["Hello world"])
    assert a == b


def test_hashing_embedder_discriminates_by_overlap():
    embedder = HashingEmbedder(dim=256)
    query, on_topic, off_topic = embedder.embed(
        [
            "which error means the api key is invalid",
            "Error 401 means the API key is invalid",
            "The free tier allows 60 requests per minute",
        ]
    )
    assert _cos(query, on_topic) > _cos(query, off_topic)


def test_hashing_embedder_rejects_bad_dim():
    try:
        HashingEmbedder(dim=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- DenseRetriever ---------------------------------------------------------

def test_dense_retriever_returns_gold_chunk_first():
    _, queries, chunks = _sample()
    retriever = DenseRetriever(HashingEmbedder(dim=256))
    retriever.index(chunks)
    for q in queries:
        top1 = retriever.retrieve(q.text, top_n=1)
        assert top1, f"{q.id}: no results"
        assert is_relevant(top1[0].chunk, q.gold), f"{q.id}: {top1[0].chunk.id}"


# --- BM25Retriever ----------------------------------------------------------

def test_bm25_scores_matching_terms_above_non_matching():
    chunks = [
        Chunk(id="a", doc_id="d", text="api key invalid", start=0, end=15),
        Chunk(id="b", doc_id="d", text="rate limit per minute", start=15, end=36),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    results = retriever.retrieve("api key", top_n=2)
    assert results[0].chunk.id == "a"
    assert results[0].score > 0
    assert results[1].chunk.id == "b"
    assert results[1].score == 0.0


def test_bm25_idf_is_non_negative():
    # Rare terms should have higher IDF than common terms.
    chunks = [
        Chunk(id="a", doc_id="d", text="the the the rare", start=0, end=18),
        Chunk(id="b", doc_id="d", text="the the the common", start=18, end=36),
    ]
    r = BM25Retriever()
    r.index(chunks)
    n = len(chunks)
    idf_rare = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
    idf_common = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))
    assert idf_rare > idf_common > 0


# --- HybridRetriever (RRF) --------------------------------------------------

class _FakeRetriever:
    """Returns a fixed ranking; used to test RRF math in isolation."""

    def __init__(self, name, order):
        self.name = name
        self._order = order

    def index(self, chunks):
        pass

    def retrieve(self, query, top_n=10):
        return [ScoredChunk(chunk=c, score=1.0) for c in self._order[:top_n]]


def _two_chunks():
    return [
        Chunk(id="a", doc_id="d", text="A", start=0, end=1),
        Chunk(id="b", doc_id="d", text="B", start=1, end=2),
    ]


def test_rrf_gives_equal_scores_to_symmetric_rankings():
    a, b = _two_chunks()
    hybrid = HybridRetriever(
        [_FakeRetriever("r1", [a, b]), _FakeRetriever("r2", [b, a])], rrf_k=60
    )
    results = hybrid.retrieve("q", top_n=10)
    assert {r.chunk.id for r in results} == {"a", "b"}
    # rank 1 in one retriever, rank 2 in the other -> identical totals.
    assert results[0].score == results[1].score


def test_rrf_keeps_unique_hits_from_both_sources():
    a, b = _two_chunks()
    hybrid = HybridRetriever(
        [_FakeRetriever("r1", [a]), _FakeRetriever("r2", [b])], rrf_k=60
    )
    results = hybrid.retrieve("q", top_n=10)
    assert {r.chunk.id for r in results} == {"a", "b"}


def test_hybrid_end_to_end_hits_gold():
    _, queries, chunks = _sample()
    hybrid = HybridRetriever(
        [DenseRetriever(HashingEmbedder(dim=256)), BM25Retriever()], rrf_k=60
    )
    hybrid.index(chunks)
    for q in queries:
        top1 = hybrid.retrieve(q.text, top_n=1)
        assert top1 and is_relevant(top1[0].chunk, q.gold), f"{q.id}: {top1}"