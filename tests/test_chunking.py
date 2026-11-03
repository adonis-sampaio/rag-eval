from rag_eval.chunking.fixed import FixedChunker
from rag_eval.chunking.recursive import RecursiveChunker
from rag_eval.chunking.semantic import SemanticChunker
from rag_eval.chunking.quality import chunking_recall
from rag_eval.data.loader import load_corpus, load_queries
from rag_eval.data.schema import Chunk, Document


def _stub_embedder():
    class Stub:
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

    return Stub()


def test_fixed_covers_whole_document():
    docs = load_corpus("data/sample/corpus.jsonl")
    chunks = FixedChunker(size=50, overlap=10).chunk(docs)
    for doc in docs:
        doc_chunks = [c for c in chunks if c.doc_id == doc.id]
        assert doc_chunks[0].start == 0
        assert doc_chunks[-1].end == len(doc.text)


def test_fixed_chunks_advance_and_are_bounded():
    docs = [Document(id="x", text="a" * 100)]
    chunks = FixedChunker(size=30, overlap=5).chunk(docs)
    assert all(c.end - c.start <= 30 for c in chunks)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    assert starts[0] == 0


def test_recursive_respects_max_size():
    docs = [Document(id="x", text="one. two. three. four. five. six.")]
    chunks = RecursiveChunker(size=10).chunk(docs)
    assert all(c.length <= 10 for c in chunks)


def test_recursive_pieces_reconstruct_document():
    docs = [Document(id="x", text="a. b. c. d. e. f. g. h.")]
    chunks = RecursiveChunker(size=8).chunk(docs)
    # Contiguity: each chunk starts where the previous one ended.
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.start == prev.end


def test_semantic_chunker_offsets_are_contiguous():
    docs = [Document(id="x", text="First. Second. Third. Fourth. Fifth.")]
    chunks = SemanticChunker(_stub_embedder(), threshold=0.99, max_size=100).chunk(docs)
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.start == prev.end


def test_all_chunkers_preserve_gold_recall():
    docs = load_corpus("data/sample/corpus.jsonl")
    queries = load_queries("data/sample/queries.jsonl", docs)
    chunkers = [
        FixedChunker(size=50, overlap=10),
        RecursiveChunker(size=80),
        SemanticChunker(_stub_embedder(), threshold=0.3, max_size=200),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(docs)
        assert chunking_recall(chunks, queries) == 1.0, (
            f"{chunker.name} dropped gold evidence"
        )