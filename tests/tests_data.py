from rag_eval.data.loader import load_corpus, load_queries, validate
from rag_eval.data.schema import Chunk, GoldSpan, is_relevant


def _load():
    docs = load_corpus("data/sample/corpus.jsonl")
    return docs, load_queries("data/sample/queries.jsonl", docs)


def test_dataset_is_valid():
    docs, queries = _load()
    assert validate(queries, docs) == []


def test_gold_text_resolves_to_offsets():
    docs, queries = _load()
    q1 = next(q for q in queries if q.id == "Q1")
    doc = next(d for d in docs if d.id == "D1")
    span = q1.gold[0]
    assert doc.text[span.start : span.end] == "Error 401 means the API key is invalid"


def test_overlap_is_half_open():
    gold = [GoldSpan(doc_id="D1", start=10, end=20)]
    touching = Chunk(id="c", doc_id="D1", text="", start=20, end=30)
    overlapping = Chunk(id="c", doc_id="D1", text="", start=15, end=25)
    assert not is_relevant(touching, gold)   # adjacent != overlapping
    assert is_relevant(overlapping, gold)


def test_relevance_is_scoped_per_document():
    # Same offsets on a *different* document must not count as relevant.
    gold = [GoldSpan(doc_id="D1", start=10, end=20)]
    other_doc = Chunk(id="c", doc_id="D2", text="", start=10, end=20)
    assert not is_relevant(other_doc, gold)