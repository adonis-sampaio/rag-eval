"""Load the corpus and the gold query set from JSONL files.

Gold spans may be written two ways:

    {"doc_id": "D1", "start": 20, "end": 38}       # explicit offsets
    {"doc_id": "D1", "text": "API key is invalid"} # resolved at load time

The second form exists because hand-maintained offsets rot the moment
someone edits a document. `load_queries` resolves text to offsets once,
then the rest of the system only ever sees offsets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from rag_eval.data.schema import Document, GoldSpan, Query


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path: str | Path) -> list[Document]:
    return [
        Document(id=row["id"], text=row["text"], meta=row.get("meta", {}))
        for row in _read_jsonl(path)
    ]


def load_queries(
    path: str | Path, documents: Sequence[Document]
) -> list[Query]:
    by_id = {doc.id: doc for doc in documents}
    queries: list[Query] = []

    for row in _read_jsonl(path):
        gold: list[GoldSpan] = []
        for g in row.get("gold", []):
            doc = by_id[g["doc_id"]]

            if "text" in g:
                start = doc.text.find(g["text"])
                if start < 0:  # fail loudly, not silently
                    raise ValueError(
                        f"query {row['id']}: gold text not found in {doc.id}"
                    )
                end = start + len(g["text"])
            else:
                start, end = g["start"], g["end"]

            gold.append(GoldSpan(doc_id=doc.id, start=start, end=end))

        queries.append(
            Query(
                id=row["id"],
                text=row["text"],
                gold=gold,
                tags=row.get("tags", []),
            )
        )

    return queries


def validate(queries: Sequence[Query], documents: Sequence[Document]) -> list[str]:
    """Return human-readable problems. Empty list means the data is clean."""
    by_id = {doc.id: doc for doc in documents}
    problems: list[str] = []

    for q in queries:
        if not q.gold:
            problems.append(f"{q.id}: no gold spans")
        for g in q.gold:
            doc = by_id.get(g.doc_id)
            if doc is None:
                problems.append(f"{q.id}: unknown doc_id {g.doc_id!r}")
            elif not (0 <= g.start < g.end <= len(doc.text)):
                problems.append(
                    f"{q.id}: span {g.start}:{g.end} outside {g.doc_id} "
                    f"(length {len(doc.text)})"
                )

    return problems