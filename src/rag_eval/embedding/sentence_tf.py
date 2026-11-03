"""Optional: real SentenceTransformer embeddings.

Not imported by default so the project stays dependency-free.
Install with:

    pip install sentence-transformers

Then use:

    from rag_eval.embedding.sentence_tf import SentenceTFEmbedder
    embedder = SentenceTFEmbedder("all-MiniLM-L6-v2")
"""


from __future__ import annotations

from typing import Sequence


class SentenceTFEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTFEmbedder requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.name = f"sentence_tf({model_name})"
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        arr = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return arr.tolist()