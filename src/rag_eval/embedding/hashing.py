"""Deterministic bag-of-words embedder using the hashing trick.

This is not a *good* embedder -- it has no semantics. It is a *baseline*
that lets the whole project run with no dependencies, and it is
surprisingly strong for lexical overlap. Swap in SentenceTFEmbedder
step 3.5 for real quality; the retriever code does not change.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.name = f"hashing(dim={dim})"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            # blake2b, NOT hash(): Python's hash() is salted per process,
            # so vectors would differ between runs and breaks determinism.
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dim
            vec[bucket] += 1.0
        return _l2_normalize(vec)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]