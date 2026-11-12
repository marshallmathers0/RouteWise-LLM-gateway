"""Embedding model (bi-encoder) and duplicate verifier (cross-encoder).

Bi-encoder: encodes each query ON ITS OWN into a vector. Vectors can be precomputed and
indexed, so comparing a new query against 100k cached ones is one fast vector search.
Cross-encoder: reads BOTH queries TOGETHER through one transformer, so attention can
compare "India" with "Indonesia" token by token. Much more accurate, but it must run
once per pair, so we only use it on the handful of candidates the bi-encoder found.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors, shape (n, dim)."""
        ...


class Verifier(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(duplicate) in [0, 1] for each pair."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # Normalized vectors => inner product == cosine similarity (what FAISS IndexFlatIP computes).
        # No BGE "query instruction" prefix: that's for query->passage retrieval; we compare
        # query<->query, a symmetric task where the plain encoding is the right one.
        vecs = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class CrossEncoderVerifier:
    def __init__(self, model_name: str = "cross-encoder/quora-distilroberta-base", device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        # Single-logit model: predict() applies a sigmoid, giving P(duplicate) in [0, 1].
        scores = self.model.predict(list(pairs), show_progress_bar=False)
        return [float(s) for s in np.atleast_1d(scores)]
