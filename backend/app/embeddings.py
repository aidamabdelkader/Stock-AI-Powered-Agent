from __future__ import annotations

from functools import cached_property
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy local semantic embedding adapter for the intended assessment path."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()


class HashingEmbedder:
    """Dependency-light smoke-test embedding provider.

    This is intentionally lexical and must not be presented as the final semantic
    embedding model. It exists so API, evaluation, and audit mechanics can run in
    restricted environments that cannot download model packages.
    """

    def __init__(self, n_features: int = 768) -> None:
        self.model_name = f"sklearn-hashing-{n_features}"
        self.n_features = n_features

    @cached_property
    def vectorizer(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        return HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.vectorizer.transform(texts).toarray().tolist()
