from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .embeddings import EmbeddingProvider
from .ingestion import corpus_version
from .models import ChunkRecord, IndexSummary


class VectorCorpusStore(Protocol):
    collection_name: str

    def rebuild(self, chunks: list[ChunkRecord]) -> IndexSummary: ...
    def load_chunks(self) -> list[ChunkRecord]: ...
    def manifest(self) -> dict[str, Any]: ...
    def count(self) -> int: ...
    def query_dense(self, query_embedding: list[float], n_results: int) -> dict[str, float]: ...


class BaseManifestStore:
    def __init__(self, *, manifest_path: Path, collection_name: str, embedder: EmbeddingProvider) -> None:
        self.manifest_path = manifest_path
        self.collection_name = collection_name
        self.embedder = embedder
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self, chunks: list[ChunkRecord], version: str) -> None:
        manifest = {
            "corpus_version": version,
            "collection_name": self.collection_name,
            "embedding_model": self.embedder.model_name,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_chunks(self) -> list[ChunkRecord]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [ChunkRecord.model_validate(item) for item in payload.get("chunks", [])]

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


class ChromaCorpusStore(BaseManifestStore):
    def __init__(
        self,
        *,
        chroma_path: Path,
        manifest_path: Path,
        collection_name: str,
        embedder: EmbeddingProvider,
    ) -> None:
        super().__init__(manifest_path=manifest_path, collection_name=collection_name, embedder=embedder)
        import chromadb

        self.chroma_path = chroma_path
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def rebuild(self, chunks: list[ChunkRecord]) -> IndexSummary:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings[start : start + batch_size],
                metadatas=[self._metadata(chunk) for chunk in batch],
            )

        version = corpus_version(chunks)
        self._write_manifest(chunks, version)
        return IndexSummary(
            articles_indexed=len({chunk.article_id for chunk in chunks}),
            chunks_indexed=len(chunks),
            corpus_version=version,
            collection_name=self.collection_name,
        )

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def query_dense(self, query_embedding: list[float], n_results: int) -> dict[str, float]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count()),
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return {
            chunk_id: max(0.0, min(1.0, 1.0 - float(distance)))
            for chunk_id, distance in zip(ids, distances, strict=False)
        }

    @staticmethod
    def _metadata(chunk: ChunkRecord) -> dict[str, str | int | float | bool]:
        return {
            "article_id": chunk.article_id,
            "article_title": chunk.article_title,
            "source": chunk.source,
            "url": chunk.url or "",
            "published_at": chunk.published_at or "",
            "tickers": ",".join(chunk.tickers),
            "chunk_index": chunk.chunk_index,
            "article_hash": chunk.article_hash,
        }


class MemoryCorpusStore(BaseManifestStore):
    """In-process cosine vector store for restricted smoke-test environments."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        collection_name: str,
        embedder: EmbeddingProvider,
    ) -> None:
        super().__init__(manifest_path=manifest_path, collection_name=collection_name, embedder=embedder)
        self._chunks: list[ChunkRecord] = []
        self._embeddings = np.empty((0, 0), dtype=float)
        self._hydrate()

    def _hydrate(self) -> None:
        self._chunks = self.load_chunks()
        if self._chunks:
            self._embeddings = np.asarray(self.embedder.encode([chunk.text for chunk in self._chunks]), dtype=float)
        else:
            self._embeddings = np.empty((0, 0), dtype=float)

    def rebuild(self, chunks: list[ChunkRecord]) -> IndexSummary:
        self._chunks = chunks
        self._embeddings = np.asarray(self.embedder.encode([chunk.text for chunk in chunks]), dtype=float)
        version = corpus_version(chunks)
        self._write_manifest(chunks, version)
        return IndexSummary(
            articles_indexed=len({chunk.article_id for chunk in chunks}),
            chunks_indexed=len(chunks),
            corpus_version=version,
            collection_name=self.collection_name,
        )

    def count(self) -> int:
        return len(self._chunks)

    def query_dense(self, query_embedding: list[float], n_results: int) -> dict[str, float]:
        if not self._chunks or self._embeddings.size == 0:
            return {}
        query = np.asarray(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query)
        row_norms = np.linalg.norm(self._embeddings, axis=1)
        denominator = np.maximum(row_norms * max(query_norm, 1e-12), 1e-12)
        scores = (self._embeddings @ query) / denominator
        indices = np.argsort(scores)[::-1][: min(n_results, len(self._chunks))]
        return {
            self._chunks[index].id: max(0.0, min(1.0, float(scores[index])))
            for index in indices
        }
