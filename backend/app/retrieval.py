from __future__ import annotations

import math
from collections import Counter, defaultdict

from .embeddings import EmbeddingProvider
from .models import ChunkRecord, RetrievedChunk
from .store import VectorCorpusStore
from .text_utils import tokenize

## Since we need the most optimimum retrieval 
## I had used BM25 index for the exact word matching and the dense retrieval for the semantic meaning 
class BM25Index:
    """Small, dependency-free BM25 implementation for the 8–10 article corpus."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in documents]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.term_frequencies = [Counter(document) for document in documents]
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        self.idf = {
            term: math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for frequencies, doc_len in zip(self.term_frequencies, self.lengths, strict=False):
            score = 0.0
            for term in query_tokens:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-12))
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores

class HybridRetriever:
    """Dense + BM25 retrieval with query expansion and article diversity."""

    COMPANY_ALIASES = {
        "cib": "Commercial International Bank CIB",
        "qnb": "QNB Alahli",
        "egx30": "EGX30 Egyptian Exchange index",
        "eastern tobacco": "Eastern Company Eastern Tobacco",
        "telecom egypt": "Telecom Egypt",
        "fra": "Financial Regulatory Authority FRA",
    }

    def __init__(
        self,
        *,
        store: VectorCorpusStore,
        embedder: EmbeddingProvider,
        dense_k: int,
        lexical_k: int,
        final_top_k: int,
        max_chunks_per_article: int,
        dense_weight: float,
        lexical_weight: float,
        min_score: float,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.dense_k = dense_k
        self.lexical_k = lexical_k
        self.final_top_k = final_top_k
        self.max_chunks_per_article = max_chunks_per_article

        total = dense_weight + lexical_weight
        self.dense_weight = dense_weight / total if total else 0.5
        self.lexical_weight = lexical_weight / total if total else 0.5
        self.min_score = min_score

        self._chunks: list[ChunkRecord] = []
        self._chunk_by_id: dict[str, ChunkRecord] = {}
        self._bm25: BM25Index | None = None

        self.refresh()

    def refresh(self) -> None:
        self._chunks = self.store.load_chunks()
        self._chunk_by_id = {
            chunk.id: chunk
            for chunk in self._chunks
        }

        lexical_documents = [
            tokenize(self._lexical_document(chunk))
            for chunk in self._chunks
        ]

        self._bm25 = (
            BM25Index(lexical_documents)
            if lexical_documents
            else None
        )

    @staticmethod
    def _lexical_document(chunk: ChunkRecord) -> str:
        """Include searchable metadata alongside the article text."""

        return " ".join(
            [
                chunk.article_id,
                chunk.article_title,
                chunk.source,
                " ".join(chunk.tickers),
                chunk.text,
            ]
        )

    def _expand_question(self, question: str) -> str:
        normalized = question.lower()
        additions: list[str] = []

        for alias, expanded_name in self.COMPANY_ALIASES.items():
            if alias in normalized:
                additions.append(expanded_name)

        if not additions:
            return question

        return f"{question} {' '.join(additions)}"

    @property
    def ready(self) -> bool:
        return bool(self._chunks) and self.store.count() > 0

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if not self.ready:
            return []

        expanded_question = self._expand_question(question)

        dense_scores = self._dense_search(expanded_question)
        lexical_scores = self._lexical_search(expanded_question)

        candidate_ids = set(dense_scores) | set(lexical_scores)
        ranked: list[RetrievedChunk] = []

        for chunk_id in candidate_ids:
            chunk = self._chunk_by_id.get(chunk_id)

            if chunk is None:
                continue

            dense = dense_scores.get(chunk_id, 0.0)
            lexical = lexical_scores.get(chunk_id, 0.0)

            fused = (
                self.dense_weight * dense
                + self.lexical_weight * lexical
            )

            # Preserve strong exact lexical matches even when the embedding
            # model is weak on a short acronym-based question.
            if lexical >= 0.75:
                fused = max(fused, 0.40)

            ranked.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=round(dense, 6),
                    lexical_score=round(lexical, 6),
                    fused_score=round(fused, 6),
                )
            )

        ranked.sort(
            key=lambda item: (
                item.fused_score,
                item.lexical_score,
                item.dense_score,
            ),
            reverse=True,
        )

        selected: list[RetrievedChunk] = []
        article_counts: dict[str, int] = defaultdict(int)

        for item in ranked:
            strong_lexical_match = item.lexical_score >= 0.75

            if (
                item.fused_score < self.min_score
                and not strong_lexical_match
            ):
                continue

            article_id = item.chunk.article_id

            if (
                article_counts[article_id]
                >= self.max_chunks_per_article
            ):
                continue

            selected.append(item)
            article_counts[article_id] += 1

            if len(selected) >= self.final_top_k:
                break

        return selected

    def _dense_search(
        self,
        question: str,
    ) -> dict[str, float]:
        vectors = self.embedder.encode([question])

        if not vectors:
            return {}

        return self.store.query_dense(
            vectors[0],
            self.dense_k,
        )

    def _lexical_search(
        self,
        question: str,
    ) -> dict[str, float]:
        if self._bm25 is None:
            return {}

        raw_scores = self._bm25.get_scores(
            tokenize(question)
        )

        if not raw_scores:
            return {}

        max_score = float(max(raw_scores))

        if max_score <= 0:
            return {}

        ranked_indices = sorted(
            range(len(raw_scores)),
            key=lambda index: raw_scores[index],
            reverse=True,
        )[: self.lexical_k]

        return {
            self._chunks[index].id: max(
                0.0,
                min(
                    1.0,
                    float(raw_scores[index]) / max_score,
                ),
            )
            for index in ranked_indices
            if raw_scores[index] > 0
        }