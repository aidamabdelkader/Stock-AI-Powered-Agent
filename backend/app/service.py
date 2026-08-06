from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .audit import AuditLogger
from .config import Settings, get_settings
from .conversation import classify_conversation
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .ingestion import build_chunks, load_articles
from .llm import (
    AzureOpenAIAnswerGenerator,
    ExtractiveAnswerGenerator,
    OpenAIAnswerGenerator,
)
from .models import AskResponse, HealthResponse, IndexSummary
from .retrieval import HybridRetriever
from .store import ChromaCorpusStore, MemoryCorpusStore
from .workflow import RagWorkflow


class StockNewsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = (
            SentenceTransformerEmbedder(settings.embedding_model)
            if settings.embedding_backend == "sentence_transformer"
            else HashingEmbedder()
        )
        self.store = (
            ChromaCorpusStore(
                chroma_path=settings.chroma_path,
                manifest_path=settings.corpus_manifest_path,
                collection_name=settings.collection_name,
                embedder=self.embedder,
            )
            if settings.vector_backend == "chroma"
            else MemoryCorpusStore(
                manifest_path=settings.corpus_manifest_path,
                collection_name=settings.collection_name,
                embedder=self.embedder,
            )
        )
        self.retriever = HybridRetriever(
            store=self.store,
            embedder=self.embedder,
            dense_k=settings.dense_k,
            lexical_k=settings.lexical_k,
            final_top_k=settings.final_top_k,
            max_chunks_per_article=settings.max_chunks_per_article,
            dense_weight=settings.dense_weight,
            lexical_weight=settings.lexical_weight,
            min_score=settings.min_retrieval_score,
        )
        self.generator = self._build_generator(settings)
        self.audit = AuditLogger(settings.audit_db_path)
        self.workflow = (
            RagWorkflow(
                settings=settings,
                retriever=self.retriever,
                generator=self.generator,
                store=self.store,
                audit=self.audit,
            )
            if self.generator is not None
            else None
        )

    @staticmethod
    def _build_generator(settings: Settings):
        if settings.llm_provider == "azure_openai":
            if not settings.generation_ready:
                return None
            return AzureOpenAIAnswerGenerator(
                api_key=settings.azure_openai_api_key or "",
                azure_endpoint=settings.azure_openai_endpoint or "",
                api_version=settings.azure_openai_api_version,
                deployment=settings.azure_openai_deployment or "",
                timeout_seconds=settings.request_timeout_seconds,
            )

        if settings.llm_provider == "openai":
            if not settings.generation_ready:
                return None
            return OpenAIAnswerGenerator(
                api_key=settings.openai_api_key or "",
                model=settings.openai_model,
                timeout_seconds=settings.request_timeout_seconds,
            )

        return ExtractiveAnswerGenerator()

    def index(self, input_path: Path) -> IndexSummary:
        articles = load_articles(input_path)
        chunks = build_chunks(
            articles,
            chunk_size_words=self.settings.chunk_size_words,
            overlap_words=self.settings.chunk_overlap_words,
        )
        summary = self.store.rebuild(chunks)
        self.retriever.refresh()
        return summary

    def ask(
        self,
        *,
        question: str,
        session_id: str | None = None,
        debug: bool = False,
    ) -> AskResponse:
        if not self.workflow:
            raise RuntimeError(
                "Generation is not configured. Check LLM_PROVIDER and its "
                "required credentials in .env."
            )

        intent = classify_conversation(question).intent
        if intent == "article_question" and not self.retriever.ready:
            raise RuntimeError("Corpus is not indexed. Run the index command first.")

        return self.workflow.invoke(
            question=question,
            session_id=session_id,
            debug=debug,
        )

    def health(self) -> HealthResponse:
        corpus_ready = self.retriever.ready
        generation_ready = self.workflow is not None
        return HealthResponse(
            status="ok" if corpus_ready and generation_ready else "degraded",
            app=self.settings.app_name,
            environment=self.settings.app_env,
            corpus_ready=corpus_ready,
            indexed_chunks=self.store.count(),
            generation_ready=generation_ready,
            llm_provider=self.settings.llm_provider,
            model=self.settings.active_model_name,
        )


@lru_cache(maxsize=1)
def get_service() -> StockNewsService:
    return StockNewsService(get_settings())
