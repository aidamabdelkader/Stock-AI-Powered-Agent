from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Stock News RAG"
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    llm_provider: str = "azure_openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    request_timeout_seconds: float = 30.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: Path = Path("../data/chroma")
    collection_name: str = "stock_news_articles"
    corpus_manifest_path: Path = Path("../data/chroma/corpus_manifest.json")
    audit_db_path: Path = Path("../data/audit.db")

    vector_backend: str = "chroma"
    embedding_backend: str = "sentence_transformer"

    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    dense_k: int = 8
    lexical_k: int = 8
    final_top_k: int = 5
    max_chunks_per_article: int = 2
    dense_weight: float = 0.6
    lexical_weight: float = 0.40
    min_retrieval_score: float = 0.15

    max_question_chars: int = 1200
    max_context_chars: int = 14000
    prompt_version: str = "stock-news-rag-v2"

    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"openai", "azure_openai", "extractive"}
        if normalized not in allowed:
            raise ValueError(
                "llm_provider must be 'openai', 'azure_openai', or 'extractive'"
            )
        return normalized

    @field_validator("vector_backend")
    @classmethod
    def validate_vector_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"chroma", "memory"}:
            raise ValueError("vector_backend must be 'chroma' or 'memory'")
        return normalized

    @field_validator("embedding_backend")
    @classmethod
    def validate_embedding_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"sentence_transformer", "hashing"}:
            raise ValueError(
                "embedding_backend must be 'sentence_transformer' or 'hashing'"
            )
        return normalized

    @field_validator("dense_weight", "lexical_weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("retrieval weights cannot be negative")
        return value

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError(
                "chunk_overlap_words must be smaller than chunk_size_words"
            )
        if self.dense_weight + self.lexical_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if min(
            self.dense_k,
            self.lexical_k,
            self.final_top_k,
            self.max_chunks_per_article,
        ) <= 0:
            raise ValueError("retrieval counts must be positive")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def generation_ready(self) -> bool:
        if self.llm_provider == "extractive":
            return True
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_model)
        if self.llm_provider == "azure_openai":
            return bool(
                self.azure_openai_api_key
                and self.azure_openai_endpoint
                and self.azure_openai_deployment
                and self.azure_openai_api_version
            )
        return False

    @property
    def active_model_name(self) -> str:
        if self.llm_provider == "azure_openai":
            return self.azure_openai_deployment or "azure-openai-unconfigured"
        if self.llm_provider == "openai":
            return self.openai_model
        return "extractive-smoke-test"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
