from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

## This module define Pydantic models for the stock-new RAG system,
# #  including Article, ChunkRecord, RetrievedChunk, Citation, LLMAnswer, AskRequest, RetrievalDebugItem, UsageInfo, AskResponse, HealthResponse, and IndexSummary. These models enforce data validation and structure for articles, chunks, citations, LLM answers, user requests and responses, health checks, and index summaries.
class Article(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    title: str = Field(min_length=1, max_length=500)
    source: str = Field(default="Unknown", max_length=200)
    url: HttpUrl | None = None
    published_at: datetime | None = None
    tickers: list[str] = Field(default_factory=list)
    body: str = Field(min_length=20)

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, values: list[str]) -> list[str]:
        return sorted({value.strip().upper() for value in values if value.strip()})

## Define Pydantic model for the chunk of an article 
class ChunkRecord(BaseModel):
    id: str ## here includes the unique id of the document 
    article_id: str
    article_title: str
    source: str
    url: str | None = None
    published_at: str | None = None
    tickers: list[str] = Field(default_factory=list)
    chunk_index: int
    text: str
    article_hash: str


class RetrievedChunk(BaseModel):
    chunk: ChunkRecord
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fused_score: float = 0.0


class Citation(BaseModel):
    article_id: str
    title: str
    source: str
    url: str | None = None
    published_at: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class LLMAnswer(BaseModel):
    answer: str = Field(min_length=1)
    cited_article_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    insufficient_evidence: bool = False
    safety_note: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1200)
    session_id: str | None = Field(default=None, max_length=100)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return " ".join(value.split())


class RetrievalDebugItem(BaseModel):
    chunk_id: str
    article_id: str
    title: str
    dense_score: float
    lexical_score: float
    fused_score: float
    preview: str


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    insufficient_evidence: bool
    safety_note: str | None = None
    recommendation_intent_detected: bool = False
    latency_ms: int
    usage: UsageInfo
    debug_retrieval: list[RetrievalDebugItem] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    environment: str
    corpus_ready: bool
    indexed_chunks: int
    generation_ready: bool
    llm_provider: str
    model: str


class IndexSummary(BaseModel):
    articles_indexed: int
    chunks_indexed: int
    corpus_version: str
    collection_name: str
