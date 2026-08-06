from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from .audit import AuditLogger
from .config import Settings
from .conversation import classify_conversation
from .guardrails import financial_safety_note, inspect_question
from .llm import AnswerGenerator, GenerationResult
from .models import (
    AskResponse,
    Citation,
    LLMAnswer,
    RetrievalDebugItem,
    RetrievedChunk,
    UsageInfo,
)
from .retrieval import HybridRetriever
from .store import VectorCorpusStore
from .validation import validate_answer


class RagState(TypedDict, total=False):
    request_id: str
    session_id: str | None
    question: str
    debug: bool
    started_at: float
    conversation_intent: str
    direct_response: str | None
    recommendation_intent: bool
    prompt_injection_signal: bool
    retrieved: list[RetrievedChunk]
    generation: GenerationResult
    validated_answer: LLMAnswer
    validation_warnings: list[str]
    response: AskResponse


class RagWorkflow:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        store: VectorCorpusStore,
        audit: AuditLogger,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.store = store
        self.audit = audit
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None

        graph = StateGraph(RagState)
        graph.add_node("classify", self._classify)
        graph.add_node("direct_response", self._direct_response)
        graph.add_node("guard", self._guard)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("abstain", self._abstain)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {"direct_response": "direct_response", "guard": "guard"},
        )
        graph.add_edge("direct_response", "finalize")
        graph.add_edge("guard", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {"abstain": "abstain", "generate": "generate"},
        )
        graph.add_edge("abstain", "finalize")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def invoke(self, *, question: str, session_id: str | None, debug: bool) -> AskResponse:
        initial: RagState = {
            "request_id": str(uuid.uuid4()),
            "session_id": session_id,
            "question": question,
            "debug": debug,
            "started_at": time.perf_counter(),
        }

        if self.graph is not None:
            final_state = self.graph.invoke(initial)
            return final_state["response"]

        # Dependency-light fallback that follows the same deterministic path.
        state: RagState = dict(initial)
        state.update(self._classify(state))

        if self._route_after_classification(state) == "direct_response":
            state.update(self._direct_response(state))
        else:
            state.update(self._guard(state))
            state.update(self._retrieve(state))
            if self._route_after_retrieval(state) == "abstain":
                state.update(self._abstain(state))
            else:
                state.update(self._generate(state))
                state.update(self._validate(state))

        state.update(self._finalize(state))
        return state["response"]

    def _classify(self, state: RagState) -> dict[str, Any]:
        decision = classify_conversation(state["question"])
        return {
            "conversation_intent": decision.intent,
            "direct_response": decision.response,
        }

    @staticmethod
    def _route_after_classification(state: RagState) -> str:
        return "direct_response" if state.get("direct_response") else "guard"

    def _direct_response(self, state: RagState) -> dict[str, Any]:
        answer = LLMAnswer(
            answer=state.get("direct_response") or "How can I help with the indexed stock-news articles?",
            cited_article_ids=[],
            confidence="high",
            insufficient_evidence=False,
            safety_note=None,
        )
        return {
            "recommendation_intent": False,
            "prompt_injection_signal": False,
            "retrieved": [],
            "generation": GenerationResult(answer=answer, input_tokens=0, output_tokens=0),
            "validated_answer": answer,
            "validation_warnings": [],
        }

    def _guard(self, state: RagState) -> dict[str, Any]:
        decision = inspect_question(state["question"], self.settings.max_question_chars)
        return {
            "question": decision.normalized_question,
            "recommendation_intent": decision.recommendation_intent,
            "prompt_injection_signal": decision.prompt_injection_signal,
        }

    def _retrieve(self, state: RagState) -> dict[str, Any]:
        return {"retrieved": self.retriever.retrieve(state["question"])}

    @staticmethod
    def _route_after_retrieval(state: RagState) -> str:
        return "generate" if state.get("retrieved") else "abstain"

    def _abstain(self, state: RagState) -> dict[str, Any]:
        safety_note = financial_safety_note(state.get("recommendation_intent", False))
        answer = LLMAnswer(
            answer="The supplied articles do not provide enough information to answer this question.",
            cited_article_ids=[],
            confidence="low",
            insufficient_evidence=True,
            safety_note=safety_note,
        )
        return {
            "generation": GenerationResult(answer=answer, input_tokens=0, output_tokens=0),
            "validated_answer": answer,
            "validation_warnings": [],
        }

    def _generate(self, state: RagState) -> dict[str, Any]:
        result = self.generator.generate(
            question=state["question"],
            retrieved=state.get("retrieved", []),
            recommendation_intent=state.get("recommendation_intent", False),
            max_context_chars=self.settings.max_context_chars,
        )
        safety_note = result.answer.safety_note or financial_safety_note(
            state.get("recommendation_intent", False)
        )
        return {
            "generation": GenerationResult(
                answer=result.answer.model_copy(update={"safety_note": safety_note}),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        }

    def _validate(self, state: RagState) -> dict[str, Any]:
        validated, warnings = validate_answer(
            state["generation"].answer,
            state.get("retrieved", []),
        )
        return {"validated_answer": validated, "validation_warnings": warnings}

    def _finalize(self, state: RagState) -> dict[str, Any]:
        answer = state["validated_answer"]
        retrieved = state.get("retrieved", [])
        generation = state["generation"]
        cited = set(answer.cited_article_ids)
        citations = self._build_citations(retrieved, cited)
        latency_ms = int((time.perf_counter() - state["started_at"]) * 1000)
        estimated_cost = (
            generation.input_tokens / 1_000_000 * self.settings.input_cost_per_million
            + generation.output_tokens / 1_000_000 * self.settings.output_cost_per_million
        )

        debug_items = None
        if state.get("debug"):
            debug_items = [
                RetrievalDebugItem(
                    chunk_id=item.chunk.id,
                    article_id=item.chunk.article_id,
                    title=item.chunk.article_title,
                    dense_score=item.dense_score,
                    lexical_score=item.lexical_score,
                    fused_score=item.fused_score,
                    preview=item.chunk.text[:240],
                )
                for item in retrieved
            ]

        response = AskResponse(
            request_id=state["request_id"],
            answer=answer.answer,
            citations=citations,
            confidence=answer.confidence,
            insufficient_evidence=answer.insufficient_evidence,
            safety_note=answer.safety_note,
            recommendation_intent_detected=state.get("recommendation_intent", False),
            latency_ms=latency_ms,
            usage=UsageInfo(
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                estimated_cost_usd=round(estimated_cost, 8),
            ),
            debug_retrieval=debug_items,
        )

        manifest = self.store.manifest()
        self.audit.log(
            {
                "request_id": response.request_id,
                "session_id": state.get("session_id"),
                "question": state["question"],
                "answer": response.answer,
                "conversation_intent": state.get("conversation_intent", "article_question"),
                "citations": [citation.model_dump(mode="json") for citation in citations],
                "retrieved": [
                    {
                        "chunk_id": item.chunk.id,
                        "article_id": item.chunk.article_id,
                        "score": item.fused_score,
                    }
                    for item in retrieved
                ],
                "recommendation_intent": state.get("recommendation_intent", False),
                "prompt_injection_signal": state.get("prompt_injection_signal", False),
                "insufficient_evidence": response.insufficient_evidence,
                "confidence": response.confidence,
                "model": self.settings.active_model_name,
                "prompt_version": self.settings.prompt_version,
                "corpus_version": manifest.get("corpus_version"),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "estimated_cost_usd": response.usage.estimated_cost_usd,
                "latency_ms": response.latency_ms,
                "validation_warnings": state.get("validation_warnings", []),
            }
        )
        return {"response": response}

    @staticmethod
    def _build_citations(
        retrieved: list[RetrievedChunk],
        cited_ids: set[str],
    ) -> list[Citation]:
        best_by_article: dict[str, RetrievedChunk] = {}
        for item in retrieved:
            article_id = item.chunk.article_id
            if article_id not in cited_ids:
                continue
            if (
                article_id not in best_by_article
                or item.fused_score > best_by_article[article_id].fused_score
            ):
                best_by_article[article_id] = item

        citations = [
            Citation(
                article_id=item.chunk.article_id,
                title=item.chunk.article_title,
                source=item.chunk.source,
                url=item.chunk.url,
                published_at=item.chunk.published_at,
                relevance_score=max(0.0, min(1.0, item.fused_score)),
            )
            for item in best_by_article.values()
        ]
        return sorted(citations, key=lambda item: item.relevance_score, reverse=True)
