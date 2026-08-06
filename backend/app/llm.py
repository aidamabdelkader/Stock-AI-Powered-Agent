from __future__ import annotations

import json
from dataclasses import dataclass

from .models import LLMAnswer, RetrievedChunk
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .text_utils import split_sentences, tokenize

@dataclass(frozen=True)
class GenerationResult:
    answer: LLMAnswer
    input_tokens: int
    output_tokens: int


class AnswerGenerator:
    def generate(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        recommendation_intent: bool,
        max_context_chars: int,
    ) -> GenerationResult:
        raise NotImplementedError


def _parsed_result(completion) -> GenerationResult:
    message = completion.choices[0].message
    refusal = getattr(message, "refusal", None)

    if refusal:
        parsed = LLMAnswer(
            answer=refusal,
            cited_article_ids=[],
            confidence="low",
            insufficient_evidence=True,
        )
    elif getattr(message, "parsed", None):
        parsed = message.parsed
    else:
        raise RuntimeError("Model returned neither a parsed response nor a refusal")

    usage = completion.usage
    return GenerationResult(
        answer=parsed,
        input_tokens=int(usage.prompt_tokens if usage else 0),
        output_tokens=int(usage.completion_tokens if usage else 0),
    )


class OpenAIAnswerGenerator(AnswerGenerator):
    def __init__(self, *, api_key: str, model: str, timeout_seconds: float) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.model = model

    def generate(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        recommendation_intent: bool,
        max_context_chars: int,
    ) -> GenerationResult:
        prompt = build_user_prompt(
            question=question,
            retrieved=retrieved,
            recommendation_intent=recommendation_intent,
            max_context_chars=max_context_chars,
        )
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=LLMAnswer,
        )
        return _parsed_result(completion)

class AzureOpenAIAnswerGenerator(AnswerGenerator):
    """Generate grounded answers through an Azure OpenAI deployment."""

    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment: str,
        timeout_seconds: float,
    ) -> None:
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout_seconds,
        )
        self.deployment = deployment

    def generate(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        recommendation_intent: bool,
        max_context_chars: int,
    ) -> GenerationResult:
        prompt = build_user_prompt(
            question=question,
            retrieved=retrieved,
            recommendation_intent=recommendation_intent,
            max_context_chars=max_context_chars,
        )

        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
        )

        message = completion.choices[0].message
        content = message.content

        if not content:
            raise RuntimeError(
                "Azure OpenAI returned an empty response."
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Azure OpenAI returned invalid JSON: {content}"
            ) from exc

        try:
            parsed = LLMAnswer.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(
                f"Azure OpenAI response failed schema validation: {payload}"
            ) from exc

        usage = completion.usage

        return GenerationResult(
            answer=parsed,
            input_tokens=int(
                usage.prompt_tokens if usage else 0
            ),
            output_tokens=int(
                usage.completion_tokens if usage else 0
            ),
        )

class ExtractiveAnswerGenerator(AnswerGenerator):
    """Offline fallback used only for smoke testing, not final answer quality."""

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "compare",
        "did", "do", "does", "for", "from", "give", "how", "in", "is",
        "it", "its", "of", "on", "or", "the", "to", "was", "were",
        "what", "when", "where", "which", "who", "why", "with",
        "company", "article",
    }

    @staticmethod
    def _stem(token: str) -> str:
        if token.endswith("ies") and len(token) > 5:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    def _focus_terms(self, question: str) -> set[str]:
        return {
            self._stem(token)
            for token in tokenize(question)
            if token not in self._STOPWORDS
        }

    def generate(
        self,
        *,
        question: str,
        retrieved: list[RetrievedChunk],
        recommendation_intent: bool,
        max_context_chars: int,
    ) -> GenerationResult:
        if not retrieved:
            return self._insufficient()

        comparison = any(
            marker in question.lower()
            for marker in ("compare", "versus", " vs ", "both", "across")
        )
        ranked_article_ids = list(
            dict.fromkeys(item.chunk.article_id for item in retrieved)
        )
        allowed_ids = set(ranked_article_ids[:2] if comparison else ranked_article_ids[:1])

        focus = self._focus_terms(question)
        candidates: list[tuple[float, str, str]] = []
        for item in retrieved:
            if item.chunk.article_id not in allowed_ids:
                continue
            for sentence in split_sentences(item.chunk.text):
                sentence = sentence.strip()
                if sentence.lower().startswith("title:"):
                    sentence = sentence.split(":", 1)[-1].strip()
                terms = {self._stem(token) for token in tokenize(sentence)}
                overlap = len(focus & terms) / max(1, len(focus))
                score = 0.65 * item.fused_score + 0.35 * overlap
                candidates.append((score, sentence, item.chunk.article_id))

        candidates.sort(key=lambda row: row[0], reverse=True)
        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _, sentence, article_id in candidates:
            key = sentence.lower()
            if not sentence or key in seen:
                continue
            seen.add(key)
            selected.append((sentence, article_id))
            if len(selected) == 3:
                break

        if not selected:
            return self._insufficient()

        grouped: dict[str, list[str]] = {}
        for sentence, article_id in selected:
            grouped.setdefault(article_id, []).append(sentence)

        paragraphs = [
            f"{' '.join(sentences)} [{article_id}]"
            for article_id, sentences in grouped.items()
        ]
        prefix = (
            "I cannot provide a personalized buy or sell recommendation. "
            if recommendation_intent
            else ""
        )
        answer_text = prefix + "\n\n".join(paragraphs)

        return GenerationResult(
            answer=LLMAnswer(
                answer=answer_text,
                cited_article_ids=list(grouped),
                confidence="medium",
                insufficient_evidence=False,
            ),
            input_tokens=0,
            output_tokens=0,
        )

    @staticmethod
    def _insufficient() -> GenerationResult:
        return GenerationResult(
            answer=LLMAnswer(
                answer=(
                    "The supplied articles do not provide enough information "
                    "to answer this question."
                ),
                cited_article_ids=[],
                confidence="low",
                insufficient_evidence=True,
            ),
            input_tokens=0,
            output_tokens=0,
        )
