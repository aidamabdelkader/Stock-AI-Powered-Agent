from __future__ import annotations

import re

from .models import LLMAnswer, RetrievedChunk


CITATION_RE = re.compile(r"\[([A-Za-z0-9._-]+)\]")


def validate_answer(
    answer: LLMAnswer,
    retrieved: list[RetrievedChunk],
) -> tuple[LLMAnswer, list[str]]:
    allowed_ids = {
        item.chunk.article_id
        for item in retrieved
    }

    inline_ids = set(
        CITATION_RE.findall(answer.answer)
    )

    declared_ids = set(
        answer.cited_article_ids
    )

    warnings: list[str] = []

    invalid_ids = sorted(
        (inline_ids | declared_ids) - allowed_ids
    )

    if invalid_ids:
        warnings.append(
            "invalid_citations:"
            + ",".join(invalid_ids)
        )

        return (
            LLMAnswer(
                answer=(
                    "The answer could not be returned because its citations "
                    "did not match the retrieved article set. Please retry."
                ),
                cited_article_ids=[],
                confidence="low",
                insufficient_evidence=True,
                safety_note=answer.safety_note,
            ),
            warnings,
        )

    if answer.insufficient_evidence:
        cleaned = answer.model_copy(
            update={
                "cited_article_ids": [],
            }
        )
        return cleaned, warnings

    # If the model declared valid citations but forgot to place them inline,
    # append them to the answer instead of rejecting an otherwise valid answer.
    if not inline_ids and declared_ids:
        valid_declared_ids = sorted(
            declared_ids & allowed_ids
        )

        if valid_declared_ids:
            citation_text = " ".join(
                f"[{article_id}]"
                for article_id in valid_declared_ids
            )

            repaired_answer = (
                f"{answer.answer.rstrip()} {citation_text}"
            )

            warnings.append(
                "inline_citations_repaired"
            )

            inline_ids = set(valid_declared_ids)

            answer = answer.model_copy(
                update={
                    "answer": repaired_answer,
                    "cited_article_ids": valid_declared_ids,
                }
            )

    if not inline_ids:
        warnings.append(
            "missing_inline_citations"
        )

        return (
            LLMAnswer(
                answer=(
                    "The supplied articles may contain relevant information, "
                    "but the generated answer lacked verifiable inline "
                    "citations. Please retry."
                ),
                cited_article_ids=[],
                confidence="low",
                insufficient_evidence=True,
                safety_note=answer.safety_note,
            ),
            warnings,
        )

    paragraphs = [
        part.strip()
        for part in re.split(
            r"\n\s*\n",
            answer.answer,
        )
        if part.strip()
    ]

    uncited_paragraphs = [
        index + 1
        for index, paragraph in enumerate(
            paragraphs
        )
        if not CITATION_RE.search(paragraph)
    ]

    if uncited_paragraphs:
        warnings.append(
            "uncited_paragraphs:"
            + ",".join(
                map(str, uncited_paragraphs)
            )
        )

        return (
            LLMAnswer(
                answer=(
                    "The generated answer contained a factual paragraph "
                    "without a verifiable article citation. Please retry."
                ),
                cited_article_ids=[],
                confidence="low",
                insufficient_evidence=True,
                safety_note=answer.safety_note,
            ),
            warnings,
        )

    cleaned = answer.model_copy(
        update={
            "cited_article_ids": sorted(
                inline_ids
            )
        }
    )

    return cleaned, warnings