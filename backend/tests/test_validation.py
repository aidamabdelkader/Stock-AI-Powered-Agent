from app.models import ChunkRecord, LLMAnswer, RetrievedChunk
from app.validation import validate_answer


def retrieved(article_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=ChunkRecord(
            id=f"{article_id}:c000",
            article_id=article_id,
            article_title="Title",
            source="Source",
            chunk_index=0,
            text="Evidence text.",
            article_hash="hash",
        ),
        fused_score=0.8,
    )


def test_valid_inline_citation_passes() -> None:
    answer = LLMAnswer(answer="Revenue increased. [A001]", cited_article_ids=["A001"], confidence="high")
    validated, warnings = validate_answer(answer, [retrieved("A001")])
    assert warnings == []
    assert validated.cited_article_ids == ["A001"]


def test_invalid_citation_fails_closed() -> None:
    answer = LLMAnswer(answer="Revenue increased. [A999]", cited_article_ids=["A999"], confidence="high")
    validated, warnings = validate_answer(answer, [retrieved("A001")])
    assert validated.insufficient_evidence is True
    assert warnings


def test_uncited_second_paragraph_fails_closed() -> None:
    answer = LLMAnswer(
        answer="Revenue increased. [A001]\n\nMargin also increased.",
        cited_article_ids=["A001"],
        confidence="high",
    )
    validated, warnings = validate_answer(answer, [retrieved("A001")])
    assert validated.insufficient_evidence is True
    assert any(warning.startswith("uncited_paragraphs") for warning in warnings)
