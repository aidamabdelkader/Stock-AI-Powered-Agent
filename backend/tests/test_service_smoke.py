import json
from pathlib import Path

from app.config import Settings
from app.service import StockNewsService


def make_service(tmp_path: Path) -> StockNewsService:
    return StockNewsService(
        Settings(
            llm_provider="extractive",
            vector_backend="memory",
            embedding_backend="hashing",
            corpus_manifest_path=tmp_path / "manifest.json",
            audit_db_path=tmp_path / "audit.db",
            chroma_path=tmp_path / "chroma",
            min_retrieval_score=0.1,
        )
    )


def test_end_to_end_smoke_and_abstention(tmp_path: Path) -> None:
    corpus_path = tmp_path / "articles.json"
    corpus_path.write_text(
        json.dumps(
            {
                "articles": [
                    {
                        "id": "A001",
                        "title": "Example Systems raises outlook",
                        "source": "Test Wire",
                        "tickers": ["EXM"],
                        "body": (
                            "Example Systems reported revenue of $2 billion. "
                            "The company raised full-year growth guidance to 12% because enterprise demand improved."
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    service = make_service(tmp_path)
    summary = service.index(corpus_path)
    assert summary.articles_indexed == 1

    answer = service.ask(question="What full-year growth guidance did Example Systems provide?")
    assert answer.insufficient_evidence is False
    assert [citation.article_id for citation in answer.citations] == ["A001"]
    assert "[A001]" in answer.answer

    abstention = service.ask(question="What was the chief executive's compensation?")
    assert abstention.insufficient_evidence is True
    assert abstention.citations == []
