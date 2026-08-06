from datetime import datetime

from app.ingestion import chunk_article
from app.models import Article


def test_chunking_preserves_article_metadata() -> None:
    article = Article(
        id="A001",
        title="Example",
        source="Wire",
        published_at=datetime(2026, 1, 1),
        tickers=["abc"],
        body=" ".join(f"word{i}" for i in range(100)),
    )
    chunks = chunk_article(article, chunk_size_words=30, overlap_words=5)
    assert len(chunks) == 4
    assert chunks[0].article_id == "A001"
    assert chunks[0].tickers == ["ABC"]
    assert chunks[1].text.startswith("word25")


def test_directory_loader_ignores_readme(tmp_path) -> None:
    from app.ingestion import load_articles

    (tmp_path / "README.md").write_text("# Instructions\nDo not index this file as an article.", encoding="utf-8")
    (tmp_path / "article.txt").write_text(
        "Example headline\nThis is a sufficiently long article body for ingestion testing.",
        encoding="utf-8",
    )
    articles = load_articles(tmp_path)
    assert [article.id for article in articles] == ["article"]
