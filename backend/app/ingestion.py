from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import Article, ChunkRecord
from .text_utils import normalize_text


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-") or "article"


def load_articles(input_path: Path) -> list[Article]:
    """Load articles from a JSON file or a directory of JSON/Markdown/text files.

    Preferred JSON shape::

        {"articles": [{"id": "A001", "title": "...", "body": "..."}]}
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Article input path does not exist: {input_path}")

    files = [input_path] if input_path.is_file() else sorted(
        path
        for path in input_path.rglob("*")
        if path.suffix.lower() in {".json", ".md", ".txt"}
        and path.name.lower() not in {"readme.md", "readme.txt"}
    )
    articles: list[Article] = []

    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("articles", []) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON list or {{'articles': [...]}} in {path}")
            articles.extend(Article.model_validate(record) for record in records)
            continue

        raw_text = normalize_text(path.read_text(encoding="utf-8"))
        lines = raw_text.splitlines()
        first_line = lines[0].strip() if lines else path.stem
        title = first_line.removeprefix("Title:").lstrip("# ").strip() or path.stem
        # Do not embed the title as part of the article body. This prevents the
        # extractive fallback and the LLM context from beginning answers with
        # "Title: ...".
        body_lines = lines[1:] if len(lines) > 1 else lines
        body = normalize_text("\n".join(body_lines)) or raw_text
        articles.append(
            Article(
                id=_slugify(path.stem),
                title=title,
                source="Local file",
                body=body,
            )
        )

    if not articles:
        raise ValueError(f"No supported article files found under {input_path}")

    seen: set[str] = set()
    for article in articles:
        if article.id in seen:
            raise ValueError(f"Duplicate article id: {article.id}")
        seen.add(article.id)
    return articles


def chunk_article(article: Article, chunk_size_words: int, overlap_words: int) -> list[ChunkRecord]:
    if overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    clean_body = normalize_text(article.body)
    words = clean_body.split()
    article_hash = hashlib.sha256(clean_body.encode("utf-8")).hexdigest()
    step = chunk_size_words - overlap_words
    chunks: list[ChunkRecord] = []

    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            continue
        text = " ".join(chunk_words)
        chunks.append(
            ChunkRecord(
                id=f"{article.id}:c{index:03d}",
                article_id=article.id,
                article_title=article.title,
                source=article.source,
                url=str(article.url) if article.url else None,
                published_at=article.published_at.isoformat() if article.published_at else None,
                tickers=article.tickers,
                chunk_index=index,
                text=text,
                article_hash=article_hash,
            )
        )
        if start + chunk_size_words >= len(words):
            break
    return chunks


def build_chunks(articles: list[Article], chunk_size_words: int, overlap_words: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for article in articles:
        chunks.extend(chunk_article(article, chunk_size_words, overlap_words))
    return chunks


def corpus_version(chunks: list[ChunkRecord]) -> str:
    payload = "|".join(f"{chunk.id}:{chunk.article_hash}" for chunk in sorted(chunks, key=lambda item: item.id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
