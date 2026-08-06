from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9$%._-]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def tokenize(text: str) -> list[str]:
    normalized = text.replace("-", " ").replace("/", " ")
    cleaned = [token.lower().strip("._-") for token in TOKEN_RE.findall(normalized)]
    return [token for token in cleaned if token]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]


def stable_join(parts: Iterable[str], separator: str = "\n") -> str:
    return separator.join(part for part in parts if part)
