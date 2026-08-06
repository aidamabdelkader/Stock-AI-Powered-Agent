from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ConversationIntent = Literal[
    "greeting",
    "capability",
    "identity",
    "thanks",
    "article_question",
]


@dataclass(frozen=True)
class ConversationDecision:
    intent: ConversationIntent
    response: str | None = None


_CAPABILITY_PATTERNS = (
    r"\bwhat can you do(?: for me)?(?: today)?\b",
    r"\bhow can you help(?: me)?\b",
    r"\bwhat do you do\b",
    r"\bwhat can i ask(?: you)?\b",
    r"\bshow me your capabilities\b",
    r"\bwhat are your capabilities\b",
    r"^\s*help\s*[!.?]*\s*$",
)

_IDENTITY_PATTERNS = (
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bhow do you work\b",
)

_GREETING_PATTERNS = (
    r"^\s*hi\s*[!.?]*\s*$",
    r"^\s*hello\s*[!.?]*\s*$",
    r"^\s*hey\s*[!.?]*\s*$",
    r"^\s*good morning\s*[!.?]*\s*$",
    r"^\s*good afternoon\s*[!.?]*\s*$",
    r"^\s*good evening\s*[!.?]*\s*$",
)

_THANKS_PATTERNS = (
    r"^\s*thanks?\s*[!.?]*\s*$",
    r"^\s*thank you\s*[!.?]*\s*$",
)


def classify_conversation(question: str) -> ConversationDecision:
    normalized = " ".join(question.strip().lower().split())

    if any(re.search(pattern, normalized) for pattern in _CAPABILITY_PATTERNS):
        return ConversationDecision("capability", capability_response())

    if any(re.search(pattern, normalized) for pattern in _IDENTITY_PATTERNS):
        return ConversationDecision("identity", identity_response())

    if any(re.search(pattern, normalized) for pattern in _GREETING_PATTERNS):
        return ConversationDecision("greeting", greeting_response())

    if any(re.search(pattern, normalized) for pattern in _THANKS_PATTERNS):
        return ConversationDecision("thanks", thanks_response())

    return ConversationDecision("article_question")


def capability_response() -> str:
    return (
        "I am your financial news assistant and can answer questions about the indexed stock-market news articles. "
        "You can ask me to summarize an article, explain earnings or market "
        "movements, compare companies or sectors, extract financial figures, "
        "and explain analyst views. I use only the indexed articles, cite the "
        "sources supporting my answer, and say clearly when the evidence is "
        "insufficient. I do not provide personalized buy, sell, hold, or "
        "price-target recommendations."
    )


def greeting_response() -> str:
    return (
        "Hello! Ask me about the indexed stock-market news articles. For "
        "example: 'Why did the EGX30 rise?', 'Compare the two banking-sector "
        "outlooks', or 'Which companies announced dividends?'"
    )


def identity_response() -> str:
    return (
        "I am a closed-book stock-news research assistant. I retrieve relevant "
        "evidence from the indexed article collection, generate a concise answer "
        "from that evidence only, validate its citations, and log the request "
        "for auditability."
    )


def thanks_response() -> str:
    return "You're welcome. Ask me another question about the indexed articles."
