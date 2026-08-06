from __future__ import annotations

import re
from dataclasses import dataclass


RECOMMENDATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bshould\s+i\s+(buy|sell|hold|short)\b",
        r"\b(buy|sell|hold|short)\s+(this|the)?\s*stock\b",
        r"\bprice\s+target\b",
        r"\bwhat\s+should\s+i\s+invest\b",
        r"\bportfolio\s+allocation\b",
        r"\bguaranteed\s+(return|profit)\b",
    ]
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"developer\s+message",
        r"act\s+as\s+an?\s+unrestricted",
    ]
]


@dataclass(frozen=True)
class GuardrailDecision:
    recommendation_intent: bool
    prompt_injection_signal: bool
    normalized_question: str


def inspect_question(question: str, max_chars: int) -> GuardrailDecision:
    normalized = " ".join(question.split())[:max_chars]
    return GuardrailDecision(
        recommendation_intent=any(pattern.search(normalized) for pattern in RECOMMENDATION_PATTERNS),
        prompt_injection_signal=any(pattern.search(normalized) for pattern in PROMPT_INJECTION_PATTERNS),
        normalized_question=normalized,
    )


def financial_safety_note(recommendation_intent: bool) -> str | None:
    if not recommendation_intent:
        return None
    return (
        "This system can summarize the supplied news, but it does not provide personalized investment advice, "
        "a buy/sell instruction, or a price target."
    )
