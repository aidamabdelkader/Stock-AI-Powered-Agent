from app.guardrails import inspect_question


def test_recommendation_intent_is_detected() -> None:
    result = inspect_question("Should I buy this stock?", max_chars=1200)
    assert result.recommendation_intent is True


def test_normal_question_is_not_recommendation() -> None:
    result = inspect_question("What revenue guidance did the company provide?", max_chars=1200)
    assert result.recommendation_intent is False
