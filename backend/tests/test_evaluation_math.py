from app.evaluation import _safe_ratio


def test_safe_ratio() -> None:
    assert _safe_ratio(1, 2) == 0.5
    assert _safe_ratio(0, 0, empty_value=1.0) == 1.0
