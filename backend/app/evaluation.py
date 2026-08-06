from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service import StockNewsService


@dataclass
class EvalRow:
    question_id: str
    question: str
    expected_article_ids: list[str]
    cited_article_ids: list[str]
    citation_recall: float
    citation_precision: float
    keyword_coverage: float
    abstention_correct: bool
    passed: bool
    hallucination_flags: list[str]
    answer: str


def _safe_ratio(numerator: int, denominator: int, empty_value: float = 1.0) -> float:
    return numerator / denominator if denominator else empty_value


def run_evaluation(service: "StockNewsService", dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = payload.get("questions", payload)
    rows: list[EvalRow] = []

    for item in questions:
        response = service.ask(question=item["question"], debug=True)
        expected_ids = set(item.get("expected_article_ids", []))
        cited_ids = {citation.article_id for citation in response.citations}
        expected_keywords = [keyword.lower() for keyword in item.get("expected_keywords", [])]
        answer_lower = response.answer.lower()
        expected_unanswerable = bool(item.get("unanswerable", False))

        recall = _safe_ratio(len(expected_ids & cited_ids), len(expected_ids), empty_value=1.0)
        precision = _safe_ratio(len(expected_ids & cited_ids), len(cited_ids), empty_value=1.0 if expected_unanswerable else 0.0)
        keyword_coverage = _safe_ratio(
            sum(1 for keyword in expected_keywords if keyword in answer_lower),
            len(expected_keywords),
            empty_value=1.0,
        )
        abstention_correct = response.insufficient_evidence == expected_unanswerable
        flags: list[str] = []
        if cited_ids - expected_ids:
            flags.append("unexpected_article_citation")
        if expected_unanswerable and not response.insufficient_evidence:
            flags.append("answered_unanswerable_question")
        if not expected_unanswerable and not cited_ids:
            flags.append("answer_without_citation")
        if response.debug_retrieval:
            retrieved_ids = {row.article_id for row in response.debug_retrieval}
            if cited_ids - retrieved_ids:
                flags.append("citation_not_retrieved")

        passed = (
            abstention_correct
            and not flags
            and (expected_unanswerable or (recall >= 0.5 and keyword_coverage >= 0.5))
        )
        rows.append(
            EvalRow(
                question_id=item["id"],
                question=item["question"],
                expected_article_ids=sorted(expected_ids),
                cited_article_ids=sorted(cited_ids),
                citation_recall=round(recall, 3),
                citation_precision=round(precision, 3),
                keyword_coverage=round(keyword_coverage, 3),
                abstention_correct=abstention_correct,
                passed=passed,
                hallucination_flags=flags,
                answer=response.answer,
            )
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "accuracy": round(_safe_ratio(sum(row.passed for row in rows), len(rows), empty_value=0.0), 3),
        "mean_citation_recall": round(_safe_ratio(sum(row.citation_recall for row in rows), len(rows), 0.0), 3),
        "mean_citation_precision": round(_safe_ratio(sum(row.citation_precision for row in rows), len(rows), 0.0), 3),
        "mean_keyword_coverage": round(_safe_ratio(sum(row.keyword_coverage for row in rows), len(rows), 0.0), 3),
        "hallucination_flag_count": sum(len(row.hallucination_flags) for row in rows),
        "rows": [row.__dict__ for row in rows],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "evaluation_report.md").write_text(_to_markdown(summary), encoding="utf-8")
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Questions: **{summary['question_count']}**",
        f"- Accuracy: **{summary['accuracy']:.1%}**",
        f"- Mean citation recall: **{summary['mean_citation_recall']:.1%}**",
        f"- Mean citation precision: **{summary['mean_citation_precision']:.1%}**",
        f"- Mean keyword coverage: **{summary['mean_keyword_coverage']:.1%}**",
        f"- Hallucination flags: **{summary['hallucination_flag_count']}**",
        "",
        "| ID | Pass | Citation recall | Citation precision | Keyword coverage | Flags |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        flags = ", ".join(row["hallucination_flags"]) or "None"
        lines.append(
            f"| {row['question_id']} | {'Yes' if row['passed'] else 'No'} | "
            f"{row['citation_recall']:.0%} | {row['citation_precision']:.0%} | "
            f"{row['keyword_coverage']:.0%} | {flags} |"
        )
    lines.extend(["", "## Detailed answers", ""])
    for row in summary["rows"]:
        lines.extend(
            [
                f"### {row['question_id']}: {row['question']}",
                "",
                row["answer"],
                "",
                f"Expected articles: `{', '.join(row['expected_article_ids']) or 'none'}`  ",
                f"Cited articles: `{', '.join(row['cited_article_ids']) or 'none'}`  ",
                f"Hallucination flags: `{', '.join(row['hallucination_flags']) or 'none'}`",
                "",
            ]
        )
    return "\n".join(lines)
