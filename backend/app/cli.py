from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import run_evaluation
from .service import get_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock News RAG command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index articles")
    index_parser.add_argument("--input", type=Path, default=Path("../data/articles"))

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--debug", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run the evaluation dataset")
    eval_parser.add_argument("--dataset", type=Path, default=Path("../eval/questions.json"))
    eval_parser.add_argument("--output", type=Path, default=Path("../eval/results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = get_service()
    if args.command == "index":
        print(service.index(args.input).model_dump_json(indent=2))
    elif args.command == "ask":
        print(service.ask(question=args.question, debug=args.debug).model_dump_json(indent=2))
    elif args.command == "evaluate":
        summary = run_evaluation(service, args.dataset, args.output)
        print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
