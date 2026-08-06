#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Run ./scripts/bootstrap.sh first."
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

source .venv/bin/activate
cd backend
python -m app.cli index --input ../data/articles
exec uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
