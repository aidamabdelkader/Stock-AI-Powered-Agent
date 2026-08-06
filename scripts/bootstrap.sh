#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./backend[rag,dev]"
(
  cd frontend
  npm install
)

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Add OPENAI_API_KEY before using LLM_PROVIDER=openai."
fi

echo "Setup complete."
