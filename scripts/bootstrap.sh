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
  echo "Created .env."
  echo "Configure Azure OpenAI settings before running the application:"
  echo "  LLM_PROVIDER=azure_openai"
  echo "  AZURE_OPENAI_ENDPOINT=..."
  echo "  AZURE_OPENAI_API_KEY=..."
  echo "  AZURE_OPENAI_DEPLOYMENT=..."
  echo "  AZURE_OPENAI_API_VERSION=2024-10-21"
fi

echo "Setup complete."