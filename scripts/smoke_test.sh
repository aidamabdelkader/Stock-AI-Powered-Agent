#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"

curl -fsS "$API_BASE/health" | python -m json.tool
curl -fsS -X POST "$API_BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Why did Northstar reduce its free-cash-flow forecast?","debug":true}' \
  | python -m json.tool
