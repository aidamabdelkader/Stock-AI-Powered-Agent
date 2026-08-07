.PHONY: setup setup-smoke index api ui test eval demo docker clean

setup:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e "./backend[rag,dev]"
	cd frontend && npm install


setup-smoke:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e "./backend[dev]"

index:
	cd backend && ../.venv/bin/python -m app.cli index --input ../data/articles

api:
	cd backend && ../.venv/bin/uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

ui:
	cd frontend && npm run dev

test:
	cd backend && ../.venv/bin/pytest --cov=app --cov-report=term-missing

eval:
	cd backend && ../.venv/bin/python -m app.cli evaluate --dataset ../eval/questions.json --output ../eval/results

demo:
	bash scripts/run_demo.sh

docker:
	docker compose up --build

clean:
	rm -rf .venv data/chroma data/audit.db frontend/node_modules frontend/dist
