.PHONY: install test run-api run-ner run-transformer frontend-install frontend-build frontend-dev

install:
	pip install -r requirements.txt
	cd frontend && npm install

test:
	python3 -m pytest
	cd frontend && npm run build

run-api:
	uvicorn src.api.main:app --reload

run-ner:
	python3 run_workflow.py --config configs/ner_workflow.yaml

run-transformer:
	python3 run_workflow.py --config configs/transformer_workflow.yaml

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build

frontend-dev:
	cd frontend && npm run dev
