# agentic-ai-model-eval-training-system

An AI Engineering portfolio project that combines under-the-hood PyTorch model training with an agentic model evaluation workflow.

The project trains either a small LSTM NER model or a manually implemented Transformer decoder language model, evaluates outputs, mines failure cases where relevant, runs a deterministic failure-analysis wrapper, and generates a Markdown report. The workflow is structured as graph nodes so later phases can add human-in-the-loop review, richer model comparisons, and external LLM analysis.

## Why This Is An AI Engineering Project

This is not a simple LangChain demo. The project includes both sides of an AI engineering workflow:

- **Model training:** custom PyTorch preprocessing, padding, masking-aware loss, token-level metrics, an LSTM sequence tagger, and a manual Transformer decoder for next-token prediction.
- **Workflow orchestration:** a LangGraph-style pipeline that moves through dataset preparation, training, evaluation, failure mining, failure analysis, and report generation.
- **Operational artifacts:** each workflow run writes checkpoints, metrics, predictions, failure cases, and a report under `outputs/runs/<run_id>/`.
- **Experiment tracking API:** FastAPI endpoints list runs, inspect metrics, read reports, and trigger workflows. Run metadata is stored in `outputs/experiments.db`.

The system is intentionally small enough to run locally, but the structure mirrors production model evaluation loops.

## Workflow Architecture

```text
load_config
  -> prepare_dataset
  -> train_model
  -> evaluate_model
  -> mine_failures
  -> analyze_failures
  -> generate_report
```

Each node receives and returns a shared workflow state dictionary. If `langgraph` is installed, `src/workflow/graph.py` builds a `StateGraph`. If it is not installed, the project uses a local sequential fallback runner so the workflow still runs without external services.

## Project Structure

```text
run_workflow.py               Main workflow entry point
train_lstm_ner.py             Training entry point backed by the workflow
evaluate.py                   Reads metrics from a workflow run
predict.py                    Runs NER prediction from a saved checkpoint
src/api/                      FastAPI application and request/response schemas
src/data/                     NER and language-modeling dataset builders
src/db/                       SQLite experiment tracking repository
src/models/                   PyTorch LSTM NER and manual Transformer decoder models
src/training/                 Masked losses, token accuracy, training loops
src/workflow/                 Workflow state, graph, and nodes
src/evaluation/               Evaluation, failure mining, report generation
src/llm/                      Failure analyzer wrapper with local fallback
src/utils/                    Config loading and seed setup
configs/                      YAML workflow configs
data/                         Tiny sample NER and text corpus datasets
tests/                        Pytest coverage for implemented phases
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Workflow

```bash
python3 run_workflow.py --config configs/ner_workflow.yaml
```

Run the Transformer decoder language-modeling workflow:

```bash
python3 run_workflow.py --config configs/transformer_workflow.yaml
```

Expected artifacts:

```text
outputs/runs/<run_id>/model.pt
outputs/runs/<run_id>/metrics.json
outputs/runs/<run_id>/predictions.jsonl
outputs/runs/<run_id>/failure_cases.jsonl
outputs/runs/<run_id>/report.md
```

## Train

```bash
python3 train_lstm_ner.py --config configs/ner_workflow.yaml
```

This runs the same workflow and saves the trained model checkpoint under the run directory.

## Evaluate A Run

```bash
python3 evaluate.py --run-dir outputs/runs/<run_id>
```

## Predict

```bash
python3 predict.py --checkpoint outputs/runs/<run_id>/model.pt --sentence "Tesla builds vehicles in California"
```

Example output:

```text
Tesla    B-ORG
builds   O
vehicles O
in       O
California B-LOC
```

Transformer generation:

```bash
python3 predict.py --task transformer --checkpoint outputs/runs/<run_id>/model.pt --prompt "machine learning"
```

Example output:

```text
machine learning models predict tokens machine learning
```

## Tests

```bash
python3 -m pytest
```

## API Server

Start the local API:

```bash
uvicorn src.api.main:app --reload
```

The API uses SQLite at:

```text
outputs/experiments.db
```

Available endpoints:

```text
GET  /api/health
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/metrics
GET  /api/runs/{run_id}/predictions
GET  /api/runs/{run_id}/failures
GET  /api/runs/{run_id}/report
POST /api/workflows/run
```

Example curl commands:

```bash
curl http://127.0.0.1:8000/api/health

curl http://127.0.0.1:8000/api/runs

curl http://127.0.0.1:8000/api/runs/<run_id>/metrics

curl http://127.0.0.1:8000/api/runs/<run_id>/report

curl -X POST http://127.0.0.1:8000/api/workflows/run \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/transformer_workflow.yaml"}'
```

## Frontend Dashboard

Start the backend first:

```bash
uvicorn src.api.main:app --reload
```

Start the React dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Dashboard workflow checks:

```text
1. Open http://localhost:5173.
2. Confirm recent runs appear in the table.
3. Click a run_id to inspect metrics, predictions, failure cases, and report text.
4. Go to New Run.
5. Select LSTM NER and click Run Workflow.
6. Confirm the app navigates to the new run detail page.
7. Go back to New Run.
8. Select Transformer Decoder and click Run Workflow.
9. Confirm generated text and perplexity appear on the run detail page.
```

## Implemented Phases

P0:

- Full project structure
- README and requirements
- YAML config loading
- Random seed setup
- Sample NER dataset
- Workflow state
- LangGraph-compatible workflow skeleton with local fallback execution
- End-to-end workflow entry point

P1:

- NER preprocessing with token/tag vocabularies
- Padding for token and tag sequences
- LSTM NER model using `nn.Embedding`, `nn.LSTM`, and a linear classifier
- Masking-aware cross entropy loss
- Token-level accuracy ignoring padding
- Training and evaluation inside workflow nodes
- Checkpoint, metrics, predictions, failure cases, and report artifacts
- Deterministic fallback failure analyzer when no API key is available
- Pytest coverage for data loading, padding, model shape, loss, metrics, and workflow state updates

P2:

- Manual Transformer decoder for next-token prediction
- Sinusoidal positional encoding
- Manual scaled dot-product attention and multi-head self-attention
- Causal masking so tokens cannot attend to future tokens
- Language-modeling dataset builder with fixed-length input-target pairs
- Transformer workflow config and workflow integration
- Transformer loss/perplexity evaluation and sample generated text artifacts
- Pytest coverage for attention, masks, decoder blocks, model shape, dataset pairs, and tiny loss decrease

P3:

- SQLite experiment tracking database at `outputs/experiments.db`
- Run metadata table for status, config, output path, metrics, and report path
- Artifact table for generated model, metrics, predictions, failures, and report files
- Workflow metadata recording for running/completed runs
- FastAPI backend for listing runs and reading run artifacts
- Synchronous workflow trigger endpoint for local dashboard integration
- CORS enabled for `http://localhost:5173`

P4:

- React + TypeScript + Vite dashboard under `frontend/`
- Axios API client for FastAPI endpoints
- Dashboard page for recent run summaries
- Run detail page for metrics, predictions, failure cases, and reports
- New run page for launching NER or Transformer workflows
- Recharts loss chart component with an empty-state fallback
- Plain CSS responsive layout for local screenshots and demos

## Planned Phases

- Human-in-the-loop review node for approving or labeling failure cases
- Per-tag precision, recall, and confusion matrix plots
- External LLM failure analysis when explicitly configured
