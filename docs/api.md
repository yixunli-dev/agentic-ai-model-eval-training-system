# API

## Start Server

```bash
uvicorn src.api.main:app --reload
```

## Health

```bash
curl http://127.0.0.1:8000/api/health
```

Response:

```json
{ "status": "ok" }
```

## List Runs

```bash
curl http://127.0.0.1:8000/api/runs
```

Returns newest runs first.

## Get Run Detail

```bash
curl http://127.0.0.1:8000/api/runs/<run_id>
```

Returns metadata, status, config path, output directory, report path, and metrics.

## Get Metrics

```bash
curl http://127.0.0.1:8000/api/runs/<run_id>/metrics
```

Reads:

```text
outputs/runs/<run_id>/metrics.json
```

## Get Predictions

```bash
curl http://127.0.0.1:8000/api/runs/<run_id>/predictions
```

Reads JSONL predictions and returns a JSON array.

## Get Failure Cases

```bash
curl http://127.0.0.1:8000/api/runs/<run_id>/failures
```

Returns an empty list if the failure file does not exist or has no rows.

## Get Report

```bash
curl http://127.0.0.1:8000/api/runs/<run_id>/report
```

Returns Markdown text.

## Trigger Workflow

```bash
curl -X POST http://127.0.0.1:8000/api/workflows/run \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/ner_workflow.yaml"}'
```

The workflow runs synchronously in P3/P4/P5. Background jobs can be added later.
