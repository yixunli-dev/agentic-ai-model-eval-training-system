import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.api.schemas import WorkflowRunRequest, WorkflowRunResponse
from src.db.database import DEFAULT_DATABASE_PATH
from src.db.init_db import init_database
from src.db.repository import get_run_by_id, list_runs
from src.workflow.graph import run_workflow


def parse_metrics(metrics_json):
    if not metrics_json:
        return None
    return json.loads(metrics_json)


def format_run(row):
    return {
        "run_id": row["run_id"],
        "task": row["task"],
        "model_type": row["model_type"],
        "status": row["status"],
        "config_path": row["config_path"],
        "output_dir": row["output_dir"],
        "metrics": parse_metrics(row["metrics_json"]),
        "report_path": row["report_path"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def read_json_file(path):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def read_jsonl_file(path):
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


def get_run_or_404(database_path, run_id):
    run = get_run_by_id(database_path, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


def create_app(database_path=DEFAULT_DATABASE_PATH):
    init_database(database_path)
    app = FastAPI(title="Agentic AI Model Eval Training System API")
    app.state.database_path = database_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/runs")
    def get_runs():
        return [format_run(row) for row in list_runs(app.state.database_path)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return format_run(get_run_or_404(app.state.database_path, run_id))

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(run_id: str):
        run = get_run_or_404(app.state.database_path, run_id)
        return read_json_file(Path(run["output_dir"]) / "metrics.json")

    @app.get("/api/runs/{run_id}/predictions")
    def get_predictions(run_id: str):
        run = get_run_or_404(app.state.database_path, run_id)
        return read_jsonl_file(Path(run["output_dir"]) / "predictions.jsonl")

    @app.get("/api/runs/{run_id}/failures")
    def get_failures(run_id: str):
        run = get_run_or_404(app.state.database_path, run_id)
        return read_jsonl_file(Path(run["output_dir"]) / "failure_cases.jsonl")

    @app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
    def get_report(run_id: str):
        run = get_run_or_404(app.state.database_path, run_id)
        report_path = Path(run["output_dir"]) / "report.md"
        if not report_path.exists():
            raise HTTPException(status_code=404, detail=f"Report not found: {run_id}")
        return report_path.read_text(encoding="utf-8")

    @app.post("/api/workflows/run", response_model=WorkflowRunResponse)
    def post_run_workflow(request: WorkflowRunRequest):
        final_state = run_workflow(
            request.config_path,
            database_path=app.state.database_path,
        )
        return {"run_id": final_state["run_id"], "status": "completed"}

    return app


app = create_app()
