import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.db.database import get_database_path
from src.db.init_db import init_database


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def connect(database_path=None):
    path = init_database(get_database_path(database_path))
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def create_run(database_path, run_id, task, model_type, status, config_path, output_dir):
    now = utc_now()
    with connect(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, task, model_type, status, config_path, output_dir,
                metrics_json, report_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                (SELECT metrics_json FROM runs WHERE run_id = ?), NULL
            ), COALESCE(
                (SELECT report_path FROM runs WHERE run_id = ?), NULL
            ), COALESCE(
                (SELECT created_at FROM runs WHERE run_id = ?), ?
            ), ?)
            """,
            (
                run_id,
                task,
                model_type,
                status,
                config_path,
                output_dir,
                run_id,
                run_id,
                run_id,
                now,
                now,
            ),
        )
        connection.commit()


def update_run(database_path, run_id, status=None, metrics_json=None, report_path=None):
    existing = get_run_by_id(database_path, run_id)
    if existing is None:
        raise KeyError(f"Run not found: {run_id}")

    next_status = status if status is not None else existing["status"]
    if metrics_json is None:
        next_metrics_json = existing["metrics_json"]
    elif isinstance(metrics_json, str):
        next_metrics_json = metrics_json
    else:
        next_metrics_json = json.dumps(metrics_json)
    next_report_path = report_path if report_path is not None else existing["report_path"]

    with connect(database_path) as connection:
        connection.execute(
            """
            UPDATE runs
            SET status = ?, metrics_json = ?, report_path = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (next_status, next_metrics_json, next_report_path, utc_now(), run_id),
        )
        connection.commit()


def list_runs(database_path=None):
    with connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def get_run_by_id(database_path, run_id):
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return row_to_dict(row)


def create_artifact(database_path, run_id, artifact_type, file_path):
    with connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO artifacts (run_id, artifact_type, file_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, artifact_type, file_path, utc_now()),
        )
        connection.commit()


def record_workflow_started(state, database_path=None):
    config = state["config"]
    create_run(
        database_path=database_path,
        run_id=state["run_id"],
        task=config.get("task", "ner"),
        model_type=config.get("model_type", "lstm_ner"),
        status="running",
        config_path=state["config_path"],
        output_dir=state["run_dir"],
    )


def record_workflow_finished(state, status="completed", database_path=None):
    run_dir = Path(state["run_dir"])
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.md"

    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    update_run(
        database_path=database_path,
        run_id=state["run_id"],
        status=status,
        metrics_json=metrics,
        report_path=str(report_path) if report_path.exists() else None,
    )

    for artifact_type, file_name in [
        ("model", "model.pt"),
        ("metrics", "metrics.json"),
        ("predictions", "predictions.jsonl"),
        ("failures", "failure_cases.jsonl"),
        ("report", "report.md"),
    ]:
        artifact_path = run_dir / file_name
        if artifact_path.exists():
            create_artifact(database_path, state["run_id"], artifact_type, str(artifact_path))
