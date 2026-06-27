import json

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.db.init_db import init_database
from src.db.repository import create_run, update_run


def create_test_run(tmp_path):
    database_path = tmp_path / "experiments.db"
    run_dir = tmp_path / "outputs" / "runs" / "run_001"
    run_dir.mkdir(parents=True)

    (run_dir / "metrics.json").write_text(
        json.dumps({"eval_loss": 0.1, "token_accuracy": 0.9}),
        encoding="utf-8",
    )
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"tokens": ["Tesla"], "predicted_tags": ["B-ORG"]}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "failure_cases.jsonl").write_text("", encoding="utf-8")
    (run_dir / "report.md").write_text("# Report\n\nModel looks good.\n", encoding="utf-8")

    init_database(database_path)
    create_run(
        database_path=database_path,
        run_id="run_001",
        task="ner",
        model_type="lstm_ner",
        status="running",
        config_path="configs/ner_workflow.yaml",
        output_dir=str(run_dir),
    )
    update_run(
        database_path=database_path,
        run_id="run_001",
        status="completed",
        metrics_json={"eval_loss": 0.1, "token_accuracy": 0.9},
        report_path=str(run_dir / "report.md"),
    )
    return database_path


def test_health_endpoint_returns_ok(tmp_path):
    database_path = tmp_path / "experiments.db"
    init_database(database_path)
    client = TestClient(create_app(database_path=database_path))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_lists_runs_and_reads_run_detail(tmp_path):
    database_path = create_test_run(tmp_path)
    client = TestClient(create_app(database_path=database_path))

    list_response = client.get("/api/runs")
    detail_response = client.get("/api/runs/run_001")

    assert list_response.status_code == 200
    assert list_response.json()[0]["run_id"] == "run_001"
    assert detail_response.status_code == 200
    assert detail_response.json()["metrics"]["token_accuracy"] == 0.9


def test_api_reads_metrics_predictions_failures_and_report(tmp_path):
    database_path = create_test_run(tmp_path)
    client = TestClient(create_app(database_path=database_path))

    metrics_response = client.get("/api/runs/run_001/metrics")
    predictions_response = client.get("/api/runs/run_001/predictions")
    failures_response = client.get("/api/runs/run_001/failures")
    report_response = client.get("/api/runs/run_001/report")

    assert metrics_response.json()["token_accuracy"] == 0.9
    assert predictions_response.json()[0]["tokens"] == ["Tesla"]
    assert failures_response.json() == []
    assert "Model looks good" in report_response.text
