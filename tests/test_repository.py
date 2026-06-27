import json

from src.db.init_db import init_database
from src.db.repository import (
    create_run,
    get_run_by_id,
    list_runs,
    update_run,
)


def test_repository_creates_and_updates_run_metadata(tmp_path):
    database_path = tmp_path / "experiments.db"
    init_database(database_path)

    create_run(
        database_path=database_path,
        run_id="run_001",
        task="ner",
        model_type="lstm_ner",
        status="running",
        config_path="configs/ner_workflow.yaml",
        output_dir="outputs/runs/run_001",
    )
    update_run(
        database_path=database_path,
        run_id="run_001",
        status="completed",
        metrics_json={"token_accuracy": 1.0},
        report_path="outputs/runs/run_001/report.md",
    )

    run = get_run_by_id(database_path, "run_001")
    runs = list_runs(database_path)

    assert run["run_id"] == "run_001"
    assert run["status"] == "completed"
    assert json.loads(run["metrics_json"]) == {"token_accuracy": 1.0}
    assert runs[0]["run_id"] == "run_001"
