from src.workflow.nodes import (
    analyze_failures_node,
    generate_report_node,
    load_config_node,
    mine_failures_node,
)
from src.workflow.state import create_initial_state


def test_workflow_state_records_completed_nodes(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 7\nrun_name: test\noutput_dir: outputs/runs\n",
        encoding="utf-8",
    )
    state = create_initial_state(str(config_path))

    state = load_config_node(state)

    assert state["config"]["seed"] == 7
    assert state["completed_nodes"] == ["load_config"]


def test_failure_analysis_and_report_nodes_update_state(tmp_path):
    state = create_initial_state("config.yaml")
    state["run_dir"] = str(tmp_path)
    state["failure_cases"] = []
    state["metrics"] = {"loss": 0.5, "accuracy": 1.0}
    state["config"] = {"epochs": 1}

    state = mine_failures_node(state)
    state = analyze_failures_node(state)
    state = generate_report_node(state)

    assert "mine_failures" in state["completed_nodes"]
    assert "analyze_failures" in state["completed_nodes"]
    assert "generate_report" in state["completed_nodes"]
    assert "No tagging failures" in state["failure_analysis"]
