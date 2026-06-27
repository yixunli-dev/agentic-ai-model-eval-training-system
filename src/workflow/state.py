from typing import Any, Dict, List, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    config_path: str
    database_path: str
    config: Dict[str, Any]
    run_id: str
    run_dir: str
    dataset: Dict[str, Any]
    model: Any
    batch: Dict[str, Any]
    training_history: List[Dict[str, float]]
    metrics: Dict[str, float]
    predictions: List[Dict[str, Any]]
    failure_cases: List[Dict[str, Any]]
    failure_analysis: str
    report_path: str
    completed_nodes: List[str]
    error: Optional[str]


def create_initial_state(config_path, database_path=None):
    return {
        "config_path": config_path,
        "database_path": database_path,
        "completed_nodes": [],
    }


def mark_node_complete(state, node_name):
    completed_nodes = list(state.get("completed_nodes", []))
    completed_nodes.append(node_name)
    state["completed_nodes"] = completed_nodes
    return state
