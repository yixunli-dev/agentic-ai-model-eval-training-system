from src.workflow.nodes import (
    analyze_failures_node,
    evaluate_model_node,
    generate_report_node,
    load_config_node,
    mine_failures_node,
    prepare_dataset_node,
    train_model_node,
)
from src.db.database import DEFAULT_DATABASE_PATH
from src.db.repository import record_workflow_finished
from src.workflow.state import create_initial_state


NODE_ORDER = [
    ("load_config", load_config_node),
    ("prepare_dataset", prepare_dataset_node),
    ("train_model", train_model_node),
    ("evaluate_model", evaluate_model_node),
    ("mine_failures", mine_failures_node),
    ("analyze_failures", analyze_failures_node),
    ("generate_report", generate_report_node),
]


def build_workflow_graph():
    try:
        from langgraph.graph import END, StateGraph

        from src.workflow.state import WorkflowState

        graph = StateGraph(WorkflowState)
        for node_name, node_function in NODE_ORDER:
            graph.add_node(node_name, node_function)

        graph.set_entry_point("load_config")
        for index in range(len(NODE_ORDER) - 1):
            graph.add_edge(NODE_ORDER[index][0], NODE_ORDER[index + 1][0])
        graph.add_edge(NODE_ORDER[-1][0], END)
        return graph.compile()
    except Exception:
        return LocalWorkflowGraph()


class LocalWorkflowGraph:
    def invoke(self, state):
        for _, node_function in NODE_ORDER:
            state = node_function(state)
        return state


def run_workflow(config_path, database_path=DEFAULT_DATABASE_PATH):
    graph = build_workflow_graph()
    initial_state = create_initial_state(config_path, database_path=database_path)
    try:
        return graph.invoke(initial_state)
    except Exception:
        if initial_state.get("run_id"):
            record_workflow_finished(
                initial_state,
                status="failed",
                database_path=database_path,
            )
        raise
