import json
from datetime import datetime
from pathlib import Path

import torch

from src.data.language_modeling_dataset import (
    build_language_modeling_dataset,
    read_text_corpus,
)
from src.data.ner_preprocess import build_ner_dataset, read_ner_data
from src.evaluation.evaluator import evaluate_language_model, evaluate_model
from src.evaluation.failure_miner import mine_failure_cases
from src.evaluation.report_generator import generate_markdown_report
from src.llm.failure_analyzer import FailureAnalyzer
from src.db.repository import record_workflow_finished, record_workflow_started
from src.training.trainer import train_lstm_ner_model, train_transformer_decoder_model
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.workflow.state import mark_node_complete


def load_config_node(state):
    config = load_config(state["config_path"])
    set_seed(config.get("seed", 42))
    run_name = config.get("run_name", "ner_workflow")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"
    run_dir = Path(config.get("output_dir", "outputs/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    state["config"] = config
    state["run_id"] = run_id
    state["run_dir"] = str(run_dir)
    if state.get("database_path"):
        record_workflow_started(state, database_path=state.get("database_path"))
    print(f"[load_config] Loaded config from {state['config_path']}")
    return mark_node_complete(state, "load_config")


def prepare_dataset_node(state):
    config = state["config"]
    if config.get("task", "ner") == "language_modeling":
        corpus_lines = read_text_corpus(config["dataset_path"])
        dataset = build_language_modeling_dataset(
            corpus_lines,
            sequence_length=config.get("sequence_length", 8),
        )
        print(f"[prepare_dataset] Loaded {len(corpus_lines)} language modeling lines")
    else:
        sentences, tags = read_ner_data(config["dataset_path"])
        dataset = build_ner_dataset(sentences, tags)
        print(f"[prepare_dataset] Loaded {len(sentences)} NER sentences")
    state["dataset"] = dataset
    return mark_node_complete(state, "prepare_dataset")


def train_model_node(state):
    config = state["config"]
    if config.get("task", "ner") == "language_modeling":
        model, history, batch = train_transformer_decoder_model(state["dataset"], config)
    else:
        model, history, batch = train_lstm_ner_model(state["dataset"], config)

    model_path = Path(state["run_dir"]) / "model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "token_to_id": state["dataset"]["token_to_id"],
        "id_to_token": state["dataset"]["id_to_token"],
        "config": config,
    }
    if config.get("task", "ner") == "language_modeling":
        checkpoint["pad_token_id"] = state["dataset"]["pad_token_id"]
        checkpoint["unk_token_id"] = state["dataset"]["unk_token_id"]
    else:
        checkpoint["tag_to_id"] = state["dataset"]["tag_to_id"]
        checkpoint["id_to_tag"] = state["dataset"]["id_to_tag"]
    torch.save(checkpoint, model_path)

    state["model"] = model
    state["batch"] = batch
    state["training_history"] = history
    print(f"[train_model] Saved model checkpoint to {model_path}")
    return mark_node_complete(state, "train_model")


def evaluate_model_node(state):
    predictions_path = Path(state["run_dir"]) / "predictions.jsonl"
    if state["config"].get("task", "ner") == "language_modeling":
        metrics, predictions = evaluate_language_model(
            state["model"],
            state["batch"],
            state["dataset"],
            predictions_path,
            state["config"],
        )
    else:
        metrics, predictions = evaluate_model(
            state["model"],
            state["batch"],
            state["dataset"],
            predictions_path,
        )
    metrics["loss_history"] = state.get("training_history", [])
    metrics_path = Path(state["run_dir"]) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    state["metrics"] = metrics
    state["predictions"] = predictions
    print(f"[evaluate_model] Saved metrics to {metrics_path}")
    return mark_node_complete(state, "evaluate_model")


def mine_failures_node(state):
    output_path = Path(state["run_dir"]) / "failure_cases.jsonl"
    if state["config"].get("task", "ner") == "language_modeling":
        failure_cases = []
        output_path.write_text("", encoding="utf-8")
    else:
        failure_cases = mine_failure_cases(state.get("predictions", []), output_path)
    state["failure_cases"] = failure_cases
    print(f"[mine_failures] Saved {len(failure_cases)} failure cases to {output_path}")
    return mark_node_complete(state, "mine_failures")


def analyze_failures_node(state):
    if state["config"].get("task", "ner") == "language_modeling":
        state["failure_analysis"] = (
            "Language modeling runs do not use token-level NER failure cases. "
            "Review loss, perplexity, and generated text samples instead."
        )
        print("[analyze_failures] Generated language modeling analysis")
        return mark_node_complete(state, "analyze_failures")

    analyzer = FailureAnalyzer()
    state["failure_analysis"] = analyzer.analyze(state.get("failure_cases", []))
    print("[analyze_failures] Generated failure analysis")
    return mark_node_complete(state, "analyze_failures")


def generate_report_node(state):
    report_path = Path(state["run_dir"]) / "report.md"
    state["report_path"] = generate_markdown_report(state, report_path)
    if state.get("run_id"):
        record_workflow_finished(
            state,
            status="completed",
            database_path=state.get("database_path"),
        )
    print(f"[generate_report] Saved report to {report_path}")
    return mark_node_complete(state, "generate_report")
