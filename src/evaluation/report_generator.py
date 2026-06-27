from pathlib import Path


def generate_markdown_report(state, output_path):
    config = state.get("config", {})
    metrics = state.get("metrics", {})
    training_history = state.get("training_history", [])
    failure_cases = state.get("failure_cases", [])
    failure_analysis = state.get("failure_analysis", "")
    predictions = state.get("predictions", [])

    final_train_metrics = training_history[-1] if training_history else {}
    task = config.get("task", "ner")
    model_type = config.get("model_type", "lstm_ner")
    sample_prediction = predictions[0] if predictions else {}

    if task == "language_modeling":
        improvement_suggestions = [
            "- Add train/validation splitting before trusting perplexity.",
            "- Increase corpus size to evaluate general next-token prediction quality.",
            "- Inspect generated samples for repetition caused by the tiny dataset.",
        ]
    else:
        improvement_suggestions = [
            "- Add more diverse NER examples before trusting generalization.",
            "- Track per-tag precision and recall in a later phase.",
            "- Add human review for repeated entity-boundary errors.",
        ]

    report = [
        "# Agentic AI Model Evaluation Report",
        "",
        "## Config Summary",
        f"- Task: `{task}`",
        f"- Model type: `{model_type}`",
        f"- Dataset: `{config.get('dataset_path', 'unknown')}`",
        f"- Epochs: `{config.get('epochs', 'unknown')}`",
        f"- Learning rate: `{config.get('learning_rate', 'unknown')}`",
        f"- Embedding dim: `{config.get('embedding_dim', 'unknown')}`",
        f"- Hidden dim: `{config.get('hidden_dim', 'unknown')}`",
        "",
        "## Training Metrics",
        f"- Final loss: `{final_train_metrics.get('loss', 'n/a')}`",
        f"- Final token accuracy: `{final_train_metrics.get('token_accuracy', 'n/a')}`",
        f"- Final perplexity: `{final_train_metrics.get('perplexity', 'n/a')}`",
        "",
        "## Evaluation Metrics",
        f"- Eval loss: `{metrics.get('eval_loss', 'n/a')}`",
        f"- Token accuracy: `{metrics.get('token_accuracy', 'n/a')}`",
        f"- Perplexity: `{metrics.get('perplexity', 'n/a')}`",
        "",
        "## Sample Generation",
        f"- Prompt: `{sample_prediction.get('prompt', config.get('sample_prompt', 'n/a'))}`",
        f"- Output: `{sample_prediction.get('generated_text', 'n/a')}`",
        "",
        "## Causal Masking Notes",
        "The Transformer decoder uses a causal mask so each token can only attend to itself and earlier tokens during next-token prediction.",
        "",
        "## Failure Cases",
        f"- Count: `{len(failure_cases)}`",
        "",
        "## Failure Analysis",
        failure_analysis,
        "",
        "## Improvement Suggestions",
    ]
    report.extend(improvement_suggestions)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return str(path)
