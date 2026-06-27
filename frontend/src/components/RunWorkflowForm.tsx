type RunWorkflowFormProps = {
  selectedConfig: string;
  isSubmitting: boolean;
  error: string | null;
  onConfigChange: (configPath: string) => void;
  onSubmit: () => void;
};

const configOptions = [
  {
    label: "LSTM NER",
    value: "configs/ner_workflow.yaml",
    description: "Train a sequence tagger, evaluate token accuracy, mine NER failures.",
    outputs: "model.pt, metrics.json, predictions.jsonl, failure_cases.jsonl, report.md"
  },
  {
    label: "Transformer Decoder",
    value: "configs/transformer_workflow.yaml",
    description: "Train a manual decoder for next-token prediction and inspect generated text.",
    outputs: "model.pt, perplexity metrics, generated text, report.md"
  }
];

export default function RunWorkflowForm({
  selectedConfig,
  isSubmitting,
  error,
  onConfigChange,
  onSubmit
}: RunWorkflowFormProps) {
  return (
    <div className="form-panel">
      <label>Choose a workflow</label>
      <div className="workflow-card-grid">
        {configOptions.map((option) => (
          <button
            className={`workflow-card ${selectedConfig === option.value ? "selected" : ""}`}
            key={option.value}
            onClick={() => onConfigChange(option.value)}
            type="button"
          >
            <strong>{option.label}</strong>
            <span>{option.value}</span>
            <p>{option.description}</p>
            <small>{option.outputs}</small>
          </button>
        ))}
      </div>
      <button className="primary-button" disabled={isSubmitting} onClick={onSubmit}>
        {isSubmitting ? "Running workflow..." : "Run Workflow"}
      </button>
      {isSubmitting ? (
        <div className="progress-note">
          Running synchronously through FastAPI. This can take a few seconds while training and
          report generation finish.
        </div>
      ) : null}
      {error ? <p className="error-text">{error}</p> : null}
    </div>
  );
}
