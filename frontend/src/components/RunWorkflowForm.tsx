type RunWorkflowFormProps = {
  selectedConfig: string;
  isSubmitting: boolean;
  error: string | null;
  onConfigChange: (configPath: string) => void;
  onSubmit: () => void;
};

const configOptions = [
  { label: "LSTM NER", value: "configs/ner_workflow.yaml" },
  { label: "Transformer Decoder", value: "configs/transformer_workflow.yaml" }
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
      <label htmlFor="config-path">Workflow config</label>
      <select id="config-path" value={selectedConfig} onChange={(event) => onConfigChange(event.target.value)}>
        {configOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label} - {option.value}
          </option>
        ))}
      </select>
      <button className="primary-button" disabled={isSubmitting} onClick={onSubmit}>
        {isSubmitting ? "Running workflow..." : "Run Workflow"}
      </button>
      {error ? <p className="error-text">{error}</p> : null}
    </div>
  );
}
