import { useState } from "react";
import { runWorkflow } from "../api/runs";
import RunWorkflowForm from "../components/RunWorkflowForm";
import { navigateTo } from "../navigation";

export default function NewRunPage() {
  const [selectedConfig, setSelectedConfig] = useState("configs/ner_workflow.yaml");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setIsSubmitting(true);
    setError(null);
    try {
      const result = await runWorkflow(selectedConfig);
      navigateTo(`/runs/${encodeURIComponent(result.run_id)}`);
    } catch (requestError) {
      setError(String(requestError));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="page-section narrow">
      <div className="page-header">
        <div>
          <h2>Run Workflow</h2>
          <p>Trigger an LSTM NER or Transformer decoder workflow through the FastAPI backend.</p>
        </div>
      </div>
      <RunWorkflowForm
        selectedConfig={selectedConfig}
        isSubmitting={isSubmitting}
        error={error}
        onConfigChange={setSelectedConfig}
        onSubmit={handleSubmit}
      />
    </section>
  );
}
