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
          <p>
            Trigger the full LangGraph path: config loading, dataset prep, training, evaluation,
            failure analysis, report generation, and experiment tracking.
          </p>
        </div>
      </div>
      <div className="workflow-steps">
        <span>1. Select config</span>
        <span>2. Train model</span>
        <span>3. Save artifacts</span>
        <span>4. Review report</span>
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
