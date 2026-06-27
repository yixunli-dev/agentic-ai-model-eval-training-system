import { useEffect, useState } from "react";
import { fetchRuns, type RunSummary } from "../api/runs";
import MetricCard from "../components/MetricCard";
import RunTable from "../components/RunTable";

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchRuns()
      .then(setRuns)
      .catch((requestError) => setError(String(requestError)))
      .finally(() => setIsLoading(false));
  }, []);

  const completedCount = runs.filter((run) => run.status === "completed").length;
  const nerCount = runs.filter((run) => run.task === "ner").length;
  const transformerCount = runs.filter((run) => run.task === "language_modeling").length;

  return (
    <section className="page-section">
      <div className="page-header">
        <div>
          <h2>Experiment Runs</h2>
          <p>Inspect training runs, model metrics, generated artifacts, and reports.</p>
        </div>
      </div>
      <div className="metric-grid">
        <MetricCard label="Total runs" value={runs.length} />
        <MetricCard label="Completed" value={completedCount} />
        <MetricCard label="NER runs" value={nerCount} />
        <MetricCard label="Transformer runs" value={transformerCount} />
      </div>
      {error ? <div className="error-box">{error}</div> : null}
      {isLoading ? <div className="empty-state">Loading runs...</div> : <RunTable runs={runs} />}
    </section>
  );
}
