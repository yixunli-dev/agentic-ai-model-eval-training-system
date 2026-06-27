import { useEffect, useMemo, useState } from "react";
import {
  fetchFailures,
  fetchMetrics,
  fetchPredictions,
  fetchReport,
  fetchRun,
  type FailureCase,
  type Metrics,
  type PredictionRow,
  type RunSummary
} from "../api/runs";
import FailureCaseList from "../components/FailureCaseList";
import LossChart from "../components/LossChart";
import MetricCard from "../components/MetricCard";
import PredictionTable from "../components/PredictionTable";
import ReportViewer from "../components/ReportViewer";

type RunDetailPageProps = {
  runId: string;
};

export default function RunDetailPage({ runId }: RunDetailPageProps) {
  const [run, setRun] = useState<RunSummary | null>(null);
  const [metrics, setMetrics] = useState<Metrics>({});
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [failures, setFailures] = useState<FailureCase[]>([]);
  const [report, setReport] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchRun(runId),
      fetchMetrics(runId),
      fetchPredictions(runId),
      fetchFailures(runId),
      fetchReport(runId)
    ])
      .then(([runData, metricData, predictionData, failureData, reportText]) => {
        setRun(runData);
        setMetrics(metricData);
        setPredictions(predictionData);
        setFailures(failureData);
        setReport(reportText);
      })
      .catch((requestError) => setError(String(requestError)))
      .finally(() => setIsLoading(false));
  }, [runId]);

  const lossPoints = useMemo(() => {
    const history = metrics.loss_history;
    if (!Array.isArray(history)) {
      return [];
    }
    return history
      .map((point) => {
        if (
          typeof point === "object" &&
          point !== null &&
          "epoch" in point &&
          "loss" in point &&
          typeof point.epoch === "number" &&
          typeof point.loss === "number"
        ) {
          return { epoch: point.epoch, loss: point.loss };
        }
        return null;
      })
      .filter((point): point is { epoch: number; loss: number } => Boolean(point));
  }, [metrics]);

  if (isLoading) {
    return <div className="empty-state">Loading run detail...</div>;
  }

  if (error || !run) {
    return <div className="error-box">{error || "Run not found."}</div>;
  }

  return (
    <section className="page-section">
      <div className="page-header">
        <div>
          <h2>{run.run_id}</h2>
          <p>
            {run.task} / {run.model_type} / {run.status}
          </p>
        </div>
      </div>
      <div className="metric-grid">
        <MetricCard label="Eval loss" value={metrics.eval_loss as number | string | undefined} />
        <MetricCard label="Token accuracy" value={metrics.token_accuracy as number | string | undefined} />
        <MetricCard label="Perplexity" value={metrics.perplexity as number | string | undefined} />
        <MetricCard label="Failures" value={failures.length} />
      </div>
      <div className="detail-grid">
        <section className="panel">
          <h3>Run Summary</h3>
          <dl className="summary-list">
            <dt>Config</dt>
            <dd>{run.config_path}</dd>
            <dt>Output directory</dt>
            <dd>{run.output_dir}</dd>
            <dt>Created</dt>
            <dd>{run.created_at}</dd>
          </dl>
        </section>
        <section className="panel">
          <h3>Training Loss</h3>
          <LossChart points={lossPoints} />
        </section>
        <section className="panel wide">
          <h3>Predictions</h3>
          <PredictionTable predictions={predictions} />
        </section>
        <section className="panel">
          <h3>Failure Cases</h3>
          <FailureCaseList failures={failures} />
        </section>
        <section className="panel wide">
          <h3>Report</h3>
          <ReportViewer report={report} />
        </section>
      </div>
    </section>
  );
}
