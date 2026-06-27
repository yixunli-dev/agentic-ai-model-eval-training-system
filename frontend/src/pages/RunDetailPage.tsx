import { useEffect, useMemo, useState } from "react";
import {
  fetchFailures,
  fetchMetrics,
  fetchPredictions,
  fetchReport,
  fetchRun,
  formatMetricValue,
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
import { navigateTo } from "../navigation";

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
  const [activeTab, setActiveTab] = useState("predictions");

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

  const isTransformerRun = run.task === "language_modeling";

  return (
    <section className="page-section">
      <div className="page-header">
        <div>
          <button className="text-button" onClick={() => navigateTo("/")}>
            Back to dashboard
          </button>
          <h2>{run.run_id}</h2>
          <p>
            {run.task} / {run.model_type} / {run.status}
          </p>
        </div>
        <button className="secondary-button" onClick={() => navigateTo("/new")}>
          Run another workflow
        </button>
      </div>
      <div className="metric-grid">
        <MetricCard label="Eval loss" value={formatMetricValue(metrics.eval_loss)} />
        <MetricCard label="Token accuracy" value={formatMetricValue(metrics.token_accuracy)} />
        <MetricCard label="Perplexity" value={formatMetricValue(metrics.perplexity)} />
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
          <h3>{isTransformerRun ? "Language Modeling Story" : "NER Evaluation Story"}</h3>
          <p className="insight-text">
            {isTransformerRun
              ? "This run trains a manual Transformer decoder, evaluates next-token prediction loss, and stores a generated sample for qualitative review."
              : "This run trains an LSTM sequence tagger, compares predicted tags with gold tags, and mines token-level failure cases."}
          </p>
        </section>
        <section className="panel wide">
          <div className="tab-bar">
            <button className={activeTab === "predictions" ? "active" : ""} onClick={() => setActiveTab("predictions")}>
              Predictions
            </button>
            <button className={activeTab === "failures" ? "active" : ""} onClick={() => setActiveTab("failures")}>
              Failure cases
            </button>
            <button className={activeTab === "loss" ? "active" : ""} onClick={() => setActiveTab("loss")}>
              Loss chart
            </button>
            <button className={activeTab === "report" ? "active" : ""} onClick={() => setActiveTab("report")}>
              Report
            </button>
          </div>
          {activeTab === "predictions" ? <PredictionTable predictions={predictions} /> : null}
          {activeTab === "failures" ? <FailureCaseList failures={failures} /> : null}
          {activeTab === "loss" ? <LossChart points={lossPoints} /> : null}
          {activeTab === "report" ? <ReportViewer report={report} /> : null}
        </section>
      </div>
    </section>
  );
}
