import { apiClient } from "./client";

export type Metrics = Record<string, unknown>;

export type LossPoint = {
  epoch: number;
  loss: number;
  token_accuracy?: number;
  perplexity?: number;
};

export type RunSummary = {
  run_id: string;
  task: string;
  model_type: string;
  status: string;
  config_path: string;
  output_dir: string;
  metrics?: Metrics | null;
  report_path?: string | null;
  created_at: string;
  updated_at: string;
};

export type PredictionRow = Record<string, unknown>;
export type FailureCase = Record<string, unknown>;

export function formatMetricValue(value: unknown) {
  if (typeof value === "number") {
    if (value >= 100) {
      return value.toFixed(1);
    }
    if (value >= 10) {
      return value.toFixed(2);
    }
    return value.toFixed(4);
  }
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  return String(value);
}

export function getPrimaryMetric(run: RunSummary) {
  const metrics = run.metrics || {};
  if (typeof metrics.token_accuracy === "number") {
    return { label: "Token accuracy", value: formatMetricValue(metrics.token_accuracy) };
  }
  if (typeof metrics.perplexity === "number") {
    return { label: "Perplexity", value: formatMetricValue(metrics.perplexity) };
  }
  if (typeof metrics.eval_loss === "number") {
    return { label: "Eval loss", value: formatMetricValue(metrics.eval_loss) };
  }
  return { label: "Metric", value: "n/a" };
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const response = await apiClient.get<RunSummary[]>("/api/runs");
  return response.data;
}

export async function fetchRun(runId: string): Promise<RunSummary> {
  const response = await apiClient.get<RunSummary>(`/api/runs/${runId}`);
  return response.data;
}

export async function fetchMetrics(runId: string): Promise<Metrics> {
  const response = await apiClient.get<Metrics>(`/api/runs/${runId}/metrics`);
  return response.data;
}

export async function fetchPredictions(runId: string): Promise<PredictionRow[]> {
  const response = await apiClient.get<PredictionRow[]>(`/api/runs/${runId}/predictions`);
  return response.data;
}

export async function fetchFailures(runId: string): Promise<FailureCase[]> {
  const response = await apiClient.get<FailureCase[]>(`/api/runs/${runId}/failures`);
  return response.data;
}

export async function fetchReport(runId: string): Promise<string> {
  const response = await apiClient.get<string>(`/api/runs/${runId}/report`, {
    responseType: "text"
  });
  return response.data;
}

export async function runWorkflow(configPath: string): Promise<{ run_id: string; status: string }> {
  const response = await apiClient.post<{ run_id: string; status: string }>("/api/workflows/run", {
    config_path: configPath
  });
  return response.data;
}
