import { apiClient } from "./client";

export type Metrics = Record<string, unknown>;

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
