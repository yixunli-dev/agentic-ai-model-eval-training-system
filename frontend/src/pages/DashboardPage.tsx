import { useEffect, useMemo, useState } from "react";
import { fetchRuns, type RunSummary } from "../api/runs";
import MetricCard from "../components/MetricCard";
import RunTable from "../components/RunTable";
import { navigateTo } from "../navigation";

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchText, setSearchText] = useState("");
  const [taskFilter, setTaskFilter] = useState("all");

  function loadRuns() {
    setIsLoading(true);
    setError(null);
    fetchRuns()
      .then(setRuns)
      .catch((requestError) => setError(String(requestError)))
      .finally(() => setIsLoading(false));
  }

  useEffect(() => {
    loadRuns();
  }, []);

  const completedCount = runs.filter((run) => run.status === "completed").length;
  const nerCount = runs.filter((run) => run.task === "ner").length;
  const transformerCount = runs.filter((run) => run.task === "language_modeling").length;
  const latestRun = runs[0];

  const filteredRuns = useMemo(() => {
    return runs.filter((run) => {
      const matchesTask = taskFilter === "all" || run.task === taskFilter;
      const searchableText = `${run.run_id} ${run.task} ${run.model_type} ${run.status}`.toLowerCase();
      const matchesSearch = searchableText.includes(searchText.toLowerCase());
      return matchesTask && matchesSearch;
    });
  }, [runs, searchText, taskFilter]);

  return (
    <section className="page-section">
      <div className="dashboard-hero">
        <div>
          <p className="section-kicker">Agentic ML evaluation system</p>
          <h2>Track model runs from training to failure analysis.</h2>
          <p>
            Compare LSTM NER and Transformer decoder workflows, inspect artifacts, and open
            generated reports from one dashboard.
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-button" onClick={() => navigateTo("/new")}>
            Run workflow
          </button>
          <button className="secondary-button" onClick={loadRuns}>
            Refresh
          </button>
        </div>
      </div>
      <div className="metric-grid">
        <MetricCard label="Total runs" value={runs.length} />
        <MetricCard label="Completed" value={completedCount} />
        <MetricCard label="NER runs" value={nerCount} />
        <MetricCard label="Transformer runs" value={transformerCount} />
      </div>
      {latestRun ? (
        <div className="spotlight-panel">
          <div>
            <span className="label">Latest run</span>
            <h3>{latestRun.run_id}</h3>
            <p>
              {latestRun.task} / {latestRun.model_type} / {latestRun.status}
            </p>
          </div>
          <button onClick={() => navigateTo(`/runs/${encodeURIComponent(latestRun.run_id)}`)}>
            View details
          </button>
        </div>
      ) : null}
      <div className="toolbar">
        <input
          value={searchText}
          onChange={(event) => setSearchText(event.target.value)}
          placeholder="Search runs, tasks, models..."
        />
        <select value={taskFilter} onChange={(event) => setTaskFilter(event.target.value)}>
          <option value="all">All tasks</option>
          <option value="ner">NER</option>
          <option value="language_modeling">Language modeling</option>
        </select>
      </div>
      {error ? <div className="error-box">{error}</div> : null}
      {isLoading ? <div className="empty-state">Loading runs...</div> : <RunTable runs={filteredRuns} />}
    </section>
  );
}
