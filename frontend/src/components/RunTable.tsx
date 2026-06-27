import { getPrimaryMetric, type RunSummary } from "../api/runs";
import { navigateTo } from "../navigation";

export default function RunTable({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <div className="empty-state">No runs found. Start a workflow to populate the dashboard.</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>run_id</th>
            <th>task</th>
            <th>model_type</th>
            <th>status</th>
            <th>created_at</th>
            <th>key metric</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} onClick={() => navigateTo(`/runs/${encodeURIComponent(run.run_id)}`)}>
              <td>
                <div className="run-id-cell">
                  <span className="mono">{run.run_id}</span>
                  <span>{run.output_dir}</span>
                </div>
              </td>
              <td>
                <span className="task-chip">{run.task === "language_modeling" ? "Language modeling" : "NER"}</span>
              </td>
              <td>{run.model_type}</td>
              <td>
                <span className={`status-pill ${run.status}`}>{run.status}</span>
              </td>
              <td>{run.created_at}</td>
              <td>
                <div className="key-metric">
                  <strong>{getPrimaryMetric(run).value}</strong>
                  <span>{getPrimaryMetric(run).label}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
