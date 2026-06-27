import type { RunSummary } from "../api/runs";
import { navigateTo } from "../navigation";

function getKeyMetric(run: RunSummary) {
  const metrics = run.metrics || {};
  if (typeof metrics.token_accuracy === "number") {
    return `accuracy ${metrics.token_accuracy.toFixed(3)}`;
  }
  if (typeof metrics.perplexity === "number") {
    return `ppl ${metrics.perplexity.toFixed(3)}`;
  }
  if (typeof metrics.eval_loss === "number") {
    return `loss ${metrics.eval_loss.toFixed(3)}`;
  }
  return "n/a";
}

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
              <td className="mono">{run.run_id}</td>
              <td>{run.task}</td>
              <td>{run.model_type}</td>
              <td>
                <span className={`status-pill ${run.status}`}>{run.status}</span>
              </td>
              <td>{run.created_at}</td>
              <td>{getKeyMetric(run)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
