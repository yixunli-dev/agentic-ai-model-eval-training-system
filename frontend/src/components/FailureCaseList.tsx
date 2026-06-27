import type { FailureCase } from "../api/runs";

export default function FailureCaseList({ failures }: { failures: FailureCase[] }) {
  if (failures.length === 0) {
    return <div className="empty-state">No failure cases found.</div>;
  }

  return (
    <div className="stack">
      {failures.map((failure, index) => (
        <pre className="json-panel" key={index}>
          {JSON.stringify(failure, null, 2)}
        </pre>
      ))}
    </div>
  );
}
