type MetricCardProps = {
  label: string;
  value: string | number | null | undefined;
};

export default function MetricCard({ label, value }: MetricCardProps) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value === null || value === undefined ? "n/a" : value}</strong>
    </div>
  );
}
