import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type LossPoint = {
  epoch: number;
  loss: number;
};

export default function LossChart({ points }: { points?: LossPoint[] }) {
  if (!points || points.length === 0) {
    return <div className="empty-state">No loss history available.</div>;
  }

  return (
    <div className="chart-panel">
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={points}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="loss" stroke="#2563eb" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
