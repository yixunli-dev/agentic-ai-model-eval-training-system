import type { PredictionRow } from "../api/runs";

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

export default function PredictionTable({ predictions }: { predictions: PredictionRow[] }) {
  if (predictions.length === 0) {
    return <div className="empty-state">No predictions found.</div>;
  }

  const firstPrediction = predictions[0];
  if (typeof firstPrediction.prompt === "string" || typeof firstPrediction.generated_text === "string") {
    return (
      <div className="stack">
        {predictions.map((prediction, index) => (
          <div className="prediction-block" key={index}>
            <div className="label">Prompt</div>
            <p>{String(prediction.prompt || "")}</p>
            <div className="label">Generated text</div>
            <p>{String(prediction.generated_text || "")}</p>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="table-wrap compact">
      <table>
        <thead>
          <tr>
            <th>token</th>
            <th>gold tag</th>
            <th>predicted tag</th>
          </tr>
        </thead>
        <tbody>
          {predictions.flatMap((prediction, predictionIndex) => {
            const tokens = isStringArray(prediction.tokens) ? prediction.tokens : [];
            const goldTags = isStringArray(prediction.gold_tags) ? prediction.gold_tags : [];
            const predictedTags = isStringArray(prediction.predicted_tags) ? prediction.predicted_tags : [];
            return tokens.map((token, tokenIndex) => (
              <tr key={`${predictionIndex}-${tokenIndex}`}>
                <td>{token}</td>
                <td>{goldTags[tokenIndex] || "n/a"}</td>
                <td>{predictedTags[tokenIndex] || "n/a"}</td>
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}
