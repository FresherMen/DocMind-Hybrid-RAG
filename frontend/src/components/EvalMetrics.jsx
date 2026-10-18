function RadialScore({ label, value }) {
  if (value === undefined || value === null) return null;
  const pct = Math.round(value * 100);
  const color = pct >= 75 ? "#4ade80" : pct >= 45 ? "#fb923c" : "#f87171";
  const r = 28;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;

  return (
    <div className="radial-score">
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r={r} fill="none" stroke="#1e293b" strokeWidth="6" />
        <circle
          cx="36" cy="36" r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round"
          transform="rotate(-90 36 36)"
        />
        <text x="36" y="40" textAnchor="middle" fill={color} fontSize="13" fontWeight="700">
          {pct}%
        </text>
      </svg>
      <span className="radial-label">{label}</span>
    </div>
  );
}

export default function EvalMetrics({ metrics }) {
  if (!metrics) {
    return (
      <div className="eval-empty">
        <p>Run a query to see RAGAS-style evaluation metrics.</p>
        <ul className="eval-legend">
          <li><strong>Faithfulness</strong> — Is the answer grounded in context?</li>
          <li><strong>Relevancy</strong> — Does it address the question?</li>
          <li><strong>Precision</strong> — Are retrieved chunks actually useful?</li>
          <li><strong>Confidence</strong> — Overall answer reliability</li>
        </ul>
      </div>
    );
  }

  return (
    <div className="eval-panel">
      <div className="eval-title">RAGAS-style Metrics</div>
      <div className="radial-grid">
        <RadialScore label="Faithfulness" value={metrics.faithfulness} />
        <RadialScore label="Relevancy" value={metrics.answer_relevancy} />
        <RadialScore label="Precision" value={metrics.context_precision} />
        <RadialScore label="Confidence" value={metrics.confidence} />
      </div>
      {metrics.reasoning && (
        <p className="eval-reasoning">💭 {metrics.reasoning}</p>
      )}
    </div>
  );
}
