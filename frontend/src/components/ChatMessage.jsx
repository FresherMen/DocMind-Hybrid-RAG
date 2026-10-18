import ReactMarkdown from "react-markdown";

function ConfidenceMeter({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 75 ? "#4ade80" : pct >= 45 ? "#fb923c" : "#f87171";
  const label = pct >= 75 ? "High" : pct >= 45 ? "Medium" : "Low";
  return (
    <div className="confidence-meter">
      <div className="conf-header">
        <span className="conf-label">Confidence</span>
        <span className="conf-value" style={{ color }}>{label} · {pct}%</span>
      </div>
      <div className="conf-track">
        <div className="conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function ScoreBadge({ label, value }) {
  const pct = Math.round(value * 100);
  const cls = pct >= 70 ? "badge-high" : pct >= 40 ? "badge-mid" : "badge-low";
  return (
    <span className={`score-badge ${cls}`}>
      {label}: {pct}%
    </span>
  );
}

function SourceCard({ chunk, index, onClick }) {
  const densePct = Math.round(chunk.dense_score * 100);
  const bm25Pct = Math.round(chunk.bm25_score * 100);
  const finalPct = Math.round(chunk.final_score * 1000) / 10;

  return (
    <div className="source-card" onClick={() => onClick(chunk)}>
      <div className="source-header">
        <span className="source-num">#{index + 1}</span>
        <span className="source-name">📄 {chunk.source}</span>
        <span className="source-page">p.{chunk.page}</span>
      </div>
      <div className="source-scores">
        <ScoreBadge label="Dense" value={chunk.dense_score} />
        <ScoreBadge label="BM25" value={chunk.bm25_score} />
        <span className="rrf-score">RRF {finalPct}</span>
      </div>
      <p className="source-preview">{chunk.content.slice(0, 160)}…</p>
    </div>
  );
}

function RAGMessage({ content, onSourceClick }) {
  return (
    <div className="rag-message">
      <div className="answer-body">
        <ReactMarkdown>{content.answer}</ReactMarkdown>
      </div>

      <div className="message-meta">
        <ConfidenceMeter value={content.confidence} />

        {content.reasoning && (
          <p className="reasoning">💭 {content.reasoning}</p>
        )}

        <div className="meta-row">
          <span className="latency">⚡ {content.latency_ms}ms</span>
          {content.expanded_queries?.length > 0 && (
            <span className="expanded-badge">
              🔍 {content.expanded_queries.length + 1} query variants
            </span>
          )}
        </div>

        {content.expanded_queries?.length > 0 && (
          <details className="expanded-queries">
            <summary>Query expansion</summary>
            <ul>
              <li><em>Original:</em> {content.query}</li>
              {content.expanded_queries.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </details>
        )}
      </div>

      {content.sources?.length > 0 && (
        <details className="sources-section" open>
          <summary>
            📚 {content.sources.length} source chunks · click to expand
          </summary>
          <div className="sources-grid">
            {content.sources.map((src, i) => (
              <SourceCard
                key={src.chunk_id || i}
                chunk={src}
                index={i}
                onClick={onSourceClick}
              />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function CompareMessage({ content }) {
  return (
    <div className="compare-message">
      <div className="compare-grid">
        {Object.entries(content.per_document || {}).map(([doc, answer]) => (
          <div key={doc} className="compare-card">
            <div className="compare-doc-name">📄 {doc}</div>
            <div className="compare-answer">
              <ReactMarkdown>{answer}</ReactMarkdown>
            </div>
          </div>
        ))}
      </div>
      {content.synthesis && (
        <div className="synthesis-block">
          <div className="synthesis-label">🧠 Synthesis</div>
          <ReactMarkdown>{content.synthesis}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

export default function ChatMessage({ message, onSourceClick }) {
  if (message.role === "user") {
    return (
      <div className="message user-message">
        <div className="message-bubble user-bubble">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="message assistant-message">
      <div className="assistant-avatar">⬡</div>
      <div className="assistant-content">
        {message.type === "rag" && (
          <RAGMessage content={message.content} onSourceClick={onSourceClick} />
        )}
        {message.type === "compare" && (
          <CompareMessage content={message.content} />
        )}
        {message.type === "error" && (
          <div className="error-message">⚠️ {message.content}</div>
        )}
      </div>
    </div>
  );
}
