export default function SourcePanel({ source, onClose }) {
  if (!source) return null;
  return (
    <div className="source-panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">📄 {source.source}</div>
          <div className="panel-meta">Page {source.page} · Chunk {source.chunk_id}</div>
        </div>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      <div className="panel-scores">
        <div className="panel-score">
          <span>Dense similarity</span>
          <strong>{Math.round(source.dense_score * 100)}%</strong>
        </div>
        <div className="panel-score">
          <span>BM25 lexical</span>
          <strong>{Math.round(source.bm25_score * 100)}%</strong>
        </div>
        <div className="panel-score">
          <span>RRF final</span>
          <strong>{(source.final_score * 1000).toFixed(1)}</strong>
        </div>
      </div>

      <div className="panel-content">{source.content}</div>
    </div>
  );
}
