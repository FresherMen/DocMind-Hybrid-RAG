import { useState, useRef, useEffect, useCallback } from "react";
import { queryRAG, compareDocuments, ingestFiles, listDocuments, initEngine, checkHealth } from "./api/client";
import ChatMessage from "./components/ChatMessage";
import SourcePanel from "./components/SourcePanel";
import EvalMetrics from "./components/EvalMetrics";
import DocumentManager from "./components/DocumentManager";
import "./App.css";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeSource, setActiveSource] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [mode, setMode] = useState("chat"); // chat | compare
  const [apiKey, setApiKey] = useState("");
  const [engineReady, setEngineReady] = useState(false);
  const [sidebarTab, setSidebarTab] = useState("docs"); // docs | eval
  const [lastEval, setLastEval] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const initStatus = async () => {
      try {
        const health = await checkHealth();
        if (health.engine_ready) {
          setEngineReady(true);
          const docs = await listDocuments();
          setDocuments(docs.documents || []);
        }
      } catch (e) {
        console.error("Failed to check health:", e);
      }
    };
    initStatus();
  }, []);


  const handleInit = async () => {
    if (!apiKey.trim()) return;
    try {
      await initEngine(apiKey);
      setEngineReady(true);
      const docs = await listDocuments();
      setDocuments(docs.documents || []);
    } catch (e) {
      alert("Failed to initialize: " + e.message);
    }
  };

  const handleUpload = async (files) => {
    if (!engineReady) return;
    try {
      const result = await ingestFiles(files);
      const docs = await listDocuments();
      setDocuments(docs.documents || []);
      return result;
    } catch (e) {
      throw e;
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading || !engineReady) return;
    const question = input.trim();
    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", content: question }]);

    try {
      let response;
      if (mode === "compare") {
        response = await compareDocuments(question);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", type: "compare", content: response },
        ]);
      } else {
        response = await queryRAG(question);
        setLastEval(response.eval_metrics);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", type: "rag", content: response },
        ]);
      }
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", type: "error", content: e.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app">
      {/* ── Sidebar ─────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="logo">⬡ DocMind</span>
          <span className="version">v2.0</span>
        </div>

        {/* API Key */}
        {!engineReady ? (
          <div className="init-section">
            <label className="field-label">OpenAI API Key</label>
            <input
              type="password"
              className="api-input"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleInit()}
            />
            <button className="btn-primary" onClick={handleInit}>
              Initialize Engine
            </button>
          </div>
        ) : (
          <div className="engine-status-container">
            <div className="engine-status">
              <span className="status-dot" /> Engine ready
            </div>
            <button className="btn-change-key" onClick={() => setEngineReady(false)}>
              Change / Reset Key
            </button>
          </div>
        )}

        {/* Sidebar Tabs */}
        <div className="sidebar-tabs">
          <button
            className={`tab ${sidebarTab === "docs" ? "active" : ""}`}
            onClick={() => setSidebarTab("docs")}
          >
            Documents
          </button>
          <button
            className={`tab ${sidebarTab === "eval" ? "active" : ""}`}
            onClick={() => setSidebarTab("eval")}
          >
            Eval Metrics
          </button>
        </div>

        {sidebarTab === "docs" ? (
          <DocumentManager
            documents={documents}
            onUpload={handleUpload}
            engineReady={engineReady}
          />
        ) : (
          <EvalMetrics metrics={lastEval} />
        )}

        {/* Mode Toggle */}
        <div className="mode-section">
          <label className="field-label">Query Mode</label>
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === "chat" ? "active" : ""}`}
              onClick={() => setMode("chat")}
            >
              Chat
            </button>
            <button
              className={`mode-btn ${mode === "compare" ? "active" : ""}`}
              onClick={() => setMode("compare")}
            >
              Compare
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main Chat ───────────────────────────── */}
      <main className="chat-main">
        <div className="chat-header">
          <div>
            <h1 className="chat-title">RAG Document Intelligence</h1>
            <p className="chat-subtitle">
              Hybrid retrieval · RRF fusion · Query expansion · RAGAS eval
            </p>
          </div>
          {documents.length > 0 && (
            <div className="doc-pills">
              {documents.map((d) => (
                <span key={d.doc_name} className="doc-pill">
                  📄 {d.doc_name}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">⬡</div>
              <p>Upload documents and ask questions.</p>
              <p className="empty-sub">
                Dense + sparse retrieval · Source citations · Confidence scoring
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              message={msg}
              onSourceClick={(src) => setActiveSource(src)}
            />
          ))}

          {loading && (
            <div className="typing-indicator">
              <span />
              <span />
              <span />
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="input-area">
          {mode === "compare" && (
            <div className="mode-badge">🔄 Compare mode — answers across all docs</div>
          )}
          <div className="input-row">
            <textarea
              className="chat-input"
              placeholder={
                !engineReady
                  ? "Initialize engine first..."
                  : !documents.length
                  ? "Upload documents first..."
                  : mode === "compare"
                  ? "Ask a question to compare across all documents..."
                  : "Ask a question about your documents..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!engineReady || !documents.length || loading}
              rows={2}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!input.trim() || loading || !engineReady || !documents.length}
            >
              ↑
            </button>
          </div>
        </div>
      </main>

      {/* ── Source Panel ─────────────────────────── */}
      {activeSource && (
        <SourcePanel source={activeSource} onClose={() => setActiveSource(null)} />
      )}
    </div>
  );
}
