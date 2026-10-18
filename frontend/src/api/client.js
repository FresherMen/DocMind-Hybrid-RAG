const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const initEngine = (apiKey, model = "gpt-4o-mini") =>
  request("/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ openai_api_key: apiKey, llm_model: model }),
  });

export const ingestFiles = (files) => {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  return request("/ingest", { method: "POST", body: fd });
};

export const queryRAG = (question, topK) =>
  request("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });

export const compareDocuments = (question) =>
  request("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

export const listDocuments = () => request("/documents");
export const clearDocuments = () => request("/documents", { method: "DELETE" });
export const getEvalMetrics = (question) =>
  request(`/eval/metrics?question=${encodeURIComponent(question)}`);

export const checkHealth = () => request("/health");

