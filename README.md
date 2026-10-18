# DocMind v2 — Production RAG System

> A senior-level Retrieval-Augmented Generation system featuring **hybrid search**, **Reciprocal Rank Fusion**, **query expansion**, **contextual compression**, and **RAGAS-style evaluation** — deployed as a FastAPI + React full-stack application with Docker.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=flat-square)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=flat-square)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│   Chat UI · Source Panel · RAGAS Metrics · Compare Mode        │
└────────────────────────┬────────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────────┐
│                     FastAPI Backend                             │
│   /ingest  /query  /compare  /eval/metrics  /documents         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                       RAG Engine                                │
│                                                                 │
│  1. Query Expansion (multi-query generation via LLM)           │
│       ↓                                                         │
│  2. Dense Retrieval   3. Sparse Retrieval (BM25)               │
│     (ChromaDB +          (lexical keyword                       │
│      cosine sim)          matching)                             │
│       ↓                      ↓                                  │
│  4. Reciprocal Rank Fusion (RRF)                               │
│       ↓                                                         │
│  5. Contextual Compression (LLM strips noise from chunks)      │
│       ↓                                                         │
│  6. Answer Generation (GPT-4o-mini with grounded context)      │
│       ↓                                                         │
│  7. RAGAS-style Evaluation (faithfulness · relevancy · precision)│
└─────────────────────────────────────────────────────────────────┘
         │                              │
   ChromaDB (persistent)          BM25 Index (session)
```

---

## Key Technical Features

### Retrieval
| Technique | Description |
|---|---|
| **Dense retrieval** | ChromaDB + OpenAI `text-embedding-3-small` / MiniLM, cosine similarity |
| **Sparse retrieval** | BM25 lexical matching for exact keyword coverage |
| **RRF fusion** | Reciprocal Rank Fusion combines dense + sparse rankings without score normalization |
| **Query expansion** | LLM generates N query variants to improve recall across semantic space |
| **Contextual compression** | LLM strips irrelevant sentences from each retrieved chunk before generation |

### Generation & Evaluation
| Feature | Description |
|---|---|
| **Grounded generation** | GPT-4o-mini / Llama 3 answers strictly from retrieved context with inline citations |
| **RAGAS-style eval** | LLM-judged: Faithfulness, Answer Relevancy, Context Precision, Confidence |
| **Multi-doc comparison** | Per-document answers + cross-source synthesis in a single request |
| **Structured outputs** | All responses are typed Pydantic models serialized as JSON |

### Infrastructure
| Layer | Tech |
|---|---|
| **Backend** | FastAPI with async endpoints, CORS, lifespan management |
| **Vector store** | ChromaDB with disk persistence (survives restarts) |
| **Frontend** | React 18, component-driven, source panel slide-over |
| **Deployment** | Docker + Docker Compose, multi-stage builds |

---

## Quick Start

### Option A — Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/FresherMen/DocMind-Hybrid-RAG.git
cd DocMind-Hybrid-RAG

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY (or GROQ_API_KEY) in .env

# Run full stack with Docker Compose
docker-compose up --build
```

Frontend: http://localhost:3000 · API: http://localhost:8000 · Docs: http://localhost:8000/docs

### Option B — Local dev

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
OPENAI_API_KEY=sk-... uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/init` | Initialize engine with API key + config |
| `POST` | `/ingest` | Upload and index documents (multipart) |
| `POST` | `/query` | Hybrid RAG query with eval metrics |
| `POST` | `/compare` | Cross-document comparison + synthesis |
| `GET` | `/documents` | List all indexed documents |
| `GET` | `/eval/metrics?question=...` | RAGAS-style metrics for a query |
| `DELETE` | `/documents` | Clear all indexed data |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 6}'
```

```json
{
  "answer": "The key findings include... [Source 1, Page 3]",
  "sources": [
    {
      "source": "report.pdf",
      "page": 3,
      "dense_score": 0.847,
      "bm25_score": 0.721,
      "final_score": 0.0312,
      "content": "..."
    }
  ],
  "confidence": 0.87,
  "expanded_queries": ["What are the main conclusions?", "What did the study find?"],
  "latency_ms": 1842,
  "eval_metrics": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "context_precision": 0.79,
    "confidence": 0.87,
    "reasoning": "Answer is well-supported by multiple high-scoring source chunks."
  }
}
```

---

## Project Structure

```
DocMind-Hybrid-RAG/
├── backend/
│   ├── core/
│   │   └── rag_engine.py       # Hybrid RAG: dense + BM25 + RRF + eval
│   ├── main.py                 # FastAPI app with all endpoints
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main app shell
│   │   ├── App.css             # Dark UI design system
│   │   ├── api/client.js       # API client
│   │   └── components/
│   │       ├── ChatMessage.jsx  # RAG + Compare message renderers
│   │       ├── EvalMetrics.jsx  # Radial RAGAS metric display
│   │       ├── DocumentManager.jsx
│   │       └── SourcePanel.jsx  # Slide-over with RRF scores
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── LICENSE
└── README.md
```

---

## Design Decisions

**Why hybrid search (dense + BM25)?**  
Dense retrieval excels at semantic similarity but can miss exact keyword matches. BM25 captures lexical overlap but does not understand meaning. RRF fusion captures both without requiring score normalization.

**Why query expansion?**  
A single query may not cover the embedding space where relevant chunks live. Generating 2–3 rephrased variants improves recall by 15–30% in typical RAG evaluations.

**Why contextual compression?**  
Retrieved chunks often contain noise — sentences unrelated to the specific question. Compressing each chunk before generation reduces hallucination risk and improves faithfulness scores.

**Why RAGAS-style self-evaluation?**  
Confidence scoring based purely on vector similarity is unreliable. LLM-judged evaluation provides a more semantically meaningful quality signal correlated with actual answer correctness.

---

## Engineering Highlights

> *"Architected a production RAG system with hybrid retrieval (dense embeddings + BM25), Reciprocal Rank Fusion, LLM-powered query expansion, and contextual chunk compression; deployed as a FastAPI + React + Docker full-stack application with RAGAS-style automated evaluation metrics."*

---

## Roadmap

- [ ] Cross-encoder reranking (ColBERT / ms-marco-MiniLM)
- [ ] Streaming responses via Server-Sent Events
- [ ] RAGAS library integration for offline batch eval
- [ ] Multi-tenancy with document namespacing
- [ ] Observability: LangSmith / Langfuse tracing

---

## License

MIT
