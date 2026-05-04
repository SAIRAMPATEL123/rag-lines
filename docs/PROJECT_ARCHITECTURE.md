# RAG-Lines Project Architecture & Feature Map

_Last updated: 2026-05-03_

## 1) Tech Stack
- **Backend API:** FastAPI + Uvicorn
- **RAG Core:** Custom Python pipeline (ingestion, chunking, embeddings, retrieval, reranking, LLM prompt/response)
- **Vector Store:** ChromaDB
- **Embeddings:** sentence-transformers
- **LLM Runtime:** Ollama local API
- **UI (MVP):** Streamlit
- **Scheduling (MVP):** In-process timer-based scheduler
- **Logging:** Loguru + FastAPI request middleware logging

## 2) High-Level Architecture
1. UI sends query/batch/schedule requests to backend API.
2. API routes invoke `RAGPipeline`.
3. Pipeline retrieves context via retriever/vector store.
4. Pipeline calls local LLM and returns response.
5. UI renders answer, context, and metadata.

## 3) Current Project Map
- `main.py` — CLI entrypoint (`ingest`, `qa`, `api`)
- `config/config.py` — central app/runtime configuration
- `src/ingestion/` — loaders + chunkers
- `src/embeddings/` — embedding model wrapper
- `src/vectorstore/` — Chroma integration
- `src/retrieval/` — hybrid retriever + reranker
- `src/llm/` — local LLM client
- `src/qa/` — end-to-end RAG pipeline
- `src/api/` — FastAPI app, routes, scheduler
- `src/ui/` — Streamlit MVP UI
- `tests/` — unit tests

## 4) Features Matrix

### Major Features (Implemented)
- Multi-format ingestion (pdf/md/txt/html/json)
- Chunking (fixed + semantic)
- Hybrid retrieval (semantic + lexical score blend)
- Optional reranking
- Local LLM generation via Ollama
- FastAPI query/customer-support endpoints
- **Batch query API endpoint**
- **Scheduled query API endpoint (in-process, one-off)**
- **MVP web UI connected to backend**
- Runtime request logging and route-level structured logs

### Minor Features (Implemented)
- Collection stats endpoint
- Health endpoint
- Basic evaluation metrics module
- CLI interactive QA mode

### Planned / Next
- Persistent job store for scheduled/batch jobs
- Cron-style recurring jobs
- Auth + role-based access
- Observability dashboards (metrics/traces)
- Production task queue (RQ/Celery)
- Advanced UI: history, citations pane, filters

## 5) API ↔ UI Connectivity
- UI single query tab -> `POST /api/v1/query`
- UI batch tab -> `POST /api/v1/batch-query`
- UI schedule tab -> `POST /api/v1/schedule-query`

## 6) Batch & Scheduling Design (MVP)
- Batch: synchronous processing of multiple questions in one request.
- Scheduling: one-off delayed execution in current process memory.
- Limitation: scheduled jobs are not durable across process restarts.

## 7) Maintenance Rule
Whenever feature code changes:
1. Update this architecture map.
2. Update README quick usage snippets if behavior changed.
3. Add/adjust tests for changed behavior.
