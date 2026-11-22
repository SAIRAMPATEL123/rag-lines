# RAG-Lines — Source Code Review & Commercial Upgrade Blueprint

**Review basis:** `rag-lines.zip` uploaded on 22 November 2026  
**Review type:** Static source-code and architecture review  
**Product goal:** Turn the current RAG prototype into a secure, maintainable, sellable document-intelligence platform.

> **Scope and limitation:** I inspected the Python source files, configuration, API routes, scheduler, Streamlit UI, tests, scripts and project documentation in the uploaded archive. All 30 Python files passed syntax compilation. I did not install the project's full dependency set, run the application against a live model/database, execute a complete integration test suite, or perform a penetration test. Findings below are therefore source-code findings, not claims that every runtime failure has been reproduced.

---

# 1. Executive decision

## Recommendation: keep the RAG core, refactor the application, replace the UI

**Do not throw away the project and start from zero. Do not continue adding features on top of the current structure unchanged.**

Retain and improve these concepts/modules:

- Document loading and parsing
- Chunking
- Embedding model wrapper
- Vector-store integration
- Retrieval and reranking
- LLM adapter concept
- RAG orchestration
- FastAPI as the backend API framework

Refactor or replace these parts:

- Configuration and environment management
- Document ingestion API
- Hybrid retrieval implementation
- LLM provider integration
- Scheduler/background processing
- Evaluation implementation
- API security and request lifecycle
- Streamlit UI
- Process manager and ad-hoc startup scripts
- Tests and deployment configuration

The current code is a useful prototype/MVP foundation. It is not yet a production SaaS application and should not be marketed as enterprise-ready until security, isolation, reliability, evaluation and deployment requirements are met.

## What “new APIs” means

Do not confuse APIs with Python files:

- **Python modules** implement the application logic: parsing, chunking, retrieval, model calls, persistence, permissions and orchestration.
- **FastAPI endpoints** expose that logic to the outside world through HTTP/JSON.
- **React frontend** calls those endpoints and renders the product interface.
- **Ollama/vLLM/model server** runs the actual inference model as a separate service.
- **Worker process** performs slow ingestion and other background jobs.

The recommended approach is a modular Python backend with a versioned FastAPI interface, a separate React frontend, and a separately managed model runtime. This is not a rewrite into “APIs only”; it is a better separation of responsibilities.

---

# 2. What is in the uploaded archive

The archive contains the Python RAG backend, Streamlit interface, process manager, configuration, tests, scripts, documentation, local database/vector data, logs, Git metadata and environment files.

Notably:

- There is **no React frontend source code** in this ZIP.
- The only full UI implementation present is `src/ui/streamlit_app.py`.
- The API is FastAPI under `src/api/`.
- `process_manager.py` is a separate Flask control panel for starting/stopping processes.
- `scripts/start_app.sh` starts Ollama, FastAPI and Streamlit together.
- `main.py` provides CLI commands for ingestion, interactive QA and API startup.

The project therefore currently has multiple ways of starting services, but no clean, unified service lifecycle.

## Archive hygiene — do this before giving code to Claude or GitHub

The archive includes a real `.env` file, `.git` history, local database files, ChromaDB data, logs and Python cache files. The example environment file also contains a hard-coded database credential.

**Do not give the original ZIP to Claude or publish it.**

1. Treat any real credentials in the archive as potentially exposed; rotate them if valid or reused.
2. Remove secrets from tracked files and Git history where applicable.
3. Keep `.env` local and ignored.
4. Commit only `.env.example` with placeholders.
5. Exclude `.git`, `__pycache__`, `.pyc`, `data/`, database files, ChromaDB files, logs, model weights, virtual environments and user documents from development handoff archives.
6. Use synthetic test documents rather than customer data.

---

# 3. Highest-priority confirmed code findings

| Priority | File/area | Finding | Required action |
|---|---|---|---|
| P0 | `src/api/routes.py` | Ingestion accepts arbitrary server-side file paths rather than uploaded file content. | Replace public ingestion with multipart upload and storage-backed ingestion. |
| P0 | `src/retrieval/retriever.py` | `_bm25_search` is a simple word-overlap calculation, not BM25. | Implement real BM25 or a maintained lexical-search library. |
| P0 | `src/api/persistent_scheduler.py` | Restart recovery schedules `_mark_job_running()` only; it does not execute the original job. | Replace timer recovery with a durable worker/queue design. |
| P0 | `src/api/app.py`, `routes.py` | No authentication, authorization or tenant isolation. | Implement auth, RBAC and tenant-scoped data access before commercial use. |
| P0 | `src/ui/streamlit_app.py` | Untrusted LLM output and retrieved document text are inserted into HTML with `unsafe_allow_html=True`. | Remove unsafe interpolation; render untrusted content as text/Markdown safely. |
| P0 | `config/config.py`, `local_llm.py`, scripts/docs | LLM URL/model settings conflict; Ollama URL composition can produce a duplicated endpoint. | Define explicit provider adapters and validate endpoint/model settings at startup. |
| P1 | `src/retrieval/retriever.py` | Documented 70/30 weighted fusion is not implemented; code uses `max()`. | Use RRF or a validated normalized score-fusion method. |
| P1 | `src/retrieval/retriever.py` | Final result count uses `self.top_k`, not the request's `top_k`. | Propagate request top-k consistently. |
| P1 | `src/ingestion/chunker.py` | Chunk size is measured in characters, not tokens; semantic chunker does not apply overlap. | Implement token-aware chunking with configurable overlap and hard maximums. |
| P1 | `src/ingestion/document_loader.py` | PDF page provenance is lost; `extract_text()` may return `None`. | Extract per page, preserve page number and handle empty/scanned pages. |
| P1 | `src/vectorstore/chroma_db.py` | Configured vector-store path is ignored in favor of a hard-coded project-relative path. | Respect validated settings and use one shared vector-store instance. |
| P1 | `src/vectorstore/chroma_db.py` | Uses `collection.add`; repeated ingestion with existing IDs can fail. | Use deterministic IDs plus upsert/versioning and explicit replacement semantics. |
| P1 | `src/api/routes.py` | Batch endpoint's list comprehension runs sequentially inside one worker thread. | Use bounded concurrency or a background batch job; do not claim parallelism as implemented. |
| P1 | `src/api/routes.py` | `async` routes directly execute blocking model/database work. | Move blocking work to controlled worker threads or use a suitable async architecture. |
| P1 | `src/evaluation/evaluator.py` | RAGAS is not actually run; word-overlap heuristics are labelled as estimates. | Implement real evaluation with a version-pinned evaluator and a golden dataset. |
| P1 | `src/api/app.py` | CORS excludes DELETE and Authorization headers; configured rate limit is ignored. | Centralize CORS and rate-limit settings; test browser preflight. |
| P1 | `src/llm/local_llm.py` | `is_available()` always calls Ollama `/api/tags`, even for OpenAI-compatible providers. | Provider-specific health checks. |
| P1 | `src/llm/local_llm.py` | `temperature or default` and `max_tokens or default` make zero values impossible. | Use explicit `is None` checks and validate ranges. |
| P1 | `src/api/persistent_scheduler.py` | Completed jobs are not fully loaded into the in-memory list after restart; list results can be incomplete. | Query persistent storage for job listing/status. |
| P1 | `src/api/routes.py` | SQLAlchemy 2.x health check uses raw `session.execute("SELECT 1")`. | Use `text("SELECT 1")` and close sessions with context managers. |
| P2 | `process_manager.py` | Separate Flask process manager kills whichever process owns a fixed port; it has no authentication and suppresses logs. | Remove from production architecture; use Docker Compose/system service management. |
| P2 | `src/ui/streamlit_app.py` | A long polling `while` loop and `sleep()` block the Streamlit interaction. | Replace with non-blocking status polling. |
| P2 | `src/ui/streamlit_app.py` | Upload uses a temporary directory and sends local paths to the API; temp files are not cleaned up. | Use direct multipart upload to the backend and object storage. |
| P2 | Tests | Tests mostly check object construction/chunk presence/embedding shape; no robust API, retrieval, scheduler or tenant tests. | Add deterministic unit, integration, security and regression tests. |

---

# 4. Class-by-class and module-by-module review

## 4.1 `config/config.py` — `Config`

**Current role:** Pydantic settings, environment configuration and directory creation.

**Problems:**

- No explicit environment prefix is configured, while scripts/documentation use `RAG_`-prefixed variables. `RAG_LLM_MODEL` will not automatically map to `llm_model` without a matching prefix/alias.
- `.env.example` uses unprefixed variables, while other documentation describes prefixed variables.
- `db_password` and `db_connection_string` are independent settings. Changing `DB_PASSWORD` does not automatically rebuild the connection string.
- The vector-store path is configured here but ignored in `ChromaVectorStore`.
- The rate-limit setting exists but `RateLimiter` is hard-coded to 100.
- Mutable list default for `supported_formats` should use a safe default factory.
- Directory creation in settings initialization makes configuration loading perform filesystem side effects.
- No strict validation for provider type, URLs, model names, paths or security-sensitive production defaults.

**Required design:**

- Use Pydantic Settings v2 consistently.
- Use a single documented prefix, preferably `RAG_`.
- Separate development/test/production configuration.
- Use `SecretStr` for secrets.
- Construct database URLs safely from individual settings or accept one canonical URL; do not maintain conflicting sources of truth.
- Validate provider configuration at startup.
- Make paths explicit and configurable.
- Do not create directories as a side effect of importing settings.
- Fail fast if production configuration is insecure or incomplete.

## 4.2 `src/ingestion/document_loader.py` — `Document`, `DocumentLoader`

**Current role:** Load PDF, Markdown, TXT, HTML and JSON; includes a web loader.

**Problems:**

- PDF extraction concatenates `page.extract_text()` directly. Some pages return `None`, which can cause a type error.
- Only total page count is retained. Per-page content and page numbers are not preserved for citations.
- Scanned PDFs have no OCR fallback.
- Directory loading logs and skips failed files without returning a structured per-file result.
- Direct file loading does not provide a unified, explicit validation result.
- Web loading accepts a URL without SSRF protections, redirect restrictions, private-network blocking, content-type validation or response-size limits. It is not currently exposed by a documented API, but it is risky if exposed later.
- No DOCX, PPTX, XLSX, CSV, images or HTML upload policy beyond current basic formats.
- No checksum, document ID, version, owner, tenant or ingestion status.

**Required design:**

- Create a parser interface with format-specific adapters.
- PDF extraction must return page-level text blocks and provenance.
- Add optional OCR for scanned PDFs.
- Add DOCX/PPTX/XLSX/CSV support only after the core formats are reliable.
- Validate extension, MIME type, file signature, file size and content.
- Sanitize filenames; store files under generated IDs, never user-controlled paths.
- Calculate SHA-256 checksums for duplicate detection.
- Return structured parser errors and per-file ingestion results.
- If URL ingestion is offered, implement strict SSRF protection and allow-list controls.

## 4.3 `src/ingestion/chunker.py` — `Chunk`, `FixedChunker`, `SemanticChunker`

**Current role:** Fixed and sentence-boundary chunking.

**Problems:**

- Both chunkers use character length, not token length, despite the documentation describing tokens.
- `FixedChunker` uses character offsets and has a possible invalid step if overlap is greater than or equal to chunk size.
- `SemanticChunker` does not implement overlap.
- Sentence-boundary splitting is not semantic embedding-based chunking.
- One long sentence can exceed the configured chunk size.
- NLTK data may be downloaded at runtime; this makes startup/ingestion dependent on network access.
- Chunk IDs include the source path and a counter that resets on each call. Re-ingesting a file can generate IDs that already exist.
- The current metadata does not provide reliable document ID, version, page, section, start/end offsets or tenant ID.

**Required design:**

- Introduce a token-counting abstraction compatible with the selected embedding model/tokenizer.
- Support configurable chunk size, overlap, minimum and maximum.
- Preserve source document ID, source filename, page number, heading/section, chunk index and offsets.
- Use deterministic IDs based on document ID + version + chunk index/content hash.
- Add explicit duplicate/update/delete semantics.
- Use heading-aware and paragraph-aware splitting before falling back to sentence splitting.
- Make NLTK/tokenizer data part of deployment setup, not an unplanned runtime download.
- Build a chunking evaluation set and compare chunking strategies using retrieval results.

## 4.4 `src/embeddings/embedding_model.py` — `EmbeddingModel`

**Current role:** Local SentenceTransformer embedding wrapper.

**Strengths:**

- Small wrapper isolates embedding generation.
- Batch encoding is used.
- Device selection is configurable.

**Problems:**

- No explicit normalization/configuration contract.
- Model name/version/dimension is not recorded with the vector collection.
- No compatibility check before querying an existing collection with a changed model.
- No controlled lifecycle or singleton dependency management.
- `embed_single()` assumes a non-empty input.
- Model loading can happen repeatedly because each new vector-store instance constructs a new embedding model.

**Required design:**

- Load the embedding model once per worker/process.
- Store embedding provider, model ID, revision, dimensions and normalization setting in collection metadata.
- Reject incompatible model/collection combinations.
- Batch and chunk large embedding jobs to control memory.
- Add timeouts, metrics and clear startup readiness checks.
- Keep embedding model separate from generation LLM configuration.

## 4.5 `src/vectorstore/chroma_db.py` — `ChromaVectorStore`

**Current role:** Chroma persistent collection, add and semantic search.

**Problems:**

- Hard-codes a project-relative path instead of using `config.vectorstore_path`.
- Constructs a new `EmbeddingModel` for every new vector-store instance.
- Catches every exception from `get_collection()` and attempts `create_collection()`. This can hide errors other than “collection not found”.
- `add()` fails for duplicate IDs rather than handling updates.
- No document-level delete, version replacement, metadata filter or tenant filter.
- No batch ingestion safeguards for large lists.
- No migration/backup/restore strategy.
- Metadata schema is not validated.
- Search result distances are converted to similarity with `1 - distance`; document the Chroma distance semantics and test the expected range.

**Required design:**

- Inject a shared embedding service and Chroma client.
- Respect the configured persistent path or use a separately deployed Chroma service.
- Use narrow exception handling for collection-not-found.
- Implement upsert, delete-by-document, delete-by-version and metadata filtering.
- Enforce tenant filters in the service layer, not only in the UI.
- Persist collection metadata including embedding model/version/dimension.
- Add backup/restore and migration tests.
- Introduce a `VectorStore` interface so Chroma can be retained initially and replaced later without changing RAG orchestration.

## 4.6 `src/retrieval/retriever.py` — `Retriever`

**Current role:** Semantic retrieval plus lexical retrieval.

**Critical problems:**

- `_bm25_search()` is not BM25; it is unique-token overlap divided by combined unique-token counts.
- It calls `collection.get()` and scans all stored documents on every query. This is an O(N) full collection scan and will become increasingly expensive.
- It drops BM25 result IDs and metadata.
- Matching by full document text can merge identical text from different sources.
- `_combine_results()` uses `max(semantic_score, bm25_score)`, not the documented 70/30 weighting.
- The final result count uses `self.top_k` rather than the `top_k` argument.
- `retrieval_score_threshold` is configured but ignored.
- Query normalization, tokenization, language handling and metadata filters are absent.

**Required design:**

- Implement true BM25 using a maintained implementation/index.
- Maintain the lexical index during ingest/update/delete; do not rebuild/scan the entire collection per query.
- Retrieve candidate lists independently from vector and lexical indexes.
- Combine ranked lists with Reciprocal Rank Fusion (RRF) as the initial default.
- Preserve IDs and metadata throughout fusion.
- Apply tenant/document filters before retrieval.
- Make candidate top-k, final top-k and thresholds explicit.
- Log retrieval latency and candidate counts.
- Add a golden query set with expected relevant source/chunk IDs.

## 4.7 `src/retrieval/reranker.py` — `Reranker`

**Current role:** Cross-encoder reranking.

**Problems:**

- Model loading is eager when `RAGPipeline` initializes.
- No explicit maximum input length/truncation policy is exposed.
- Scores are raw model scores and are not calibrated probabilities.
- It mutates the incoming dictionaries.
- No graceful fallback if model inference fails.
- No separate measurement of reranking latency.

**Required design:**

- Load once per worker and configure device/batch size.
- Keep raw reranker score separate from retrieval score.
- Never label raw score as a probability/confidence.
- Handle model errors with a controlled fallback and telemetry.
- Limit candidate count before reranking.
- Evaluate whether reranking improves Recall@K, MRR and nDCG on the golden dataset.

## 4.8 `src/llm/local_llm.py` — `LocalLLM`

**Current role:** Requests-based client for Ollama and OpenAI-compatible chat completions.

**Problems:**

- Provider detection relies on whether `"/v1"` appears in the URL.
- Ollama endpoint construction assumes the base URL does not already include `/api/generate`; current docs/example configuration can include it, causing duplicate paths.
- Default config points to an OpenAI-compatible local server port while documentation describes Ollama's default port.
- `is_available()` always calls `/api/tags`, which is not a valid generic health check for OpenAI-compatible providers.
- `temperature or default` and `max_tokens or default` reject intentional zero values.
- Retry logic exists both in the HTTP adapter and in the method, creating overlapping retry behavior.
- Broad exception handling can retry non-retryable errors.
- No streaming response implementation.
- No structured output contract, token usage reporting, provider-specific timeout settings or model capability checks.
- No prompt/context length preflight.

**Required design:**

Create a provider interface, for example:

- `OllamaProvider`
- `OpenAICompatibleProvider` (LM Studio, vLLM and compatible servers)
- Optional `AnthropicProvider` or other hosted provider only if explicitly configured with a separate API credential and approved commercial terms.

Each provider must implement:

- `generate()`
- `stream()`
- `health_check()`
- `list_models()` where supported
- provider-specific timeout/error mapping
- token usage/cost metadata where available

Use explicit provider configuration; do not infer provider from URL substrings. Keep model serving independent from the FastAPI process.

## 4.9 `src/qa/qa_pipeline.py` — `RAGPipeline`

**Current role:** Retrieve, optionally rerank, build context and call the LLM.

**Problems:**

- Prompt asks for a helpful answer but does not explicitly require evidence-only answers or abstention when context is insufficient.
- Retrieved document content is untrusted input; there are no prompt-injection boundaries or instructions to ignore instructions inside documents.
- Context truncates each source to 500 characters, potentially removing the evidence needed to answer.
- It appends an ellipsis even when text may not be truncated.
- Source metadata is represented as a path/string, not structured citations.
- No conversation history or query rewriting.
- No query classification, metadata filter, retrieval diagnostics or response timing.
- No structured answer schema.
- No explicit context budget based on model context window.
- No feedback, answer trace or trace ID.

**Required design:**

Split pipeline responsibilities into testable services:

1. Query validation and optional rewrite.
2. Tenant/permission-scoped retrieval.
3. Candidate fusion.
4. Reranking.
5. Context assembly with source references.
6. Prompt construction with untrusted-context boundaries.
7. Generation.
8. Citation validation.
9. Response formatting and telemetry.

The system must say it does not have enough information when evidence is insufficient. It must not fabricate citations or present retrieval similarity as answer confidence.

## 4.10 `src/api/routes.py` — API routes and request models

**Current role:** Query, batch, schedule, customer support, ingestion, stats and database health.

**Problems:**

- Public ingestion accepts arbitrary server paths. This is unsuitable and can expose local files.
- No authentication, RBAC, tenant context or document ownership enforcement.
- Async routes run blocking RAG/LLM/database code directly.
- Batch implementation is sequential despite the parallel-processing claim.
- Error handlers return raw exception messages in some 500 responses.
- `customer_id` is echoed but not used for authorization, data isolation or persistence.
- `priority` is echoed but does not trigger escalation/routing.
- Confidence is copied from the first retrieval score; this is not calibrated answer confidence.
- Schedule request validates ISO syntax but not a timezone-aware future time.
- Job listing/status behavior is incomplete after restart.
- CORS currently blocks browser DELETE and Authorization headers.
- No document list, delete, update, upload, chat, history, feedback or analytics APIs.

**Required design:**

- Version all endpoints under `/api/v1`.
- Use `UploadFile` multipart uploads, not file paths.
- Apply authentication and tenant authorization to every document/chat operation.
- Use Pydantic v2 models and a common error schema.
- Use controlled worker execution for blocking tasks.
- Use background jobs for ingestion and large batch requests.
- Return `202 Accepted` plus job ID for long ingestion jobs.
- Implement document lifecycle and conversation APIs.
- Return structured citations with document ID, filename, page, section, chunk ID and snippet.
- Never expose local server paths.
- Add pagination, filtering and idempotency keys where appropriate.

## 4.11 `src/api/app.py` — FastAPI app and `RateLimiter`

**Problems:**

- Rate limit is hard-coded to 100 instead of reading configuration.
- Rate limiting is in-memory and per-process; it is not shared across multiple workers/instances.
- Client identity is based on `request.client.host`, which may be a reverse proxy address in deployment.
- The in-memory request dictionary can grow with many client identifiers.
- CORS is hard-coded to localhost origins.
- CORS methods omit DELETE; allowed headers omit Authorization.
- No lifespan-managed initialization/cleanup of clients and workers.
- `/health` is only a liveness check; no readiness check verifies model/vector DB/database dependencies.
- No structured request ID/trace ID or standard error response.

**Required design:**

- Use FastAPI lifespan for controlled dependency startup/shutdown.
- Add `/health/live` and `/health/ready`.
- Use Redis-backed rate limiting or gateway-level limits in multi-instance production.
- Configure CORS from environment.
- Support Authorization and required methods.
- Use trusted proxy configuration only when deployed behind a known proxy.
- Add request IDs, structured logs and metrics.
- Do not log raw customer prompts or document contents by default.

## 4.12 `src/api/persistent_scheduler.py` — `ScheduledJob`, `PersistentScheduler`

**Critical bug:** Restart recovery does not restore the executable callback. `_load_pending_jobs()` uses a timer callback that only calls `_mark_job_running()`. It does not run the original query.

Other problems:

- A `Timer` per job is not a robust distributed scheduler.
- Multiple API instances can execute the same persisted job.
- `_save_job()` is called while holding a lock and performs database I/O.
- Completed/failed jobs are not fully loaded into the in-memory `jobs` dictionary on startup.
- `list_jobs()` reads only in-memory jobs, so it can omit persisted completed jobs.
- The job payload stores a query but not a durable task handler/version contract.
- Fallback JSON storage has no multi-process locking or robust atomic write strategy.
- No retry policy, lease/lock, idempotency, dead-letter handling or worker separation.
- Timezone handling needs a single explicit policy.

**Required design:**

For a commercial RAG product, move ingestion and long-running tasks to a queue/worker system. Use Redis plus Celery, RQ or Dramatiq (choose one; do not implement all).

- Persist job records in the application database.
- Queue job IDs, not Python callbacks.
- Worker loads job type and payload from the database.
- Use idempotent handlers.
- Add retry/backoff, max attempts and failed-job inspection.
- Use a lock/lease to prevent duplicate execution.
- Keep API request/response fast.
- If scheduled queries remain a product feature, implement scheduling with a durable scheduler designed for multiple workers.

## 4.13 `src/api/scheduler.py` — `InProcessScheduler`

This is a second scheduler implementation. It is process-memory-only and duplicates the persistent scheduler concept. It is not the scheduler imported by the current routes, but its presence increases ambiguity.

**Action:** Remove it after migration, or keep only as a test fake. There should be one production job architecture.

## 4.14 `src/evaluation/evaluator.py` — `RAGEvaluator`

**Problems:**

- RAGAS imports are attempted but actual RAGAS evaluation is not executed.
- The reported answer relevancy/faithfulness are word-overlap heuristics.
- `ground_truths` length is not validated against predictions.
- The `ContextPrecision`/`ContextRecall` imports are unused.
- Empty prediction lists and missing fields need robust handling.
- No golden dataset management or experiment/version tracking.
- No retrieval ranking metrics such as Recall@K, MRR or nDCG.
- No latency/cost breakdown.

**Required design:**

Implement a proper offline evaluation harness:

- Question
- Expected answer/ground truth
- Expected source document/chunk IDs
- Retrieved contexts
- Generated answer
- Citation references
- Model and prompt versions

Measure retrieval and generation separately:

- Recall@K
- Precision@K
- MRR
- nDCG
- Context precision/recall
- Faithfulness/groundedness
- Answer relevancy
- Citation correctness
- No-answer/abstention correctness
- Retrieval, reranking and generation latency
- Token usage/cost for hosted providers

Use a pinned, compatible RAGAS version or another clearly documented evaluator. Never label a heuristic as a RAGAS score.

## 4.15 `src/ui/streamlit_app.py` — Streamlit UI

**Current state:** A 990-line single-file Streamlit application with search, batch, scheduling and document upload pages.

**Problems:**

- It is not a React application and is not structured as a commercial multi-user frontend.
- CSS, page routing, API calls and presentation are all in one file.
- Uses `unsafe_allow_html=True` with LLM answers, retrieved text, filenames and error content. This is an HTML/script injection risk.
- Query history is in Streamlit session state only; it is not persistent chat history.
- “Show sources” displays snippets/context but does not provide verified clickable citations.
- Page metadata is shown using `metadata["pages"]`, which is total page count, not the actual page for a chunk.
- Upload creates a temporary directory, writes files there and sends local paths to the backend.
- Temporary upload directories are not cleaned up.
- This upload method breaks when frontend and backend are deployed separately.
- Scheduled query UI blocks in a long `while`/`sleep` polling loop.
- Batch UI depends on a backend endpoint that is not actually parallel.
- UI displays “High” quality without a measured quality signal.
- API URL is local by default and not a robust deployment configuration.
- No authentication, user settings, workspace/tenant selection or role-specific views.

**Recommendation:** Do not spend substantial time polishing this Streamlit UI. Keep it temporarily as an internal developer console if useful, but build the customer-facing interface separately with React + TypeScript.

## 4.16 `process_manager.py` — `ProcessManager` and Flask dashboard

**Problems:**

- It is a separate Flask control panel, not integrated with FastAPI.
- It starts/stops fixed ports rather than using the configured service settings.
- `kill_by_port()` can kill any process using a port, not necessarily a process owned by RAG-Lines.
- No authentication/authorization protects process-control endpoints.
- Standard output/error are discarded, making failures harder to debug.
- It uses Flask and psutil, which are not listed in the shown requirements file.
- It does not manage the model runtime consistently.
- It duplicates responsibilities better handled by Docker Compose, systemd or the deployment platform.

**Action:** Remove from the customer product. Use Docker Compose for local development and a proper process/orchestration service for deployment.

## 4.17 `main.py`

**Strengths:** Simple CLI for ingest, QA and API startup.

**Problems:**

- Imports core modules at the top level, increasing startup coupling.
- API startup does not itself guarantee that the model runtime is available.
- CLI ingestion and API ingestion use different entry paths but do not share a well-defined ingestion service.
- There is no explicit `worker` or migration command.
- No clear production command contract.

**Action:** Retain a small CLI, but make it invoke shared application services. Add commands for migrations, worker startup and safe document ingestion if needed.

## 4.18 `requirements.txt`

**Problems:**

- LangChain packages are included but the reviewed core pipeline does not use them.
- The `ollama` package is included while the current LLM client uses `requests`.
- RAGAS is not listed despite being claimed in documentation.
- Flask and psutil are used by `process_manager.py` but are not listed.
- Dependency ranges are broad and there is no lock file shown.
- SQLAlchemy version range should be validated against actual runtime compatibility; do not assume the requested minimum is available in every deployment environment.
- PDF libraries are duplicated (PyPDF2 and pdfplumber) without a clear adapter strategy.

**Action:**

- Remove unused dependencies.
- Add only libraries used by the chosen architecture.
- Pin tested versions using a lock file.
- Separate runtime and development dependencies.
- Run dependency vulnerability checks.
- Add a clean CI installation test.

## 4.19 Tests

`tests/test_pipeline.py` primarily tests document object construction, whether chunkers return chunks and embedding shape. It loads real embedding model code, which makes tests slow/network-dependent and not deterministic.

`test_fixes.py` is largely a manual verification script rather than a clean set of isolated automated tests.

**Missing tests:**

- API request/response contract tests
- Upload security tests
- File parser tests with fixtures
- PDF page citation tests
- Chunk ID stability tests
- Duplicate ingestion/update/delete tests
- Embedding model compatibility tests
- Vector-store filtering tests
- BM25 correctness tests
- RRF ranking tests
- Reranker tests with mocks
- Prompt-injection tests
- Abstention/grounding tests
- Provider health/retry/timeout tests
- Scheduler restart/idempotency tests
- Tenant isolation tests
- CORS/auth tests
- Frontend component/API tests
- End-to-end Docker Compose smoke tests

Use mocks for model calls in unit tests. Keep a small number of opt-in integration tests for real local model execution.

---

# 5. Recommended target architecture

## 5.1 Architecture principle

Start with a **modular monolith**, not a large microservices system.

One Python backend repository can contain well-separated modules and expose FastAPI endpoints. Run the API, background worker and model runtime as separate processes/containers. This gives you clear operational boundaries without creating unnecessary service complexity.

## 5.2 Proposed technology choices

| Layer | Recommended starting choice | Notes |
|---|---|---|
| Customer UI | React + TypeScript + Vite | Separate frontend project; use a component system and responsive design. |
| API | FastAPI + Pydantic v2 | Keep Python; reorganize into routers/services/schemas. |
| Database | PostgreSQL + SQLAlchemy + Alembic | Users, tenants, documents, conversations, jobs, feedback and audit metadata. |
| Vector store | Keep Chroma adapter initially | Do not migrate until benchmark and isolation requirements justify it. Evaluate pgvector or Qdrant later. |
| Lexical retrieval | Real BM25 implementation | Maintain index during document lifecycle. |
| Fusion | Reciprocal Rank Fusion | Combine ranked candidates robustly. |
| Reranking | Cross-encoder adapter | Optional, configurable, measured. |
| Embeddings | Sentence Transformers adapter | Keep separate from generation model. |
| Local inference | Ollama for development; evaluate vLLM for GPU production | Do not run model weights inside React or the FastAPI process. |
| Hosted inference | Optional provider adapters | Must be opt-in, configured securely and disclosed to customer. |
| Background jobs | Redis + Celery, RQ or Dramatiq | Choose one; use durable job records and idempotent workers. |
| File storage | Local volume in development; S3-compatible object storage in production | Store original files outside vector DB. |
| Observability | Structured logs + metrics + tracing | Redact prompts and customer data by default. |
| Deployment | Docker Compose for local; managed containers/VMs for production | Separate frontend, API, worker, database, Redis and model runtime. |
| CI/CD | Automated lint, type checks, tests, image build and deployment | Do not deploy directly from untested code generated by an agent. |

## 5.3 High-level system

```text
Customer Browser
    |
    v
React + TypeScript Frontend
    |
    | HTTPS / JSON / SSE
    v
FastAPI API
    |
    +--> Authentication / RBAC / Tenant Context
    |
    +--> Document Service
    |       |
    |       +--> Object Storage (original files)
    |       +--> Job Queue --> Ingestion Worker
    |                            |
    |                            +--> Parser / OCR
    |                            +--> Token-aware Chunker
    |                            +--> Embedding Provider
    |                            +--> Chroma / Vector Store
    |                            +--> BM25 Index
    |
    +--> Chat / RAG Service
    |       |
    |       +--> Query Processing
    |       +--> Tenant-scoped Vector Retrieval
    |       +--> BM25 Retrieval
    |       +--> RRF Fusion
    |       +--> Reranking
    |       +--> Context Assembly / Citation Mapping
    |       +--> LLM Provider
    |       +--> Citation Validation / Abstention
    |
    +--> PostgreSQL
    |       +--> Tenants / Users / Roles
    |       +--> Documents / Versions / Chunks metadata
    |       +--> Conversations / Messages
    |       +--> Jobs / Feedback / Audit records
    |
    +--> Redis / Background Workers
    |
    +--> Observability
```

## 5.4 Service processes and how to start them

Do not make one Python script responsible for launching every service in production.

Development should have distinct commands/services:

1. **Frontend:** React development server, e.g. Vite.
2. **API:** FastAPI/Uvicorn.
3. **Worker:** queue worker process.
4. **Model runtime:** Ollama locally, or a separately hosted inference server.
5. **Database:** PostgreSQL.
6. **Queue/cache:** Redis.
7. **Object storage:** local development volume or S3-compatible service.

Docker Compose should start/stop the local stack with one command. The services remain independently observable and restartable.

The API should not automatically download multi-gigabyte model weights on every startup. Model installation/pull is a setup or deployment step. Startup should check the configured model is available and report readiness.

---

# 6. Product features required to sell RAG-Lines

Do not build every possible feature at once. Build a reliable core product first.

## 6.1 Essential customer-facing features

### Workspace and access management
- Registration/invitation flow
- Login/logout
- Password reset or SSO later
- Organizations/workspaces
- Roles: owner, admin, editor, viewer
- Tenant-level data isolation
- API keys for integrations
- Audit log for administrative actions

### Document management
- Drag-and-drop upload
- Multiple-file upload
- Supported file-type/size validation
- Upload progress
- Ingestion status: queued, processing, completed, failed
- Retry failed ingestion
- List/search/filter documents
- Document preview
- Page-level source mapping
- Delete document
- Replace/update document
- Version history
- Duplicate detection
- Storage usage and limits
- Optional OCR for scanned PDFs

### Chat and knowledge retrieval
- New conversation
- Conversation history
- Rename/delete conversation
- Streaming responses
- Source citations linked to document/page
- Expandable evidence snippets
- Clear “not found in uploaded sources” behavior
- Follow-up questions
- Feedback: helpful/not helpful
- Copy answer
- Export conversation
- Filters by selected document/collection
- Optional response mode: concise/detailed

### Admin and analytics
- Document count and storage usage
- Ingestion success/failure
- Query volume
- Latency breakdown
- Model/provider health
- Retrieval diagnostics for administrators
- Feedback trends
- Usage limits and plan entitlements

## 6.2 Features to postpone

Do not begin with these until the core is reliable:

- Fine-tuning models
- Autonomous agents
- Complex multi-agent workflows
- Many vector database backends
- Multiple queue systems
- Workflow marketplace
- Arbitrary web crawling
- Complex scheduled query automation
- Custom model training
- Dozens of integrations

A focused, trustworthy document-intelligence product is easier to test, deploy, support and sell.

---

# 7. API contract proposal

Keep `/api/v1` and define a stable contract. Suggested endpoints:

## Authentication and workspace
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/refresh`
- `POST /api/v1/auth/logout`
- `GET /api/v1/me`
- `GET /api/v1/workspaces`
- `POST /api/v1/workspaces`
- `GET /api/v1/workspaces/{workspace_id}/members`
- `POST /api/v1/workspaces/{workspace_id}/members`

## Documents
- `POST /api/v1/documents` — multipart upload; returns document ID and job ID.
- `GET /api/v1/documents` — paginated list, workspace scoped.
- `GET /api/v1/documents/{document_id}` — metadata/status.
- `GET /api/v1/documents/{document_id}/content` — authorized preview.
- `DELETE /api/v1/documents/{document_id}` — remove document and derived chunks.
- `POST /api/v1/documents/{document_id}/reprocess` — create a new processing job.
- `GET /api/v1/jobs/{job_id}` — job status/progress/error.

## Chat
- `POST /api/v1/conversations`
- `GET /api/v1/conversations`
- `GET /api/v1/conversations/{conversation_id}`
- `DELETE /api/v1/conversations/{conversation_id}`
- `POST /api/v1/conversations/{conversation_id}/messages`
- `POST /api/v1/conversations/{conversation_id}/messages/stream` — SSE or a clearly defined streaming route.
- `POST /api/v1/feedback`

## Admin/health
- `GET /api/v1/analytics/usage`
- `GET /api/v1/admin/system`
- `GET /health/live`
- `GET /health/ready`

Every endpoint must enforce authorization and workspace scope. Do not trust a workspace ID supplied by the browser without checking membership.

## Suggested chat response schema

```json
{
  "conversation_id": "uuid",
  "message_id": "uuid",
  "answer": "Answer grounded in the uploaded documents.",
  "citations": [
    {
      "citation_id": "C1",
      "document_id": "uuid",
      "filename": "employee-policy.pdf",
      "page": 8,
      "chunk_id": "uuid",
      "snippet": "Relevant supporting passage..."
    }
  ],
  "retrieval": {
    "candidate_count": 20,
    "reranked_count": 5
  },
  "timing_ms": {
    "retrieval": 120,
    "reranking": 85,
    "generation": 1800
  },
  "status": "answered"
}
```

This is a proposed contract, not a claim that the current application returns this schema.

Do not expose raw local file paths, internal prompts, secrets or unfiltered vector-store metadata in customer responses.

---

# 8. Model and provider strategy — important for Claude usage

There are two separate model-related decisions.

## 8.1 Claude is your development assistant

Use Claude Opus for:
- Architecture review
- Large refactors
- Complex debugging
- Security-sensitive code review
- Reviewing implementation plans and diffs

Use Claude Sonnet for:
- Most feature implementation
- Tests
- UI components
- Routine refactoring

Use Claude Haiku for:
- Small documentation changes
- Simple repetitive edits
- Lightweight code explanations

If “Hyco” meant Claude Haiku, this is the intended distinction.

Do not ask the coding agent to replace every file in one giant operation. Ask it to inspect, plan, implement one phase, run tests and report changed files before moving to the next phase.

## 8.2 Claude is not automatically your application's inference runtime

The customer's RAG request should be processed by the model provider configured for the deployed application.

Recommended provider options:

- **Ollama:** easiest local development and single-machine model serving.
- **vLLM:** evaluate for GPU-backed production serving and throughput.
- **OpenAI-compatible server:** useful adapter for LM Studio/vLLM and other compatible runtimes.
- **Hosted model API:** optional provider, only with explicit customer configuration, cost controls, data disclosure and suitable contractual/privacy terms.

A Claude Enterprise seat should not be assumed to include production API access or permission to resell API usage. If you want Claude models as an inference option in the product, separately verify Anthropic API access, billing, data handling and applicable commercial terms.

Keep the provider interface independent of the RAG pipeline so you can switch models without rewriting retrieval or the UI.

## 8.3 Model runtime lifecycle

- Download/pull models during setup or deployment.
- Store model weights in persistent model storage.
- Start the inference runtime as a separate service.
- Configure API with provider URL, model ID, timeout and limits.
- Run a health check before declaring the service ready.
- Do not expose Ollama/model-server ports publicly.
- Do not let the browser call the model server directly.
- Add concurrency limits and queueing to protect GPU/CPU memory.
- Record model/version in query telemetry.

---

# 9. Phased implementation plan for Claude

Claude must work in phases. Each phase must end with tests and a written report.

## Phase 0 — Baseline and safe preparation

- Inspect the repository and produce a dependency/file map.
- Do not delete or rewrite code.
- Identify secrets and local data; do not print secrets.
- Add/verify `.gitignore`.
- Prepare `.env.example` with placeholders.
- Add a lock file and clean installation instructions.
- Record current CLI/API behavior as a baseline.
- Create a branch before changes.

**Exit criteria:** clean repository inventory, secrets checklist, baseline test report and migration plan.

## Phase 1 — Stabilize configuration and backend foundations

- Standardize Pydantic Settings v2.
- Establish `RAG_` environment prefix.
- Fix database URL construction.
- Fix vector-store path configuration.
- Add structured errors and request IDs.
- Add application lifespan and readiness checks.
- Remove duplicate process/scheduler startup paths.
- Add linting, typing and test configuration.

**Exit criteria:** application starts predictably with validated configuration; health/readiness tests pass.

## Phase 2 — Correct RAG quality

- Fix token-aware chunking and metadata.
- Add deterministic document/chunk IDs.
- Preserve PDF page/section provenance.
- Fix duplicate ingestion and update/delete semantics.
- Implement real BM25.
- Implement RRF.
- Fix `top_k` propagation.
- Add metadata/tenant filters.
- Improve context construction, evidence boundaries and abstention.
- Add citation mapping.
- Add golden retrieval dataset and regression tests.

**Exit criteria:** retrieval tests show expected sources; citations map to actual page/chunk; unsupported questions trigger abstention.

## Phase 3 — Provider and model runtime

- Add explicit provider interface.
- Implement Ollama adapter.
- Implement OpenAI-compatible adapter.
- Add optional hosted provider adapter only if requested.
- Fix URL handling and health checks.
- Add streaming.
- Add timeouts, retries and concurrency controls.
- Add model readiness checks.
- Add tests with mocked provider responses.

**Exit criteria:** provider can be switched via configuration without changing RAG service or frontend.

## Phase 4 — Document lifecycle and background worker

- Add multipart upload endpoint.
- Add persistent object storage abstraction.
- Add document records and versions.
- Add background ingestion queue.
- Add job progress/status.
- Add retry and failed-job handling.
- Add list/preview/delete/reprocess APIs.
- Ensure vector and lexical indexes update together.
- Add duplicate and rollback tests.

**Exit criteria:** files can be uploaded from a separate frontend host, processed asynchronously, listed and deleted safely.

## Phase 5 — React customer UI

Build a new React + TypeScript application. Do not convert the 990-line Streamlit file line by line.

Screens:
1. Login
2. Workspace selection
3. Dashboard
4. Documents list
5. Upload/ingestion progress
6. Chat workspace
7. Conversation history
8. Citation/source viewer
9. Settings
10. Usage/admin panel

Use a clean reusable component architecture, API client layer, typed response schemas, loading/error/empty states, responsive layouts and accessible controls.

**Exit criteria:** React uses only FastAPI endpoints; no direct browser access to Chroma, Ollama, PostgreSQL or object storage.

## Phase 6 — Security and commercial readiness

- Authentication and RBAC
- Workspace/tenant isolation
- Secure upload validation
- Rate limiting
- API key management
- Secret management
- Audit logging
- CORS/CSRF policy as appropriate
- Prompt-injection and data-leakage tests
- Dependency and container scans
- Backup/restore
- Data retention and deletion
- Usage quotas and model cost controls

**Exit criteria:** security tests prove one tenant cannot read, search, modify or delete another tenant's documents or conversations.

## Phase 7 — Evaluation, deployment and release

- Real RAG evaluation
- Golden dataset regression
- Load and concurrency tests
- Docker Compose local stack
- Production container build
- CI pipeline
- Database migrations
- Observability dashboards
- Deployment/rollback guide
- Backup/restore drill
- Release checklist

**Exit criteria:** a clean environment can deploy the application from documented instructions and pass smoke, integration, security and evaluation gates.

---

# 10. Proposed repository structure

This is a target structure, not a requirement to rename every file immediately. Migrate incrementally and preserve behavior while tests are added.

```text
rag-lines/
├── apps/
│   └── web/                       # React + TypeScript frontend
├── backend/
│   ├── pyproject.toml
│   ├── alembic/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   ├── security.py
│   │   │   └── errors.py
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── auth.py
│   │   │       ├── workspaces.py
│   │   │       ├── documents.py
│   │   │       ├── conversations.py
│   │   │       ├── jobs.py
│   │   │       └── admin.py
│   │   ├── schemas/
│   │   ├── models/
│   │   ├── repositories/
│   │   ├── services/
│   │   │   ├── document_service.py
│   │   │   ├── ingestion_service.py
│   │   │   ├── chat_service.py
│   │   │   └── workspace_service.py
│   │   ├── rag/
│   │   │   ├── pipeline.py
│   │   │   ├── chunking/
│   │   │   ├── embeddings/
│   │   │   ├── retrieval/
│   │   │   ├── reranking/
│   │   │   ├── context/
│   │   │   ├── citations/
│   │   │   └── evaluation/
│   │   ├── providers/
│   │   │   ├── llm/
│   │   │   ├── vectorstores/
│   │   │   └── storage/
│   │   ├── workers/
│   │   └── tests/
├── infra/
│   ├── docker/
│   └── compose.yaml
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── security.md
│   ├── local-development.md
│   └── deployment.md
├── .env.example
├── .gitignore
└── README.md
```

Do not create empty folders just to match this tree. Create modules when the corresponding feature is implemented and tested.

---

# 11. Acceptance criteria — what “excellent and sellable” means

The product is not complete merely because the chat returns an answer.

## RAG correctness
- Every citation resolves to a real document and correct page/chunk.
- No unsupported answer is presented as fact.
- Retrieval is evaluated with a versioned golden dataset.
- BM25 is genuine lexical retrieval.
- Fusion/reranking are measurable and configurable.
- Re-ingesting or replacing a document does not create uncontrolled duplicates.

## Security
- All protected routes require authentication.
- Every document, conversation, job and vector query is tenant scoped.
- Upload validation prevents path traversal and unsafe file types.
- No secrets in Git, logs or API responses.
- Model and database ports are private.
- Prompt injection in source documents is tested.
- Tenant isolation tests pass.

## Reliability
- Ingestion runs asynchronously.
- Failed jobs can be inspected and retried.
- Jobs survive process restarts and do not execute twice unexpectedly.
- Database and vector-store backup/restore are documented and tested.
- Readiness checks verify required dependencies.
- API has timeouts and bounded concurrency.

## User experience
- Upload progress and clear processing status.
- Responsive React UI.
- Chat history.
- Streaming answers.
- Clickable source citations.
- Useful empty/error states.
- Document update/delete.
- No fake “quality” or “confidence” labels.

## Operations
- One-command local startup using Docker Compose.
- Separate frontend/API/worker/model runtime.
- Version-pinned dependencies.
- CI tests on every change.
- Logs, metrics and request IDs.
- Deployment and rollback instructions.
- Usage limits and model cost visibility.

---

# 12. Claude execution rules

Claude must follow these rules throughout development:

1. **Inspect before editing.** Read relevant modules, tests, configuration and docs before proposing a change.
2. **No blind rewrites.** Do not replace the entire repository or delete working code in one operation.
3. **Preserve working behavior.** Add tests before changing behavior; document intentional API changes.
4. **Small phases.** Work on one phase and a small set of related files at a time.
5. **No fake completion.** Do not say a feature is complete unless its tests pass.
6. **Run tests.** Run relevant unit/integration tests, lint and type checks after each phase.
7. **Report files.** List files added/changed/deleted and explain why.
8. **No secret disclosure.** Never print `.env` values, tokens, passwords, private URLs or customer content.
9. **No production data.** Use synthetic fixtures.
10. **Ask before destructive migrations.** Never delete Chroma data, databases, model files or documents without an explicit migration/backup plan.
11. **Do not invent performance metrics.** Benchmark before claiming latency or quality improvements.
12. **Keep docs synchronized.** Update API, architecture, setup and deployment documentation with each completed phase.
13. **No microservices by default.** Use a modular monolith plus separate API/worker/model processes until a measured scaling need justifies more.
14. **No unnecessary dependencies.** Explain why each new package is needed.
15. **Security is a release gate.** Do not call the system production-ready while P0 security issues remain.

---

# 13. Immediate next steps for the project owner

1. Preserve the original project in a backup branch.
2. Rotate any real/reused credentials found in the archive and clean the repository.
3. Create a sanitized working copy without local data, logs, secrets, `.git` history or caches.
4. Give Claude the sanitized source and this review document.
5. Ask Claude to complete **Phase 0 only** and return its audit/report.
6. Review the plan before authorizing Phase 1.
7. Build the new React UI only after the backend upload/chat contracts are agreed.
8. Keep Ollama as the local development runtime initially; evaluate GPU serving only when deployment hardware and expected concurrency are known.

## Final verdict

**Continue with RAG-Lines as the foundation. Refactor the Python backend, correct the RAG algorithms, replace the unsafe/local-path ingestion flow, add a durable worker architecture, and build a new React frontend.**

Do not start with a complete rewrite, and do not start by polishing the existing Streamlit interface. First make the backend contracts, retrieval quality, security and document lifecycle dependable. Then build the customer UI on top of those stable APIs.
