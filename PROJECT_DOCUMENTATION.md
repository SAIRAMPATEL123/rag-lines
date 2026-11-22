# RAG-Lines: Complete Project Documentation

**Project Version:** 1.2.0  
**Last Updated:** 2026-05-03  
**Status:** Production Ready (MVP)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [API Reference](#api-reference)
6. [Core Components](#core-components)
7. [Configuration](#configuration)
8. [Quick Start Guide](#quick-start-guide)
9. [Features Matrix](#features-matrix)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## 📌 Project Overview

**RAG-Lines** is a comprehensive **Retrieval-Augmented Generation (RAG)** system built in Python for Q&A and customer support applications. It supports multiple document types with local LLM inference, eliminating external API dependencies.

### Core Capabilities
- Multi-format document support (PDF, Markdown, Plain Text, Web Content, Databases)
- Local LLM inference via Ollama (no external dependencies)
- Semantic vector database (Chroma)
- Hybrid retrieval (semantic + BM25 search)
- Quality metrics via RAGAS evaluation framework
- Scalable from local development to server deployment
- FastAPI REST API with rate limiting
- Streamlit web UI
- Scheduled query execution

### Key Benefits
- **Privacy**: All data processed locally
- **Cost-Effective**: No third-party API charges
- **Fast**: Sub-second retrieval and generation
- **Flexible**: Easy to integrate with existing systems

---

## 🏗️ Architecture

### High-Level Flow

```
┌─────────────────┐
│   User Input    │
│   (UI/API)      │
└────────┬────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  API Layer (FastAPI + Rate Limiting)    │
    │  - Query routing                        │
    │  - Batch processing                     │
    │  - Scheduled jobs                       │
    └────┬───────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────┐
    │     RAG Pipeline Orchestration        │
    │  ┌─────────────────────────────────┐  │
    │  │ 1. Document Retrieval           │  │
    │  │    - Semantic search (Chroma)   │  │
    │  │    - BM25 lexical search        │  │
    │  │    - Hybrid combination         │  │
    │  └─────────────────────────────────┘  │
    │  ┌─────────────────────────────────┐  │
    │  │ 2. Result Reranking (optional)  │  │
    │  │    - Cross-encoder scoring      │  │
    │  │    - Top-5 selection            │  │
    │  └─────────────────────────────────┘  │
    │  ┌─────────────────────────────────┐  │
    │  │ 3. Context Building             │  │
    │  │    - Format metadata            │  │
    │  │    - Extract snippets           │  │
    │  └─────────────────────────────────┘  │
    │  ┌─────────────────────────────────┐  │
    │  │ 4. LLM Generation               │  │
    │  │    - Prompt construction        │  │
    │  │    - Local inference (Ollama)   │  │
    │  └─────────────────────────────────┘  │
    └────┬──────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────┐
    │   Response Formatting & Logging       │
    │   - Answer + retrieved documents     │
    │   - Performance metrics              │
    │   - Structured logging               │
    └────┬──────────────────────────────────┘
         │
    ┌────▼────────────────┐
    │  User Response      │
    │  (JSON/UI Display)  │
    └─────────────────────┘
```

### Data Flow Layers

1. **Ingestion Layer** → Document loading & chunking
2. **Embedding Layer** → Vector representation generation
3. **Storage Layer** → Chroma persistent vector database
4. **Retrieval Layer** → Hybrid semantic + lexical search
5. **Reranking Layer** → Optional cross-encoder refinement
6. **Generation Layer** → Local LLM inference
7. **API Layer** → FastAPI route handling & serialization
8. **UI Layer** → Streamlit frontend

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Backend API** | FastAPI | Latest | REST API framework |
| **ASGI Server** | Uvicorn | Latest | Production server |
| **Vector Database** | ChromaDB | 1.x | Semantic search storage |
| **Embeddings** | sentence-transformers | Latest | Text → Vector conversion |
| **LLM Runtime** | Ollama | Latest | Local model inference |
| **Reranker** | cross-encoder | Latest | Result ranking (optional) |
| **Chunking** | Custom Python | 1.0 | Semantic/fixed chunking |
| **Task Scheduling** | APScheduler | Latest | Delayed job execution |
| **UI Framework** | Streamlit | Latest | Web interface |
| **Database** | MySQL (optional) | 8.0+ | Persistent job storage |
| **Logging** | Loguru | Latest | Structured logging |
| **Configuration** | Pydantic | Latest | Settings management |
| **Testing** | pytest | Latest | Unit testing |

---

## 📁 Project Structure

```
rag-lines/
├── config/
│   ├── __init__.py
│   └── config.py                      # Centralized configuration
├── docs/
│   └── PROJECT_ARCHITECTURE.md        # System architecture notes
├── scripts/
│   └── start_app.sh                   # One-command launcher
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py         # Multi-format document loading
│   │   └── chunker.py                 # Semantic/fixed chunking
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py         # Vector representation
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_db.py               # ChromaDB integration
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py               # Hybrid retrieval (semantic + BM25)
│   │   └── reranker.py                # Cross-encoder reranking
│   ├── llm/
│   │   ├── __init__.py
│   │   └── local_llm.py               # Ollama/LM Studio integration
│   ├── qa/
│   │   ├── __init__.py
│   │   └── qa_pipeline.py             # End-to-end RAG orchestration
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py               # RAGAS metrics
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI application setup
│   │   ├── routes.py                  # API endpoint definitions
│   │   ├── scheduler.py               # In-process scheduler
│   │   └── persistent_scheduler.py    # DB-backed scheduler
│   └── ui/
│       └── streamlit_app.py           # Streamlit web interface
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py               # Pipeline unit tests
│   └── test_fixes.py                  # Specific fix tests
├── main.py                            # CLI entry point
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── DATABASE_SETUP.md                  # MySQL setup guide
├── README.md                          # Quick start
├── TESTING_GUIDE.md                   # Test documentation
└── PROJECT_DOCUMENTATION.md           # This file

```

---

## 🔌 API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
- Currently **no authentication** (add via middleware for production)
- Rate limiting: **100 requests/minute** per IP

### Response Format
All endpoints return JSON with standard structure:
```json
{
  "status": "success|error",
  "data": {},
  "timestamp": "2026-11-22T10:30:00Z"
}
```

---

### 1. Query Endpoint

#### **POST** `/query`

Execute a single Q&A query with optional document context.

**Request Body:**
```json
{
  "question": "What is customer support?",
  "top_k": 20,
  "include_context": true
}
```

**Parameters:**
| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `question` | string | ✓ | - | 1-2000 chars, non-empty |
| `top_k` | integer | ✗ | 20 | 1-100 |
| `include_context` | boolean | ✗ | false | - |

**Response (Success - 200):**
```json
{
  "question": "What is customer support?",
  "answer": "Customer support is the process of providing assistance...",
  "retrieved_documents": [
    {
      "document": "Full document text (truncated to 500 chars)...",
      "score": 0.892,
      "semantic_score": 0.892,
      "bm25_score": 0.123,
      "metadata": {
        "source": "/path/to/document.pdf",
        "chunk_id": "doc_001_chunk_1"
      }
    }
  ],
  "context_used": "[Document 1] (Score: 0.892)...",
  "document_count": 5
}
```

**Response (No Documents - 200):**
```json
{
  "question": "What is customer support?",
  "answer": "⚠️ No documents found. Please ingest documents first using: python main.py ingest <path>",
  "retrieved_documents": [],
  "context_used": null,
  "document_count": 0
}
```

**Error Response (500):**
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I reset my password?",
    "top_k": 10,
    "include_context": true
  }'
```

**Processing Flow:**
1. Validate question (non-empty, length check)
2. Retrieve top-K documents via hybrid search
3. If empty collection, return warning message
4. Rerank if enabled (top-5 from retrieved)
5. Build context string with metadata
6. Generate answer via local LLM
7. Format and return response with timestamps

**Latency:** ~2-5 seconds (depends on LLM model and query complexity)

---

### 2. Batch Query Endpoint

#### **POST** `/batch-query`

Process multiple questions in a single request with concurrent execution.

**Request Body:**
```json
{
  "questions": [
    "What is customer support?",
    "How to contact support?",
    "What are support hours?"
  ],
  "top_k": 15,
  "include_context": false
}
```

**Parameters:**
| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `questions` | array[string] | ✓ | - | 1-100 items, each 1-2000 chars |
| `top_k` | integer | ✗ | 20 | 1-100 |
| `include_context` | boolean | ✗ | false | - |

**Response (Success - 200):**
```json
{
  "count": 3,
  "results": [
    {
      "question": "What is customer support?",
      "answer": "...",
      "retrieved_documents": [...],
      "context_used": "...",
      "document_count": 5
    },
    {
      "question": "How to contact support?",
      "answer": "...",
      "retrieved_documents": [...],
      "context_used": "...",
      "document_count": 4
    },
    {
      "question": "What are support hours?",
      "answer": "...",
      "retrieved_documents": [...],
      "context_used": "...",
      "document_count": 3
    }
  ]
}
```

**Error Response (400):**
```json
{
  "detail": "Each question must be non-empty"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/batch-query \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["Q1?", "Q2?", "Q3?"],
    "top_k": 10
  }'
```

**Processing Flow:**
1. Validate all questions
2. Create ThreadPoolExecutor with 4 workers
3. Execute queries concurrently
4. Aggregate results with original order maintained
5. Return batch response

**Latency:** ~3-6 seconds for 3 queries (concurrent, not sequential)

**Note:** Max 100 questions per batch to prevent resource exhaustion.

---

### 3. Schedule Query Endpoint

#### **POST** `/schedule-query`

Schedule a query to run at a specific future time.

**Request Body:**
```json
{
  "question": "What is customer support?",
  "run_at_iso": "2026-12-01T14:30:00",
  "top_k": 20,
  "include_context": true
}
```

**Parameters:**
| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `question` | string | ✓ | - | 1-2000 chars |
| `run_at_iso` | string | ✓ | - | ISO 8601 format, future time |
| `top_k` | integer | ✗ | 20 | 1-100 |
| `include_context` | boolean | ✗ | false | - |

**Response (Success - 200):**
```json
{
  "job_id": "job_001_abc123def456",
  "status": "scheduled",
  "run_at": "2026-12-01T14:30:00"
}
```

**Error Response (400):**
```json
{
  "detail": "Invalid ISO format datetime"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/schedule-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Get status report",
    "run_at_iso": "2026-11-23T09:00:00",
    "top_k": 15
  }'
```

**Processing Flow:**
1. Validate question and ISO datetime
2. Parse datetime to Python `datetime` object
3. Register job with scheduler
4. Job waits in memory until scheduled time
5. At execution time, run query pipeline
6. Store result (can be retrieved via job_id)

**Limitations:**
- Jobs are **not persistent** across process restarts
- For production, use `persistent_scheduler.py` with MySQL backend
- Jobs run in process thread pool (up to 4 concurrent)

**Note:** Scheduled jobs can be queried, cancelled, or listed using related endpoints below.

---

### 4. Get Scheduled Job

#### **GET** `/schedule-query/{job_id}`

Retrieve status and results of a scheduled job.

**Path Parameters:**
| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job identifier from schedule response |

**Response (Job Pending - 200):**
```json
{
  "job_id": "job_001_abc123def456",
  "status": "scheduled",
  "run_at": "2026-12-01T14:30:00",
  "question": "What is customer support?"
}
```

**Response (Job Completed - 200):**
```json
{
  "job_id": "job_001_abc123def456",
  "status": "completed",
  "run_at": "2026-12-01T14:30:00",
  "result": {
    "question": "What is customer support?",
    "answer": "...",
    "retrieved_documents": [...],
    "document_count": 5
  }
}
```

**Error Response (404):**
```json
{
  "detail": "Job job_invalid not found"
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/schedule-query/job_001_abc123def456
```

---

### 5. Cancel Scheduled Job

#### **DELETE** `/schedule-query/{job_id}`

Cancel a scheduled job before execution.

**Path Parameters:**
| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job identifier |

**Response (Success - 200):**
```json
{
  "job_id": "job_001_abc123def456",
  "status": "cancelled"
}
```

**Error Response (404):**
```json
{
  "detail": "Job job_invalid not found"
}
```

**Example cURL:**
```bash
curl -X DELETE http://localhost:8000/api/v1/schedule-query/job_001_abc123def456
```

**Processing Flow:**
1. Look up job by ID
2. If found and not completed, remove from scheduler
3. Return success or 404 if not found

---

### 6. List Scheduled Jobs

#### **GET** `/schedule-query`

List all scheduled jobs with optional status filter.

**Query Parameters:**
| Field | Type | Optional | Values |
|-------|------|----------|--------|
| `status` | string | Yes | `scheduled`, `completed`, `failed`, `cancelled` |

**Response (200):**
```json
{
  "count": 3,
  "jobs": [
    {
      "job_id": "job_001_abc123def456",
      "status": "scheduled",
      "run_at": "2026-12-01T14:30:00",
      "question": "Question 1?"
    },
    {
      "job_id": "job_002_xyz789uvw012",
      "status": "completed",
      "run_at": "2026-11-22T10:00:00",
      "result": {...}
    }
  ]
}
```

**Example cURL:**
```bash
# List all jobs
curl http://localhost:8000/api/v1/schedule-query

# List only scheduled jobs
curl "http://localhost:8000/api/v1/schedule-query?status=scheduled"

# List only completed jobs
curl "http://localhost:8000/api/v1/schedule-query?status=completed"
```

---

### 7. Customer Support Endpoint

#### **POST** `/customer-support`

Specialized endpoint for customer support queries with priority tracking.

**Request Body:**
```json
{
  "customer_id": "CUST_12345",
  "message": "How do I reset my password?",
  "priority": "normal"
}
```

**Parameters:**
| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `customer_id` | string | ✓ | - | 1-255 chars |
| `message` | string | ✓ | - | 1-2000 chars, non-empty |
| `priority` | string | ✗ | normal | `low`, `normal`, `high`, `urgent` |

**Response (Success - 200):**
```json
{
  "customer_id": "CUST_12345",
  "query": "How do I reset my password?",
  "response": "To reset your password: 1. Go to login page...",
  "priority": "normal",
  "confidence": 0.892
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/customer-support \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_12345",
    "message": "I cannot log in",
    "priority": "high"
  }'
```

**Processing Flow:**
1. Validate customer_id and message
2. Route to customer_support_query method
3. Retrieve top-10 documents (default for support)
4. Generate contextual answer
5. Extract confidence from retrieval score
6. Return with priority metadata

**Confidence Metric:**
- Ranges from 0-1 (from first retrieved document)
- Higher = more relevant match found
- Use to trigger escalation if < 0.5

---

### 8. Document Ingestion Endpoint

#### **POST** `/ingest`

Upload and ingest documents into the vector database.

**Request Body:**
```json
{
  "file_paths": [
    "/path/to/document1.pdf",
    "/path/to/document2.md",
    "/path/to/document3.txt"
  ]
}
```

**Parameters:**
| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `file_paths` | array[string] | ✓ | 1-100 items, max 500 chars each |

**Supported Formats:**
- `.pdf` - PDF documents
- `.md` - Markdown files
- `.txt` - Plain text files
- `.html` - HTML web content
- `.json` - JSON data

**Response (Success - 200):**
```json
{
  "documents_ingested": 3,
  "chunks_created": 47,
  "total_in_db": 150
}
```

**Error Response (400):**
```json
{
  "detail": "Error loading /path/to/document.pdf: File not found"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/home/user/docs/guide.pdf",
      "/home/user/docs/faq.md"
    ]
  }'
```

**Processing Flow:**
1. Validate file paths (1-100 items)
2. Load documents using DocumentLoader
3. Split into chunks (semantic or fixed strategy)
4. Generate embeddings for each chunk
5. Add to Chroma vector store
6. Return ingestion statistics

**Limitations:**
- Max file size: 50MB per document
- Supported formats only (see above)
- Files must be accessible by the API server

**Performance:**
- ~100-200 chunks per document (depends on content)
- Embedding generation: ~5-10 sec for 100 chunks

---

### 9. Collection Statistics Endpoint

#### **GET** `/stats`

Get vector database collection statistics.

**Response (Success - 200):**
```json
{
  "collection_name": "rag_collection",
  "document_count": 150,
  "chunk_count": 523,
  "avg_chunk_size": 512,
  "total_tokens": 42500,
  "last_updated": "2026-11-22T10:30:00"
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/stats
```

**Returns:**
- Total documents ingested
- Number of chunks in database
- Average chunk size
- Approximate token count
- Last update timestamp

---

### 10. Database Health Check Endpoint

#### **GET** `/health-db`

Check database connectivity and status.

**Response (MySQL Connected - 200):**
```json
{
  "status": "connected",
  "database": "mysql",
  "host": "localhost",
  "port": 3306,
  "database_name": "rag_lines",
  "message": "MySQL connection successful ✅"
}
```

**Response (Fallback Mode - 200):**
```json
{
  "status": "fallback",
  "database": "JSON (fallback mode)",
  "message": "Using JSON file storage instead of database"
}
```

**Response (Connection Error - 500):**
```json
{
  "status": "error",
  "database": "mysql",
  "message": "Database error: Connection refused"
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/health-db
```

**Purpose:**
- Monitor persistent scheduler database
- Validate scheduled job storage
- Detect connection issues early

---

### 11. Health Check Endpoint

#### **GET** `/health`

Simple liveness probe for monitoring.

**Response (Success - 200):**
```json
{
  "status": "healthy",
  "service": "rag-lines-api",
  "version": "1.2.0"
}
```

**Example cURL:**
```bash
curl http://localhost:8000/health
```

**Use Cases:**
- Kubernetes liveness probe
- Docker healthcheck
- Load balancer monitoring
- Uptime monitoring

---

## 📊 Core Components

### 1. Document Loader (`src/ingestion/document_loader.py`)

**Purpose:** Load and parse multiple document formats

**Key Methods:**
- `load_documents(path: str) → List[Document]` - Load from directory or single file
- `_load_file(file_path: str) → Document` - Load single file
- `_load_pdf(file_path: str)` - Extract text from PDF
- `_load_markdown(file_path: str)` - Parse Markdown
- `_load_text(file_path: str)` - Read plain text
- `_load_html(file_path: str)` - Extract from HTML
- `_load_json(file_path: str)` - Parse JSON

**Configuration:**
- Max file size: 50MB (configurable)
- Supported formats: PDF, MD, TXT, HTML, JSON
- Logging: All operations logged via Loguru

**Example Usage:**
```python
from src.ingestion.document_loader import DocumentLoader

loader = DocumentLoader()
docs = loader.load_documents("/path/to/documents")
print(f"Loaded {len(docs)} documents")
```

---

### 2. Semantic Chunker (`src/ingestion/chunker.py`)

**Purpose:** Split documents into semantic chunks

**Strategies:**
- **Fixed**: Split by token count with overlap
- **Semantic**: Split at semantic boundaries (paragraphs, sentences)

**Configuration:**
- Chunk size: 512 tokens (default)
- Overlap: 50 tokens
- Strategy: "semantic" or "fixed"

**Output Structure:**
```python
class Chunk:
    chunk_id: str           # Unique identifier
    text: str               # Chunk content
    source: str             # Document source
    metadata: Dict          # {source, page, chunk_index, etc}
```

**Example:**
```python
from src.ingestion.chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk(documents)
print(f"Created {len(chunks)} chunks")
```

---

### 3. Embedding Model (`src/embeddings/embedding_model.py`)

**Purpose:** Generate vector embeddings for text

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight (22M parameters)
- Fast inference (~100k tokens/sec)
- 384-dimensional embeddings

**Key Methods:**
- `embed(texts: List[str]) → np.ndarray` - Batch embedding
- `embed_single(text: str) → np.ndarray` - Single text embedding
- `get_device()` - Return device (CPU/CUDA)

**Example:**
```python
from src.embeddings.embedding_model import EmbeddingModel

embedder = EmbeddingModel()
embeddings = embedder.embed(["text1", "text2", "text3"])
print(embeddings.shape)  # (3, 384)
```

---

### 4. Chroma Vector Store (`src/vectorstore/chroma_db.py`)

**Purpose:** Persistent semantic search storage

**Features:**
- In-memory + disk persistence
- Cosine similarity search
- Metadata filtering
- Collection management

**Key Methods:**
- `add_documents(chunks: List[Chunk])` - Add to index
- `search(query: str, top_k: int) → List[Tuple]` - Semantic search
- `count_documents() → int` - Get collection size
- `get_collection_stats() → Dict` - Statistics

**Database Location:**
```
./data/chroma_db/
```

**Example:**
```python
from src.vectorstore.chroma_db import ChromaVectorStore

vectorstore = ChromaVectorStore()
results = vectorstore.search("customer support", top_k=10)
print(f"Found {len(results)} results")
```

---

### 5. Retriever (`src/retrieval/retriever.py`)

**Purpose:** Hybrid retrieval combining semantic + BM25 search

**Hybrid Strategy:**
- 70% weight: Semantic (vector similarity)
- 30% weight: BM25 (lexical relevance)

**Key Methods:**
- `retrieve(query: str, top_k: int) → List[Dict]` - Hybrid search
- `_bm25_search(query: str, top_k: int)` - Lexical search
- `_combine_results(semantic, bm25)` - Score fusion

**Output Structure:**
```python
{
    "document": "text content",
    "score": 0.85,                    # Final score (0-1)
    "semantic_score": 0.9,            # Vector similarity
    "bm25_score": 0.5,                # Lexical overlap
    "metadata": {"source": "...", "page": 1}
}
```

**Example:**
```python
from src.retrieval.retriever import Retriever

retriever = Retriever()
results = retriever.retrieve("How do I?", top_k=5)
for result in results:
    print(f"Score: {result['score']:.3f} - {result['document'][:100]}")
```

---

### 6. Reranker (`src/retrieval/reranker.py`)

**Purpose:** Rerank retrieval results using cross-encoder

**Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Trained on MS MARCO dataset
- Optimized for relevance ranking
- More accurate than semantic similarity alone

**Key Methods:**
- `rerank(query: str, documents: List[Dict], top_k: int)` - Rerank results

**Example:**
```python
from src.retrieval.reranker import Reranker

reranker = Reranker()
reranked = reranker.rerank(query, results, top_k=5)
```

---

### 7. Local LLM (`src/llm/local_llm.py`)

**Purpose:** Interface with local LLM via Ollama or LM Studio

**Supported Models:**
- Ollama: Mistral, Llama 2, Neural Chat, etc.
- LM Studio: Any GGML-quantized model
- vLLM: OpenAI-compatible models

**Configuration:**
- Model: `mistral-7b-instruct-v0.1` (default)
- Base URL: `http://localhost:11434/api/generate` (Ollama)
- Temperature: 0.3 (deterministic)
- Max tokens: 1024

**Key Methods:**
- `generate(prompt: str, temperature: float = None) → str` - Generate text
- `_create_session()` - Connection pooling + retries

**Example:**
```python
from src.llm.local_llm import LocalLLM

llm = LocalLLM()
response = llm.generate("What is RAG?")
print(response)
```

**Retry Logic:**
- Max 3 retries with exponential backoff
- Handles: 429, 500, 502, 503, 504 errors
- Connection timeout: 60 seconds

---

### 8. RAG Pipeline (`src/qa/qa_pipeline.py`)

**Purpose:** Orchestrate end-to-end Q&A pipeline

**Processing Steps:**
1. Retrieve documents (hybrid search)
2. Rerank if enabled (top-5)
3. Build context string
4. Generate prompt
5. Call LLM for answer
6. Format response

**Key Methods:**
- `query(question: str, top_k: int, include_context: bool) → Dict` - Main query method
- `customer_support_query(message: str) → Dict` - Support-specific routing
- `_build_context(docs: List[Dict]) → str` - Format context
- `_build_prompt(question: str, context: str) → str` - Prompt engineering

**Example:**
```python
from src.qa.qa_pipeline import RAGPipeline

rag = RAGPipeline()
result = rag.query("What is customer support?", top_k=10)
print(result["answer"])
```

**Response Structure:**
```python
{
    "question": "...",
    "answer": "...",
    "retrieved_documents": [...],
    "context_used": "...",
    "document_count": 5
}
```

---

### 9. Persistent Scheduler (`src/api/persistent_scheduler.py`)

**Purpose:** Persistent job scheduling with database backend

**Features:**
- MySQL/JSON fallback storage
- Job status tracking (scheduled/completed/failed)
- Job retrieval and cancellation
- Automatic cleanup of old jobs

**Configuration:**
- DB Type: MySQL (or JSON fallback)
- DB Host: localhost
- DB Port: 3306
- DB Name: rag_lines

**Key Methods:**
- `schedule_once(run_at: datetime, func, payload)` - Schedule one-off job
- `get_job(job_id: str) → Dict` - Get job status
- `cancel_job(job_id: str) → bool` - Cancel scheduled job
- `list_jobs(status: str = None)` - List all jobs

**Example:**
```python
from src.api.persistent_scheduler import scheduler
from datetime import datetime, timedelta

future_time = datetime.now() + timedelta(hours=1)
job_id = scheduler.schedule_once(future_time, my_func, payload)
print(f"Scheduled job: {job_id}")
```

---

### 10. FastAPI Application (`src/api/app.py`)

**Purpose:** REST API server with rate limiting and logging

**Middleware:**
- **CORS**: Allow localhost:8501 (Streamlit) and localhost:3000
- **Rate Limiting**: 100 requests/minute per IP
- **Request Logging**: Performance metrics per endpoint

**Configuration:**
- Host: 0.0.0.0
- Port: 8000
- Workers: 4 (default via Uvicorn)

**Example:**
```python
from src.api.app import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 11. Streamlit UI (`src/ui/streamlit_app.py`)

**Purpose:** Web interface for RAG system

**Tabs:**
1. **Single Query**: One question at a time
2. **Batch Query**: Multiple questions
3. **Schedule Query**: Delayed execution

**Features:**
- Real-time streaming of responses
- Document retrieval display
- Scheduling calendar picker
- Job status monitoring

**Launch:**
```bash
streamlit run src/ui/streamlit_app.py
```

**Default URL:** `http://localhost:8501`

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# LLM Configuration
RAG_LLM_MODEL=mistral-7b-instruct-v0.1
RAG_LLM_BASE_URL=http://localhost:11434/api/generate
RAG_LLM_TEMPERATURE=0.3
RAG_LLM_MAX_TOKENS=1024

# Embedding Configuration
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_EMBEDDING_DEVICE=cpu

# Retrieval Configuration
RAG_RETRIEVAL_TOP_K=20
RAG_USE_RERANKING=true

# Vector Store
RAG_VECTORSTORE_PATH=./data/chroma_db

# API Configuration
RAG_API_HOST=0.0.0.0
RAG_API_PORT=8000
RAG_API_RATE_LIMIT_PER_MINUTE=100

# Database Configuration
RAG_DB_TYPE=mysql
RAG_DB_HOST=localhost
RAG_DB_PORT=3306
RAG_DB_USER=root
RAG_DB_PASSWORD=password
RAG_DB_NAME=rag_lines

# Logging
RAG_LOG_LEVEL=INFO
RAG_LOG_FILE=./logs/rag-system.log
```

### Configuration Management (`config/config.py`)

```python
from config.config import get_config

config = get_config()
print(config.llm_model)
print(config.api_port)
```

**Key Settings:**
- LLM model and URL
- Embedding model
- Chunk size (512 tokens)
- Retrieval top-K (20 documents)
- Reranking enabled/disabled
- Database connection
- API rate limits

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/SAIRAMPATEL123/rag-lines.git
cd rag-lines
git checkout rag-development

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Python Version:** 3.11+ (3.13.3 tested)

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your settings (optional, defaults work)
```

### 3. Start Ollama

```bash
# Install Ollama from https://ollama.ai
ollama pull mistral
ollama serve
```

Ollama will start on `http://localhost:11434`

### 4. Ingest Documents

**Option A: CLI**
```bash
python main.py ingest ./path/to/documents
```

**Option B: API**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["/path/to/file.pdf"]}'
```

**Option C: Python**
```python
from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.chroma_db import ChromaVectorStore
from src.ingestion.chunker import SemanticChunker

loader = DocumentLoader()
docs = loader.load_documents("./documents")
chunker = SemanticChunker()
chunks = chunker.chunk(docs)
vectorstore = ChromaVectorStore()
vectorstore.add_documents(chunks)
```

### 5. Run Q&A

**Option A: CLI**
```bash
python main.py qa
# Interactive prompt: Type questions and get answers
```

**Option B: API**
```bash
# Start API server
python main.py api

# Query in another terminal
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is support?"}'
```

**Option C: Web UI**
```bash
# Terminal 1: Start API
python main.py api

# Terminal 2: Start Streamlit
streamlit run src/ui/streamlit_app.py
# Visit http://localhost:8501
```

### 6. One-Command Launch

```bash
./scripts/start_app.sh
```

This starts:
- Ollama server
- FastAPI backend (port 8000)
- Streamlit UI (port 8501)

---

## 📊 Features Matrix

### Implemented Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Multi-format ingestion** | ✅ | PDF, MD, TXT, HTML, JSON |
| **Semantic chunking** | ✅ | Paragraph-aware splitting |
| **Hybrid retrieval** | ✅ | Semantic + BM25 combining |
| **Reranking** | ✅ | Cross-encoder refinement |
| **Local LLM** | ✅ | Ollama / LM Studio support |
| **FastAPI endpoints** | ✅ | 11 RESTful endpoints |
| **Batch queries** | ✅ | Concurrent processing (4 workers) |
| **Scheduled queries** | ✅ | One-off job scheduling |
| **Streamlit UI** | ✅ | Web interface |
| **Database health** | ✅ | MySQL connection check |
| **Rate limiting** | ✅ | 100 req/min per IP |
| **Logging** | ✅ | Structured logs via Loguru |
| **Request/response timing** | ✅ | Performance metrics |

### Planned Features

| Feature | Priority | Timeline |
|---------|----------|----------|
| Persistent job store | High | Q1 2026 |
| Cron-style jobs | High | Q1 2026 |
| Authentication & RBAC | High | Q2 2026 |
| Observability dashboards | Medium | Q2 2026 |
| Production queue (Celery/RQ) | Medium | Q2 2026 |
| Advanced UI (history, citations) | Medium | Q2 2026 |
| Multi-model support | Low | Q3 2026 |
| Fine-tuning pipeline | Low | Q3 2026 |

---

## 🌐 Deployment

### Local Development

```bash
# Start all services
./scripts/start_app.sh

# Access points:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - UI: http://localhost:8501
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "main.py", "api"]
```

```bash
docker build -t rag-lines .
docker run -p 8000:8000 rag-lines
```

### Production Deployment

**Recommended Stack:**
- **Reverse Proxy**: Nginx
- **ASGI Server**: Gunicorn + Uvicorn workers
- **Process Manager**: Systemd or Supervisor
- **Database**: MySQL 8.0+
- **Job Queue**: Redis + Celery (future)

**Example Systemd Service:**
```ini
[Unit]
Description=RAG-Lines API
After=network.target

[Service]
Type=notify
User=appuser
WorkingDirectory=/opt/rag-lines
Environment="PATH=/opt/rag-lines/venv/bin"
ExecStart=/opt/rag-lines/venv/bin/gunicorn \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000 \
  src.api.app:create_app()

[Install]
WantedBy=multi-user.target
```

---

## 🔍 Troubleshooting

### Issue: "No documents found" error

**Symptom:** Queries return warning message about no documents

**Solution:**
1. Check if documents ingested: `GET /stats`
2. Verify document paths exist
3. Supported formats: .pdf, .md, .txt, .html, .json
4. Ingest documents: `python main.py ingest ./path`

---

### Issue: Slow query responses (>10 seconds)

**Causes:**
- Large LLM model (use Mistral 7B instead of larger models)
- CPU inference (use GPU if available)
- Large retrieval top_k (reduce from 20 to 10)
- Many chunks in database (optimize chunking)

**Solutions:**
```bash
# Use faster model
RAG_LLM_MODEL=mistral-7b-instruct-v0.1 python main.py api

# Reduce retrieval size
# Edit config.py: retrieval_top_k = 10

# Use GPU
RAG_EMBEDDING_DEVICE=cuda python main.py api
```

---

### Issue: Ollama connection refused

**Symptom:** `ConnectionError: http://localhost:11434/api/generate`

**Solution:**
1. Verify Ollama running: `ps aux | grep ollama`
2. Start Ollama: `ollama serve`
3. Check port: `lsof -i :11434`
4. Verify model: `ollama list`

---

### Issue: Out of memory during embedding

**Symptom:** `MemoryError` during document ingestion

**Solution:**
1. Reduce batch size in config: `embedding_batch_size = 8`
2. Reduce chunk size: `chunk_size = 256`
3. Ingest in smaller batches
4. Add swap space (temporary)

---

### Issue: "Job not found" in scheduler

**Symptom:** `Detail: Job {job_id} not found` when retrieving scheduled job

**Causes:**
- Job ID typo
- Job expired/cleaned up
- Server restart (in-memory mode)

**Solutions:**
1. Verify job_id from creation response
2. Check job list: `GET /schedule-query`
3. Use persistent scheduler (MySQL backend)

---

### Issue: MySQL connection failed

**Symptom:** Database error when checking health

**Solution:**
1. Verify MySQL running: `mysql -u root -p`
2. Create database: `CREATE DATABASE rag_lines;`
3. Update credentials in `.env`
4. Fallback to JSON: Set `RAG_DB_TYPE=json`

---

### Issue: Rate limit exceeded (429 error)

**Symptom:** "Too many requests. Max 100 requests per minute."

**Solution:**
1. Reduce request rate
2. Increase limit in config: `api_rate_limit_per_minute = 200`
3. Use batch endpoint for multiple queries

---

### Issue: "Invalid ISO format datetime"

**Symptom:** Schedule query fails with datetime error

**Solution:**
Use ISO 8601 format: `2026-12-01T14:30:00`

**Examples:**
```json
{
  "run_at_iso": "2026-11-23T09:00:00"      // ✅ Correct
}

{
  "run_at_iso": "11/23/2026 09:00:00"      // ❌ Wrong format
}

{
  "run_at_iso": "2026-11-23 09:00:00"      // ❌ Missing T
}
```

---

## 📚 Additional Resources

### Documentation Files
- `README.md` - Quick start guide
- `DATABASE_SETUP.md` - MySQL setup
- `TESTING_GUIDE.md` - Test documentation
- `docs/PROJECT_ARCHITECTURE.md` - Architecture notes

### Key Files Reference
- API Routes: `src/api/routes.py`
- FastAPI App: `src/api/app.py`
- Configuration: `config/config.py`
- Main CLI: `main.py`

### External Resources
- Ollama: https://ollama.ai
- ChromaDB: https://www.trychroma.com
- FastAPI: https://fastapi.tiangolo.com
- Streamlit: https://streamlit.io

---

## 📞 Support & Contact

### Reporting Issues
Create GitHub issues with:
- Detailed error message
- Steps to reproduce
- Configuration used
- Log output

### Contributing
1. Fork repository
2. Create feature branch from `rag-development`
3. Submit pull request with tests
4. Update this documentation

---

**End of Documentation**

*Last Updated: 2026-11-22*  
*Version: 1.2.0*  
*Maintainer: Sairam Patel (@SAIRAMPATEL123)*
