# Phase 1 & 2 Fixes - Complete Testing Guide

## Overview

All Phase 1 (5 critical) and Phase 2 (4 production-ready) fixes have been implemented and compiled successfully. This guide walks through testing each fix.

---

## Pre-Testing Checklist

```bash
cd "C:\Users\sairam.bathula\Desktop\r coidel\rag - utra\rag-lines"

# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Copy environment file
cp .env.example .env

# 3. Make sure Ollama is running (for LLM tests)
# In a separate terminal: ollama serve

# 4. Ingest some documents first
python main.py ingest ./docs  # or your document directory
```

---

## Section 1: Phase 1 Critical Fixes Testing

### Fix 1: Hybrid Retrieval Method - FIXED

**What was broken:** The `retrieve()` method didn't call BM25 search or combine results.

**What was fixed:**
- Line 27-37 in `src/retrieval/retriever.py` now:
  1. Calls semantic search ✅
  2. Calls BM25 search ✅
  3. Combines results with weighted scoring (0.7 semantic + 0.3 BM25) ✅

**How to verify via code:**
```python
from src.retrieval.retriever import Retriever

retriever = Retriever()
results = retriever.retrieve("What is machine learning?", top_k=5)

# Should have results from BOTH semantic and BM25
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"  Semantic: {result.get('semantic_score', 0):.3f}")
    print(f"  BM25: {result.get('bm25_score', 0):.3f}")
    print(f"  Doc: {result['document'][:100]}...")
```

**Expected output:**
```
Score: 0.755
  Semantic: 0.750
  BM25: 0.800
  Doc: Machine learning is a subset of artificial intelligence...
```

---

### Fix 2: Input Validation - FIXED

**What was missing:** No validation on user input (question length, top_k range, etc.)

**What was added:**
- QueryRequest: `question` (1-2000 chars), `top_k` (1-100)
- BatchQueryRequest: `questions` (1-100 items), each 1-2000 chars
- ScheduleQueryRequest: validates ISO datetime format
- CustomerSupportRequest: validates priority enum
- All sanitized with `.strip()`

**Test in terminal:**

```bash
# 1. Valid request - should work
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AI?", "top_k":20, "include_context":false}'

# Expected: 200 OK with answer

# 2. Question too long - should fail with 422
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"'$(python -c "print('x'*2001)")'", "top_k":20}'

# Expected: 422 Unprocessable Entity - "ensure this value has at most 2000 characters"

# 3. top_k out of range - should fail with 422
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AI?", "top_k":200}'

# Expected: 422 Unprocessable Entity - "ensure this value is less than or equal to 100"

# 4. Empty question - should fail with 422
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"   ", "top_k":20}'

# Expected: 422 Unprocessable Entity - custom validator error
```

**Or test programmatically:**
```python
from src.api.routes import QueryRequest
from pydantic import ValidationError

# Valid
req = QueryRequest(question="What is AI?", top_k=20)
print("✓ Valid request accepted")

# Invalid: too long
try:
    req = QueryRequest(question="x"*2001, top_k=20)
except ValidationError as e:
    print("✓ Long question rejected:", e.errors()[0]['msg'])

# Invalid: top_k out of range
try:
    req = QueryRequest(question="test", top_k=200)
except ValidationError as e:
    print("✓ High top_k rejected:", e.errors()[0]['msg'])
```

---

### Fix 3: Rate Limiting - FIXED

**What was missing:** No rate limiting; API vulnerable to DoS abuse.

**What was added:**
- RateLimiter class in `src/api/app.py` (lines 13-33)
- Middleware that checks rate limit per IP (lines 57-67)
- Limit: 100 requests per minute per IP
- Returns 429 (Too Many Requests) when exceeded

**How to test:**

```bash
# Start the API
python main.py api

# In another terminal, send 101 requests rapidly
for i in {1..101}; do
  response=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:8000/api/v1/query \
    -H "Content-Type: application/json" \
    -d '{"question":"test query", "top_k":5}')
  
  if [ "$response" = "429" ]; then
    echo "Request $i: $response ✓ (rate limited)"
    break
  else
    echo "Request $i: $response"
  fi
done

# Expected: First 100 requests get 200, request 101 gets 429
```

**Logs will show:**
```
Rate limit exceeded for 127.0.0.1
```

---

### Fix 4: File Size Validation - FIXED

**What was missing:** No check on file size before loading (config had max_file_size=50MB but never enforced).

**What was added:**
- `_load_file()` now checks file size before loading (lines 41-46 in document_loader.py)
- Raises ValueError with clear message if file > 50MB
- Prevents memory exhaustion from large uploads

**How to test:**

```bash
# Create a 51MB test file
python << 'EOF'
with open("test_large_file.txt", "wb") as f:
    f.write(b"x" * (51 * 1024 * 1024))  # 51MB
EOF

# Try to ingest it
python main.py ingest ./test_large_file.txt

# Expected error:
# Error loading ./test_large_file.txt: 
# File .../test_large_file.txt is 51.0MB, exceeds max 50.0MB

# Clean up
rm test_large_file.txt
```

---

### Fix 5: Removed /debug Endpoint - FIXED

**What was wrong:** Endpoint at `GET /api/v1/debug` exposed:
- Database path
- All collection names
- Document counts
- Raw ChromaDB internals

**What was fixed:**
- Entire `/debug` endpoint removed from `src/api/routes.py` (was lines 126-154)

**How to verify:**

```bash
# This should now return 404
curl http://localhost:8000/api/v1/debug

# Expected: 404 Not Found
```

---

## Section 2: Phase 2 Production-Ready Fixes Testing

### Fix 6: Persistent Job Store - FIXED

**What was broken:** Scheduled jobs only in memory; restart = lost jobs.

**What was fixed:**
- New `PersistentScheduler` in `src/api/persistent_scheduler.py`
- Jobs saved to JSON in `./data/scheduled_jobs/`
- Auto-loads and reschedules pending jobs on startup
- New endpoints to list and cancel jobs

**How to test:**

```bash
# 1. Schedule a job 5 seconds from now
curl -X POST http://localhost:8000/api/v1/schedule-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "run_at_iso": "'$(python -c "from datetime import datetime, timezone, timedelta; print((datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat())")'",
    "top_k": 10,
    "include_context": false
  }'

# Expected response:
# {
#   "job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "status": "scheduled",
#   "run_at": "2026-11-17T15:30:45.123456+00:00"
# }

# 2. List all scheduled jobs
curl http://localhost:8000/api/v1/schedule-query

# Expected: List with the job you just scheduled

# 3. Get specific job status
curl http://localhost:8000/api/v1/schedule-query/{job_id}

# Expected: Job details with status "scheduled"

# 4. Wait for job to execute (5 seconds)
sleep 6

# 5. Check job status again
curl http://localhost:8000/api/v1/schedule-query/{job_id}

# Expected: status changed to "completed" or "running"

# 6. Check job file on disk
ls -la ./data/scheduled_jobs/
cat ./data/scheduled_jobs/{job_id}.json

# Expected: JSON file exists with full job details and result

# 7. Cancel a job
job_id2=$(curl -s -X POST http://localhost:8000/api/v1/schedule-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Test cancellation",
    "run_at_iso": "'$(python -c "from datetime import datetime, timezone, timedelta; print((datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat())")'",
    "top_k": 10
  }' | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

curl -X DELETE http://localhost:8000/api/v1/schedule-query/$job_id2

# Expected: {"job_id": "...", "status": "cancelled"}

# 8. Test job persistence on restart
# a) Schedule another job
# b) Stop API (Ctrl+C)
# c) Restart API: python main.py api
# d) Check job still there: curl http://localhost:8000/api/v1/schedule-query
#    Expected: Job still shows as "scheduled" and will run at original time
```

**Log output to expect:**
```
Loaded 2 pending jobs
Rescheduled pending job xxx to run in 45.2s
```

---

### Fix 7: Async Batch Processing - FIXED

**What was slow:** Batch queries processed sequentially; 10 questions = 10× slower.

**What was fixed:**
- Batch query now uses ThreadPoolExecutor with max_workers=4 (lines 73-85 in routes.py)
- Processes up to 4 questions in parallel
- Much better performance

**How to test:**

```bash
# Time a batch of 5 queries
time curl -X POST http://localhost:8000/api/v1/batch-query \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is machine learning?",
      "What is deep learning?",
      "What is natural language processing?",
      "What is computer vision?",
      "What is reinforcement learning?"
    ],
    "top_k": 10,
    "include_context": false
  }'

# Compare to 5 sequential single queries:
time for i in {1..5}; do
  curl -s -X POST http://localhost:8000/api/v1/query \
    -H "Content-Type: application/json" \
    -d '{"question":"What is AI?", "top_k":10}'
done

# Expected:
# - Batch time: ~15-20s (4 questions process in parallel)
# - Sequential time: ~40-50s (5 questions one by one)
# - Improvement: 2-3x faster!
```

---

### Fix 8: Thread-Safe Pipeline Singleton - FIXED

**What was wrong:** Global `_rag_pipeline` without lock; race condition possible under high concurrency.

**What was fixed:**
- Added `_pipeline_lock = threading.Lock()` (line 14 in routes.py)
- Double-checked locking pattern in `get_pipeline()` (lines 18-20)
- Safe for multi-threaded FastAPI server

**How to verify (programmatically):**

```python
import threading
from src.api.routes import get_pipeline

pipelines = []

def get_pipeline_thread():
    p = get_pipeline()
    pipelines.append(p)

# Create 10 threads trying to get pipeline simultaneously
threads = [threading.Thread(target=get_pipeline_thread) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Check all threads got the SAME instance
ids = [id(p) for p in pipelines]
unique_ids = set(ids)

if len(unique_ids) == 1:
    print("✓ All threads got the same singleton (thread-safe!)")
else:
    print("✗ Different instances created (not thread-safe!)")
```

---

### Fix 9: LLM Connection Pooling & Retry Logic - FIXED

**What was wrong:**
- Timeout: 300s (wastes resources, poor UX)
- No connection reuse
- No retry logic
- Any Ollama hiccup = timeout

**What was fixed:**
- Timeout reduced to 60s (lines 48-50 in local_llm.py)
- HTTPAdapter with connection pooling (pool_connections=10, pool_maxsize=10)
- Retry strategy with exponential backoff for errors 429, 500, 502, 503, 504
- Up to 3 retry attempts with 1s, 2s, 4s delays

**How to test:**

```bash
# 1. Start API with normal Ollama
python main.py api

# 2. Make a query and watch logs
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AI?", "top_k":5}'

# Logs should show: "Generated 156 characters"

# 3. Test retry logic (advanced):
# a) Stop Ollama while API is running
# b) Make a query
# c) Watch logs for "cannot connect to LLM, retrying"
# d) Restart Ollama
# e) Query should succeed on retry

# 4. Check connection pooling:
# - Make 2-3 quick queries
# - Logs should show faster responses (connection reuse)
```

**Logs to expect:**
```
LocalLLM initialized with model: mistral
Generated 156 characters
Cannot connect to LLM, retrying in 1s (attempt 1/3)
LLM connection established (after retrying)
```

---

## Bonus Tests

### CORS Hardening

```bash
# This should fail (wrong origin)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Origin: http://example.com" \
  -d '{"question":"test", "top_k":5}'

# Expected: CORS error (no Access-Control-Allow-Origin header)

# This should work (allowed origin)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Origin: http://localhost:8501" \
  -d '{"question":"test", "top_k":5}'

# Expected: 200 OK with response
```

### Improved Sentence Splitting

```python
from src.ingestion.chunker import SemanticChunker
from src.ingestion.document_loader import Document

# Test abbreviations
doc = Document(
    content="Dr. Smith works at the company. Mr. Jones is the CEO. What a team!",
    source="test.txt",
    doc_type="text"
)

chunker = SemanticChunker()
chunks = chunker.chunk([doc])

# Should split into 3 sentences (not mangled by "Dr." or "Mr.")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  {i+1}: {chunk.text}")

# Expected: Correctly split sentences, not broken at abbreviations
```

---

## UI Testing (When Ready)

Once the API passes all above tests, start the UI:

```bash
# Terminal 1: API
python main.py api

# Terminal 2: Streamlit UI
streamlit run src/ui/streamlit_app.py

# Browser: http://localhost:8501
```

**UI tests to perform:**

1. **Single Query Tab:**
   - Ask a question
   - Verify answer appears in < 30s
   - Check "Include context" checkbox works

2. **Batch Query Tab:**
   - Enter 3-5 questions
   - Watch timer (should be ~parallel time, not 5× single query time)
   - Verify all answers appear

3. **Scheduled Query Tab:**
   - Schedule query for 1 minute from now
   - Auto-poll should show "scheduled", then "completed"
   - Job ID should persist

4. **Error Cases:**
   - Try question > 2000 chars (should show validation error)
   - Try batch with 0 questions (should show validation error)
   - Spam 101 requests/minute (should get 429 errors)

---

## Success Criteria

All Phase 1 fixes should:
- ✅ Input validation rejects invalid requests with 422
- ✅ Rate limiter returns 429 after 100 requests/minute
- ✅ File upload rejects >50MB with clear error
- ✅ /debug endpoint returns 404
- ✅ Hybrid retrieval returns combined semantic+BM25 results

All Phase 2 fixes should:
- ✅ Jobs persist in `./data/scheduled_jobs/*.json`
- ✅ Batch queries complete 2-3x faster
- ✅ Concurrent requests get same pipeline singleton
- ✅ LLM retries on connection errors
- ✅ Timeout reduced from 300s to 60s

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| 500 error on query | Check Ollama is running: `ollama serve` |
| 429 rate limit too soon | Increase `API_RATE_LIMIT_PER_MINUTE` in `.env` |
| Batch queries still slow | Check logs for parallel execution (should see 4 concurrent) |
| Job file not persisting | Check `./data/scheduled_jobs/` has write permissions |
| LLM timeout | Reduce model size or increase timeout in config |
| "Cannot connect to LLM" | Verify Ollama running at http://localhost:11434 |

---

## Next Steps

After successful UI testing:
1. Monitor logs for any errors
2. Load-test with many concurrent requests (rate limiter should handle)
3. Restart API and verify scheduled jobs resume
4. Consider Phase 3 improvements (database-backed scheduler, API keys, etc.)
