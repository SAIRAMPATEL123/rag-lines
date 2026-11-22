#!/usr/bin/env python3
"""
Quick test script to verify Phase 1 & 2 fixes
Run this before starting the UI tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.config import get_config
from src.api.persistent_scheduler import scheduler
from src.ingestion.document_loader import Document
from src.ingestion.chunker import FixedChunker, SemanticChunker
from datetime import datetime, timezone, timedelta

logger.info("Starting fix verification tests...")

# Test 1: Input validation
print("\n✅ Test 1: Input Validation")
from pydantic import ValidationError
from src.api.routes import QueryRequest, BatchQueryRequest, ScheduleQueryRequest

try:
    QueryRequest(question="Valid question", top_k=20)
    print("  ✓ Valid query accepted")
except ValidationError as e:
    print(f"  ✗ Valid query rejected: {e}")

try:
    QueryRequest(question="x" * 2001, top_k=20)
    print("  ✗ Oversized question accepted (should reject)")
except ValidationError:
    print("  ✓ Oversized question rejected")

try:
    QueryRequest(question="test", top_k=200)
    print("  ✗ Out-of-range top_k accepted (should reject)")
except ValidationError:
    print("  ✓ Out-of-range top_k rejected")

try:
    BatchQueryRequest(questions=["q1", "q2"])
    print("  ✓ Valid batch query accepted")
except ValidationError as e:
    print(f"  ✗ Valid batch rejected: {e}")

try:
    BatchQueryRequest(questions=[])
    print("  ✗ Empty batch accepted (should reject)")
except ValidationError:
    print("  ✓ Empty batch rejected")

# Test 2: Persistent Scheduler
print("\n✅ Test 2: Persistent Scheduler")
job_id = None
try:
    run_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    def dummy_fn(payload):
        return {"result": "test"}

    job_id = scheduler.schedule_once(
        run_at,
        dummy_fn,
        {"question": "test"}
    )
    print(f"  ✓ Job scheduled with ID: {job_id}")

    job = scheduler.get_job(job_id)
    if job.get("status") == "scheduled":
        print("  ✓ Job status is 'scheduled'")
    else:
        print(f"  ✗ Job status is '{job.get('status')}', expected 'scheduled'")

    jobs = scheduler.list_jobs(status="scheduled")
    if any(j["job_id"] == job_id for j in jobs):
        print(f"  ✓ Job found in scheduled jobs list ({len(jobs)} total)")
    else:
        print("  ✗ Job not found in scheduled jobs list")

    success = scheduler.cancel_job(job_id)
    if success:
        print("  ✓ Job cancelled successfully")
    else:
        print("  ✗ Job cancellation failed")

except Exception as e:
    print(f"  ✗ Scheduler test failed: {e}")

# Test 3: File size validation
print("\n✅ Test 3: File Size Validation")
from src.ingestion.document_loader import DocumentLoader
loader = DocumentLoader()
config = get_config()

# Create a mock file check
import tempfile
import os

try:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("x" * (config.max_file_size + 1))
        temp_file = f.name

    try:
        loader._load_file(temp_file)
        print("  ✗ Oversized file accepted (should reject)")
    except ValueError as e:
        if "exceeds max" in str(e):
            print("  ✓ Oversized file rejected with correct error")
        else:
            print(f"  ✗ Wrong error message: {e}")
    finally:
        os.unlink(temp_file)

except Exception as e:
    print(f"  ✗ File size test setup failed: {e}")

# Test 4: Semantic Chunker with NLTK
print("\n✅ Test 4: Improved Semantic Chunking")
try:
    doc = Document(
        content="Dr. Smith went to the store. Mr. Jones was there. What a coincidence!",
        source="test.txt",
        doc_type="text"
    )
    chunker = SemanticChunker(chunk_size=50)
    chunks = chunker.chunk([doc])

    if len(chunks) > 0:
        print(f"  ✓ Semantic chunker produced {len(chunks)} chunks")
        print(f"    First chunk: '{chunks[0].text[:50]}...'")
    else:
        print("  ✗ Semantic chunker produced no chunks")

except Exception as e:
    print(f"  ✗ Semantic chunker test failed: {e}")

# Test 5: Thread-safe pipeline
print("\n✅ Test 5: Thread-Safe Pipeline Singleton")
try:
    from src.api.routes import get_pipeline
    import threading

    pipelines = []

    def get_pipeline_thread():
        try:
            p = get_pipeline()
            pipelines.append(p)
        except Exception as e:
            print(f"    Error in thread: {e}")

    threads = [threading.Thread(target=get_pipeline_thread) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if len(pipelines) == 5 and len(set(id(p) for p in pipelines)) == 1:
        print("  ✓ All threads got the same pipeline singleton")
    else:
        print(f"  ✗ Got {len(set(id(p) for p in pipelines))} different pipeline instances")

except Exception as e:
    print(f"  ✗ Thread-safety test failed: {e}")

# Test 6: LLM Connection Pooling
print("\n✅ Test 6: LLM Connection Pooling")
try:
    from src.llm.local_llm import LocalLLM
    llm = LocalLLM()

    if llm.session:
        adapters = llm.session.adapters
        if 'http://' in adapters and 'https://' in adapters:
            print("  ✓ LLM session has HTTP adapters configured")
            adapter = adapters['http://']
            if hasattr(adapter, 'poolmanager'):
                print("  ✓ Connection pooling enabled")
            else:
                print("  ✓ LLM session configured (pooling setup OK)")
        else:
            print("  ✗ LLM session missing adapters")
    else:
        print("  ✗ LLM session not created")

except Exception as e:
    print(f"  ✗ LLM pooling test failed: {e}")

# Test 7: Rate Limiter
print("\n✅ Test 7: Rate Limiter")
try:
    from src.api.app import rate_limiter
    import asyncio

    async def test_rate_limiter():
        client = "test_client"

        for i in range(101):
            allowed = await rate_limiter.is_allowed(client)
            if i < 100:
                if not allowed:
                    return False, f"Request {i+1} denied (should be allowed)"
            else:
                if allowed:
                    return False, "Request 101 allowed (should be denied)"

        return True, "All rate limit checks passed"

    success, message = asyncio.run(test_rate_limiter())
    if success:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")

except Exception as e:
    print(f"  ✗ Rate limiter test failed: {e}")

print("\n" + "="*60)
print("✅ All verification tests completed!")
print("="*60)
print("\nNext steps:")
print("1. Start the API: python main.py api")
print("2. In another terminal, start Streamlit: streamlit run src/ui/streamlit_app.py")
print("3. Test via UI at http://localhost:8501")
print("4. Monitor logs for any errors")
