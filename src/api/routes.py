from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from loguru import logger
from src.qa.qa_pipeline import RAGPipeline
from src.vectorstore.chroma_db import ChromaVectorStore
from src.api.persistent_scheduler import scheduler
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker
import threading

router = APIRouter()
_rag_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        with _pipeline_lock:
            if _rag_pipeline is None:
                _rag_pipeline = RAGPipeline()
    return _rag_pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(20, ge=1, le=100)
    include_context: bool = False

    @validator('question')
    def sanitize_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty')
        return v


class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., min_items=1, max_items=100)
    top_k: int = Field(20, ge=1, le=100)
    include_context: bool = False

    @validator('questions', each_item=True)
    def validate_question_length(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Each question must be non-empty')
        if len(v) > 2000:
            raise ValueError('Each question must be <= 2000 characters')
        return v


class ScheduleQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    run_at_iso: str
    top_k: int = Field(20, ge=1, le=100)
    include_context: bool = False

    @validator('question')
    def sanitize_scheduled_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty')
        return v

    @validator('run_at_iso')
    def validate_iso_format(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError('Invalid ISO format datetime')
        return v


class CustomerSupportRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=2000)
    priority: str = Field("normal", pattern="^(low|normal|high|urgent)$")

    @validator('message')
    def sanitize_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Message cannot be empty')
        return v


class IngestRequest(BaseModel):
    file_paths: List[str] = Field(..., min_items=1, max_items=100)

    @validator('file_paths', each_item=True)
    def validate_file_path(cls, v):
        if not v or len(v) > 500:
            raise ValueError('Invalid file path')
        return v


@router.post("/query")
async def query(request: QueryRequest):
    try:
        logger.info(f"/query question_len={len(request.question)} top_k={request.top_k}")
        result = get_pipeline().query(
            request.question,
            top_k=request.top_k,
            include_context=request.include_context,
        )
        return result
    except Exception as e:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-query")
async def batch_query(request: BatchQueryRequest):
    try:
        logger.info(f"/batch-query count={len(request.questions)} top_k={request.top_k}")
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def process_query(q):
            return get_pipeline().query(q, top_k=request.top_k, include_context=request.include_context)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = await loop.run_in_executor(
                executor,
                lambda: [process_query(q) for q in request.questions]
            )
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.exception("Batch query error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule-query")
async def schedule_query(request: ScheduleQueryRequest):
    try:
        run_at = datetime.fromisoformat(request.run_at_iso)
        payload = request.model_dump()

        def _execute(p):
            return get_pipeline().query(
                p["question"],
                top_k=p["top_k"],
                include_context=p["include_context"],
            )

        job_id = scheduler.schedule_once(run_at, _execute, payload)
        return {"job_id": job_id, "status": "scheduled", "run_at": request.run_at_iso}
    except Exception as e:
        logger.exception("Schedule query error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule-query/{job_id}")
async def get_scheduled_job(job_id: str):
    return scheduler.get_job(job_id)


@router.delete("/schedule-query/{job_id}")
async def cancel_scheduled_job(job_id: str):
    """Cancel a scheduled job"""
    success = scheduler.cancel_job(job_id)
    if success:
        return {"job_id": job_id, "status": "cancelled"}
    else:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.get("/schedule-query")
async def list_scheduled_jobs(status: str = None):
    """List all scheduled jobs, optionally filtered by status"""
    jobs = scheduler.list_jobs(status=status)
    return {"count": len(jobs), "jobs": jobs}


@router.post("/customer-support")
async def customer_support(request: CustomerSupportRequest):
    try:
        result = get_pipeline().customer_support_query(request.message)
        return {
            "customer_id": request.customer_id,
            "query": request.message,
            "response": result["answer"],
            "priority": request.priority,
            "confidence": result.get("retrieved_documents", [{}])[0].get("score", 0),
        }
    except Exception as e:
        logger.exception("Customer support error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Ingest documents from uploaded files"""
    try:
        logger.info(f"Ingesting {len(request.file_paths)} files")

        loader = DocumentLoader()
        all_documents = []

        for file_path in request.file_paths:
            try:
                doc = loader._load_file(file_path)
                all_documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise HTTPException(status_code=400, detail=f"Error loading {file_path}: {str(e)}")

        if not all_documents:
            raise HTTPException(status_code=400, detail="No documents could be loaded")

        chunker = SemanticChunker()
        chunks = chunker.chunk(all_documents)

        vectorstore = ChromaVectorStore()
        vectorstore.add_documents(chunks)

        stats = vectorstore.get_collection_stats()

        logger.info(f"Ingestion complete: {len(all_documents)} docs, {len(chunks)} chunks")
        return {
            "documents_ingested": len(all_documents),
            "chunks_created": len(chunks),
            "total_in_db": stats.get("document_count", 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-db")
async def health_database():
    """Check database connectivity"""
    try:
        from src.api.persistent_scheduler import scheduler

        # Test database connection
        if scheduler.use_db:
            session = scheduler.Session()
            try:
                session.execute("SELECT 1")
                session.close()
                return {
                    "status": "connected",
                    "database": scheduler.config.db_type,
                    "host": scheduler.config.db_host,
                    "port": scheduler.config.db_port,
                    "database_name": scheduler.config.db_name,
                    "message": "MySQL connection successful ✅"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "database": scheduler.config.db_type,
                    "message": f"Database error: {str(e)} ❌"
                }
        else:
            return {
                "status": "fallback",
                "database": "JSON (fallback mode)",
                "message": "Using JSON file storage instead of database"
            }
    except Exception as e:
        logger.exception("Database health check error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    try:
        vectorstore = ChromaVectorStore()
        return vectorstore.get_collection_stats()
    except Exception as e:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(e))


