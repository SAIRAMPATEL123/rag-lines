from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from src.qa.qa_pipeline import RAGPipeline
from src.vectorstore.chroma_db import ChromaVectorStore
from src.api.scheduler import scheduler

router = APIRouter()
_rag_pipeline = None


def get_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


class QueryRequest(BaseModel):
    question: str
    top_k: int = 20
    include_context: bool = False


class BatchQueryRequest(BaseModel):
    questions: List[str]
    top_k: int = 20
    include_context: bool = False


class ScheduleQueryRequest(BaseModel):
    question: str
    run_at_iso: str
    top_k: int = 20
    include_context: bool = False


class CustomerSupportRequest(BaseModel):
    customer_id: str
    message: str
    priority: str = "normal"


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
        results = [
            get_pipeline().query(q, top_k=request.top_k, include_context=request.include_context)
            for q in request.questions
        ]
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


@router.get("/stats")
async def get_stats():
    try:
        vectorstore = ChromaVectorStore()
        return vectorstore.get_collection_stats()
    except Exception as e:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug")
async def debug_chromadb():
    """Debug endpoint — shows exact ChromaDB state"""
    import chromadb, os
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        db_path = os.path.join(project_root, "data", "chroma_db")
        
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        
        result = {
            "db_path": db_path,
            "db_path_exists": os.path.exists(db_path),
            "db_files": os.listdir(db_path) if os.path.exists(db_path) else [],
            "collections": []
        }
        
        for col in collections:
            c = client.get_collection(col.name)
            result["collections"].append({
                "name": col.name,
                "count": c.count()
            })
        
        return result
    except Exception as e:
        return {"error": str(e)}