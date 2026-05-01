from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from src.qa.qa_pipeline import RAGPipeline
from src.vectorstore.chroma_db import ChromaVectorStore

router = APIRouter()
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    """Query request model"""
    question: str
    top_k: int = 20
    include_context: bool = False

class QueryResponse(BaseModel):
    """Query response model"""
    question: str
    answer: str
    document_count: int
    score: float = 0.0

class CustomerSupportRequest(BaseModel):
    """Customer support query model"""
    customer_id: str
    message: str
    priority: str = "normal"

@router.post("/query")
async def query(request: QueryRequest):
    """Execute Q&A query"""
    try:
        result = rag_pipeline.query(
            request.question,
            top_k=request.top_k,
            include_context=request.include_context
        )
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/customer-support")
async def customer_support(request: CustomerSupportRequest):
    """Handle customer support query"""
    try:
        result = rag_pipeline.customer_support_query(request.message)
        return {
            "customer_id": request.customer_id,
            "query": request.message,
            "response": result["answer"],
            "priority": request.priority,
            "confidence": result.get("retrieved_documents", [{}])[0].get("score", 0)
        }
    except Exception as e:
        logger.error(f"Customer support error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        vectorstore = ChromaVectorStore()
        stats = vectorstore.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
