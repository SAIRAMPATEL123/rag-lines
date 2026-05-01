from typing import List, Dict
from loguru import logger
from config.config import get_config

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("CrossEncoder not available, reranking disabled")

class Reranker:
    """Rerank retrieved documents for improved quality"""
    
    def __init__(self, model_name: str = None):
        self.config = get_config()
        self.model_name = model_name or self.config.reranker_model
        
        if RERANKER_AVAILABLE:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.available = True
        else:
            self.available = False
            logger.warning("Reranker not available")
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        """Rerank documents based on query relevance"""
        if not self.available or not documents:
            return documents
        
        top_k = top_k or len(documents)
        
        # Prepare pairs for reranking
        pairs = [[query, doc["document"]] for doc in documents]
        
        # Get reranker scores
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # Update documents with reranker scores
        for i, doc in enumerate(documents):
            doc["reranker_score"] = float(scores[i])
        
        # Sort by reranker score
        documents.sort(key=lambda x: x["reranker_score"], reverse=True)
        
        logger.debug(f"Reranked {len(documents)} documents")
        return documents[:top_k]
