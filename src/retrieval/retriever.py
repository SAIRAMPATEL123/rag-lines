from typing import List, Dict, Tuple
from loguru import logger
from config.config import get_config
from src.vectorstore.chroma_db import ChromaVectorStore
import re
from collections import Counter

class Retriever:
    """Hybrid retrieval: semantic + BM25"""
    
    def __init__(self, vectorstore: ChromaVectorStore = None):
        self.config = get_config()
        self.vectorstore = vectorstore or ChromaVectorStore()
        self.top_k = self.config.retrieval_top_k
        logger.info("Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents using hybrid search"""
        top_k = top_k or self.top_k

        # Guard: return empty if collection has no documents yet
        if self.vectorstore.count_documents() == 0:
            logger.warning("Vector store is empty — ingest documents first: python main.py ingest <path>")
            return []

        # Semantic search
        semantic_results = self.vectorstore.search(query, top_k=top_k)
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Simple BM25 implementation using term frequency"""
        query_terms = set(query.lower().split())
        
        # Get all documents from collection
        results = self.vectorstore.collection.get()
        
        scores = []
        for i, doc_text in enumerate(results['documents']):
            doc_terms = set(doc_text.lower().split())
            # Simple overlap score
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / (len(query_terms) + len(doc_terms))
                scores.append((doc_text, score, results['ids'][i]))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(score[0], score[1]) for score in scores[:top_k]]
    
    def _combine_results(self, semantic_results, bm25_results) -> List[Dict]:
        """Combine semantic and BM25 results with weighted scoring"""
        combined = {}
        
        # Add semantic results (weight: 0.7)
        for doc, score, metadata in semantic_results:
            if doc not in combined:
                combined[doc] = {"semantic_score": 0, "bm25_score": 0, "metadata": metadata}
            combined[doc]["semantic_score"] = score
        
        # Add BM25 results (weight: 0.3)
        for doc, score in bm25_results:
            if doc not in combined:
                combined[doc] = {"semantic_score": 0, "bm25_score": 0, "metadata": {}}
            combined[doc]["bm25_score"] = score
        
        # Calculate final scores
        final_results = []
        for doc, scores in combined.items():
            final_score = (0.7 * scores["semantic_score"]) + (0.3 * scores["bm25_score"])
            if final_score >= self.config.retrieval_score_threshold:
                final_results.append({
                    "document": doc,
                    "score": final_score,
                    "semantic_score": scores["semantic_score"],
                    "bm25_score": scores["bm25_score"],
                    "metadata": scores["metadata"]
                })
        
        # Sort by final score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:self.top_k]
