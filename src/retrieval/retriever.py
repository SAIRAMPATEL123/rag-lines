from typing import List, Dict, Tuple
from loguru import logger
from config.config import get_config
from src.vectorstore.chroma_db import ChromaVectorStore
from rank_bm25 import BM25Okapi
import numpy as np

class Retriever:
    """Hybrid retrieval: semantic + real BM25 + RRF"""

    def __init__(self, vectorstore: ChromaVectorStore = None):
        self.config = get_config()
        self.vectorstore = vectorstore or ChromaVectorStore()
        self.top_k = self.config.retrieval_top_k
        self.bm25_index = None
        logger.info("Retriever initialized with real BM25 + RRF")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve using hybrid search: semantic + BM25 with RRF fusion"""
        top_k = top_k or self.top_k

        # Guard: return empty if no documents
        if self.vectorstore.count_documents() == 0:
            logger.warning("Vector store is empty — ingest documents first")
            return []

        # Semantic search
        semantic_results = self.vectorstore.search(query, top_k=top_k)

        # BM25 search (real lexical retrieval)
        bm25_results = self._bm25_search(query, top_k=top_k)

        # Combine with RRF (Reciprocal Rank Fusion)
        combined_results = self._rrf_combine(semantic_results, bm25_results, top_k=top_k)

        logger.info(f"Retrieval (semantic={len(semantic_results)}, bm25={len(bm25_results)}, fused={len(combined_results)})")
        return combined_results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Real BM25 lexical search using rank_bm25"""
        try:
            # Get all documents
            all_docs = self.vectorstore.collection.get()

            if not all_docs['documents']:
                return []

            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in all_docs['documents']]

            # Create BM25 index
            bm25 = BM25Okapi(tokenized_docs)

            # Search
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            # Pair scores with documents
            results = []
            for idx, score in enumerate(scores):
                if score > 0:  # Only keep positive scores
                    results.append((
                        all_docs['documents'][idx],
                        float(score),
                        all_docs['metadatas'][idx] if all_docs['metadatas'] else {}
                    ))

            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"BM25: {len(results)} docs with positive scores")
            return results[:top_k]

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _rrf_combine(self, semantic_results, bm25_results, top_k: int) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) to combine semantic and BM25"""
        # RRF formula: score = 1/(k + rank)  where k=60 (standard)
        rrf_k = 60
        combined = {}

        # Add semantic results with RRF scoring
        for rank, (doc, sem_score, metadata) in enumerate(semantic_results, 1):
            rrf_score = 1 / (rrf_k + rank)
            if doc not in combined:
                combined[doc] = {
                    "semantic_score": sem_score,
                    "bm25_score": 0,
                    "rrf_score": 0,
                    "metadata": metadata,
                    "doc_id": None
                }
            combined[doc]["semantic_score"] = sem_score
            combined[doc]["rrf_score"] += rrf_score

        # Add BM25 results with RRF scoring
        for rank, (doc, bm25_score, metadata) in enumerate(bm25_results, 1):
            rrf_score = 1 / (rrf_k + rank)
            if doc not in combined:
                combined[doc] = {
                    "semantic_score": 0,
                    "bm25_score": bm25_score,
                    "rrf_score": 0,
                    "metadata": metadata,
                    "doc_id": None
                }
            combined[doc]["bm25_score"] = bm25_score
            combined[doc]["rrf_score"] += rrf_score

        # Build final results sorted by RRF score
        final_results = []
        for doc, scores in combined.items():
            final_results.append({
                "document": doc,
                "score": scores["rrf_score"],  # RRF combined score
                "semantic_score": scores["semantic_score"],
                "bm25_score": scores["bm25_score"],
                "metadata": scores["metadata"]
            })

        # Sort by RRF score
        final_results.sort(key=lambda x: x["score"], reverse=True)

        # Filter by minimum threshold (RRF scores are 1/(k+rank) so very small)
        # Use a reasonable threshold for RRF
        threshold = 0.001  # ~1/(k+100) where k=60
        results_to_return = [r for r in final_results if r["score"] > threshold]

        logger.debug(f"RRF fused {len(final_results)} docs, returning {len(results_to_return)}")
        return results_to_return
