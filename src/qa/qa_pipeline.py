from typing import Dict, List
from loguru import logger
from config.config import get_config
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.llm.local_llm import LocalLLM

class RAGPipeline:
    """End-to-end RAG pipeline: retrieve documents and generate answers"""
    
    def __init__(self, use_reranking: bool = None):
        self.config = get_config()
        self.retriever = Retriever()
        self.use_reranking = use_reranking if use_reranking is not None else self.config.use_reranking
        self.reranker = Reranker() if self.use_reranking else None
        self.llm = LocalLLM()
        
        logger.info("RAG Pipeline initialized")
    
    def query(self, question: str, top_k: int = None, include_context: bool = True) -> Dict:
        """Process a query through the RAG pipeline"""
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant documents
        top_k = top_k or self.config.retrieval_top_k
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Rerank if enabled
        if self.use_reranking and self.reranker:
            retrieved_docs = self.reranker.rerank(question, retrieved_docs, top_k=5)
        
        # Step 3: Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Step 4: Generate answer using LLM
        prompt = self._build_prompt(question, context)
        answer = self.llm.generate(prompt)
        
        # Step 5: Format response
        response = {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context_used": context if include_context else None,
            "document_count": len(retrieved_docs)
        }
        
        logger.info(f"Query processed successfully")
        return response
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            score = doc.get("score", 0)
            context_parts.append(
                f"[Document {i}] (Score: {score:.3f})\n"
                f"Source: {source}\n"
                f"Content: {doc['document'][:500]}...\n"
            )
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""Based on the following context, answer the question accurately and helpfully.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        return prompt
    
    def customer_support_query(self, customer_question: str) -> Dict:
        """Process a customer support query"""
        logger.info("Processing customer support query")
        return self.query(customer_question, top_k=10)
