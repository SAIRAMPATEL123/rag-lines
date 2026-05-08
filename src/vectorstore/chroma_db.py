from typing import List, Dict, Any, Tuple
import chromadb
import os
import numpy as np
from loguru import logger
from config.config import get_config
from src.ingestion.chunker import Chunk
from src.embeddings.embedding_model import EmbeddingModel

class ChromaVectorStore:
    """Chroma vector database for semantic search"""
    
    def __init__(self, collection_name: str = "rag_collection"):
        self.config = get_config()
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel()
        
        # Use absolute path to avoid relative path issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        self.db_path = os.path.join(project_root, "data", "chroma_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        logger.info(f"Initializing Chroma at ABSOLUTE path: {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # List existing collections for debug
        existing = self.client.list_collections()
        logger.info(f"Existing collections in DB: {[c.name for c in existing]}")
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        doc_count = self.collection.count()
        logger.info(f"Collection '{collection_name}' ready — {doc_count} documents loaded")

    def add_documents(self, chunks: List[Chunk]) -> None:
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store at {self.db_path}")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed(texts)
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
        logger.info(f"Successfully added {len(chunks)} chunks — total now: {self.collection.count()}")

    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        top_k = top_k or self.config.retrieval_top_k
        actual_count = self.collection.count()
        
        logger.info(f"SEARCH called — collection has {actual_count} docs, requesting top {top_k}")
        
        if actual_count == 0:
            logger.warning("Collection is empty!")
            return []
        
        # CRITICAL: n_results cannot exceed actual document count
        n_results = min(top_k, actual_count)
        logger.info(f"Using n_results={n_results}")
        
        query_embedding = self.embedding_model.embed_single(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                retrieved_docs.append((doc, similarity, metadata))
        
        logger.info(f"Search complete — returned {len(retrieved_docs)} documents")
        return retrieved_docs

    def delete_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")

    def count_documents(self) -> int:
        count = self.collection.count()
        logger.info(f"count_documents() = {count}")
        return count

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "document_count": self.count_documents(),
            "db_path": self.db_path,
            "embedding_dimension": self.embedding_model.get_embedding_dimension()
        }