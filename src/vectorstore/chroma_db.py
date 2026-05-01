from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
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
        
        # Initialize Chroma client
        logger.info(f"Initializing Chroma at {self.config.vectorstore_path}")
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.config.vectorstore_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Chroma collection '{collection_name}' ready")
    
    def add_documents(self, chunks: List[Chunk]) -> None:
        """Add chunks to vector store"""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract texts and generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed(texts)
        
        # Prepare metadata
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Add to Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
        logger.info(f"Successfully added {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents"""
        top_k = top_k or self.config.retrieval_top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_single(query)
        
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                # Convert distance to similarity (cosine similarity)
                similarity = 1 - distance
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                retrieved_docs.append((doc, similarity, metadata))
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query")
        return retrieved_docs
    
    def delete_collection(self) -> None:
        """Delete the collection"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")
    
    def count_documents(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "name": self.collection_name,
            "document_count": self.count_documents(),
            "embedding_dimension": self.embedding_model.get_embedding_dimension()
        }
