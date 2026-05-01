from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from config.config import get_config

class EmbeddingModel:
    """Generate embeddings using local sentence transformers"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.config = get_config()
        self.model_name = model_name or self.config.embedding_model
        self.device = device or self.config.embedding_device
        
        logger.info(f"Loading embedding model: {self.model_name} on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Embedding model loaded successfully")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        if not texts:
            return np.array([])
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.embed([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Get embedding vector dimension"""
        return self.model.get_sentence_embedding_dimension()
