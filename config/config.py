import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Config(BaseSettings):
    """Application configuration management"""
    
    # LLM Configuration
    llm_model: str = "mistral"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.3
    llm_top_k: int = 40
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1024
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # cpu or cuda
    
    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "semantic"  # semantic or fixed
    
    # Retrieval Configuration
    retrieval_top_k: int = 20
    retrieval_score_threshold: float = 0.3
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Vector Store Configuration
    vectorstore_path: str = "./data/chroma_db"
    vectorstore_type: str = "chroma"
    
    # Document Processing
    document_storage_path: str = "./data/documents"
    max_file_size: int = 52428800  # 50MB
    supported_formats: list = [".pdf", ".md", ".txt", ".html", ".json"]
    
    # Database Configuration
    db_type: str = "sqlite"
    db_connection_string: str = "sqlite:///./data/app.db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/rag-system.log"
    
    # Evaluation
    evaluation_enabled: bool = True
    save_eval_metrics: bool = True
    metrics_output_path: str = "./data/metrics"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.vectorstore_path,
            self.document_storage_path,
            "./logs",
            self.metrics_output_path
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached config instance"""
    return Config()
