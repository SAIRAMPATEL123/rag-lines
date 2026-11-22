import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Config(BaseSettings):
    """Application configuration management with RAG_ prefix"""

    # LLM Configuration
    llm_model: str = Field(default="mistral-7b-instruct-v0.1", description="LLM model name")
    llm_base_url: str = Field(default="http://localhost:11434/api/generate", description="LLM server URL")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_top_k: int = Field(default=40, ge=1)
    llm_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1024, ge=128)

    # Embedding Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str = Field(default="cpu", pattern="^(cpu|cuda)$")

    # Chunking Configuration
    chunk_size: int = Field(default=512, ge=100)
    chunk_overlap: int = Field(default=50, ge=0)
    chunking_strategy: str = Field(default="semantic", pattern="^(semantic|fixed)$")

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    use_reranking: bool = Field(default=True)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Vector Store Configuration
    vectorstore_path: str = Field(default="./data/chroma_db")
    vectorstore_type: str = Field(default="chroma", pattern="^(chroma|qdrant|pgvector)$")

    # Document Processing
    document_storage_path: str = Field(default="./data/documents")
    max_file_size: int = Field(default=52428800, description="Max file size in bytes")
    supported_formats: list = Field(default_factory=lambda: [".pdf", ".md", ".txt", ".html", ".json"])

    # Database Configuration
    db_type: str = Field(default="mysql", pattern="^(mysql|postgresql)$")
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=3306, ge=1, le=65535)
    db_user: str = Field(default="root")
    db_password: str = Field(default="password")
    db_name: str = Field(default="rag_lines")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_debug: bool = Field(default=False)
    api_rate_limit_per_minute: int = Field(default=100, ge=10)
    scheduled_jobs_dir: str = Field(default="./data/scheduled_jobs")

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: str = Field(default="./logs/rag-system.log")

    # Evaluation
    evaluation_enabled: bool = Field(default=True)
    save_eval_metrics: bool = Field(default=True)
    metrics_output_path: str = Field(default="./data/metrics")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",
    )

    @field_validator('db_host')
    @classmethod
    def validate_db_host(cls, v):
        if not v or len(v) == 0:
            raise ValueError('db_host cannot be empty')
        return v

    @field_validator('db_user')
    @classmethod
    def validate_db_user(cls, v):
        if not v or len(v) == 0:
            raise ValueError('db_user cannot be empty')
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.vectorstore_path,
            self.document_storage_path,
            "./logs",
            self.metrics_output_path,
            self.scheduled_jobs_dir,
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    @property
    def db_connection_string(self) -> str:
        """Generate database connection string dynamically"""
        if self.db_type == "mysql":
            return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            raise ValueError(f"Unknown db_type: {self.db_type}")

@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached config instance"""
    return Config()
