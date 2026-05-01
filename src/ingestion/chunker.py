from typing import List
from src.ingestion.document_loader import Document
from loguru import logger
from config.config import get_config

class Chunk:
    """Represents a chunk of text with metadata"""
    def __init__(self, text: str, chunk_id: str, source: str, chunk_index: int, metadata: dict = None):
        self.text = text
        self.chunk_id = chunk_id
        self.source = source
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

class FixedChunker:
    """Fixed-size chunking strategy"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.config = get_config()
        self.chunk_size = chunk_size or self.config.chunk_size
        self.overlap = overlap or self.config.chunk_overlap
    
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into fixed-size chunks"""
        chunks = []
        chunk_counter = 0
        
        for doc in documents:
            text = doc.content
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk_text = text[i:i + self.chunk_size]
                if chunk_text.strip():
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{doc.source}_chunk_{chunk_counter}",
                        source=doc.source,
                        chunk_index=chunk_counter,
                        metadata={
                            **doc.metadata,
                            "doc_type": doc.doc_type,
                            "original_index": i
                        }
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
        
        logger.info(f"Created {len(chunks)} chunks using FixedChunker")
        return chunks

class SemanticChunker:
    """Semantic-aware chunking strategy"""
    
    def __init__(self, chunk_size: int = None):
        self.config = get_config()
        self.chunk_size = chunk_size or self.config.chunk_size
    
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into semantically meaningful chunks"""
        chunks = []
        chunk_counter = 0
        
        for doc in documents:
            # Split by sentence boundaries
            sentences = self._split_into_sentences(doc.content)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk.strip():
                        chunk = Chunk(
                            text=current_chunk.strip(),
                            chunk_id=f"{doc.source}_chunk_{chunk_counter}",
                            source=doc.source,
                            chunk_index=chunk_counter,
                            metadata={
                                **doc.metadata,
                                "doc_type": doc.doc_type
                            }
                        )
                        chunks.append(chunk)
                        chunk_counter += 1
                    current_chunk = sentence
            
            # Add remaining chunk
            if current_chunk.strip():
                chunk = Chunk(
                    text=current_chunk.strip(),
                    chunk_id=f"{doc.source}_chunk_{chunk_counter}",
                    source=doc.source,
                    chunk_index=chunk_counter,
                    metadata={
                        **doc.metadata,
                        "doc_type": doc.doc_type
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Created {len(chunks)} chunks using SemanticChunker")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple implementation)"""
        import re
        # Split by common sentence terminators
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
