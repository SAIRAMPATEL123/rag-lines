from typing import List
from src.ingestion.document_loader import Document
from loguru import logger
from config.config import get_config
import hashlib

class Chunk:
    """Represents a chunk of text with metadata"""
    def __init__(self, text: str, chunk_id: str, source: str, chunk_index: int, metadata: dict = None):
        self.text = text
        self.chunk_id = chunk_id
        self.source = source
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for token counting"""
    return text.split()


def _extract_page_from_text(text: str) -> int:
    """Extract page number from [PAGE N] marker at start of text"""
    import re
    match = re.match(r'\[PAGE (\d+)\]', text)
    if match:
        return int(match.group(1))
    return None


def _count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(_tokenize(text))


class FixedChunker:
    """Legacy fixed-size chunking (deprecated in favor of TokenAwareChunker)"""

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.config = get_config()
        self.chunk_size = chunk_size or self.config.chunk_size
        self.overlap = overlap or self.config.chunk_overlap
        logger.warning("FixedChunker is deprecated; use TokenAwareChunker instead")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into fixed-size chunks (char-based, legacy)"""
        chunks = []
        chunk_counter = 0

        for doc in documents:
            tokens = _tokenize(doc.content)

            doc_hash = hashlib.sha256(f"{doc.source}:v1".encode()).hexdigest()[:8]

            for i in range(0, len(tokens), self.chunk_size - self.overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]

                if not chunk_tokens:
                    continue

                chunk_text = " ".join(chunk_tokens)
                chunk_id = f"{doc_hash}_{chunk_counter:04d}"

                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=doc.source,
                    chunk_index=chunk_counter,
                    metadata={
                        **doc.metadata,
                        "doc_type": doc.doc_type,
                        "token_count": len(chunk_tokens),
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1

        logger.info(f"Created {len(chunks)} chunks using FixedChunker (legacy)")
        return chunks


class TokenAwareChunker:
    """Token-aware chunking strategy (not character-based)"""

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.config = get_config()
        self.chunk_size = chunk_size or self.config.chunk_size  # in tokens
        self.overlap = overlap or self.config.chunk_overlap    # in tokens
        logger.info(f"TokenAwareChunker: {self.chunk_size} tokens per chunk, {self.overlap} token overlap")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into token-based chunks with overlap"""
        chunks = []

        for doc in documents:
            tokens = _tokenize(doc.content)

            # Create deterministic chunk IDs based on doc + version + index
            doc_hash = hashlib.sha256(f"{doc.source}:v1".encode()).hexdigest()[:8]

            chunk_index = 0
            for i in range(0, len(tokens), self.chunk_size - self.overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]

                if not chunk_tokens:
                    continue

                chunk_text = " ".join(chunk_tokens)

                # Deterministic chunk ID: doc_hash + chunk_index
                chunk_id = f"{doc_hash}_{chunk_index:04d}"

                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=doc.source,
                    chunk_index=chunk_index,
                    metadata={
                        **doc.metadata,
                        "doc_type": doc.doc_type,
                        "token_count": len(chunk_tokens),
                        "doc_hash": doc_hash,
                        "version": 1,
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Created {len(chunks)} token-aware chunks")
        return chunks


class SemanticChunker(TokenAwareChunker):
    """Semantic-aware chunking (now token-based with overlap)"""

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into semantically meaningful chunks by paragraphs/sentences"""
        chunks = []

        for doc in documents:
            # Split by paragraphs first
            paragraphs = doc.content.split("\n\n")

            # Deterministic chunk ID
            doc_hash = hashlib.sha256(f"{doc.source}:v1".encode()).hexdigest()[:8]
            chunk_index = 0
            current_chunk_tokens = []
            current_page = None  # Track current page

            for paragraph in paragraphs:
                # Extract page marker if present
                page_num = _extract_page_from_text(paragraph)
                if page_num:
                    current_page = page_num

                para_tokens = _tokenize(paragraph)

                # Check if adding this paragraph would exceed chunk_size
                if (len(current_chunk_tokens) + len(para_tokens)) > self.chunk_size and current_chunk_tokens:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk_tokens)
                    chunk_id = f"{doc_hash}_{chunk_index:04d}"

                    metadata = {
                        **doc.metadata,
                        "doc_type": doc.doc_type,
                        "token_count": len(current_chunk_tokens),
                        "doc_hash": doc_hash,
                        "version": 1,
                    }
                    if current_page:
                        metadata["page"] = current_page

                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source=doc.source,
                        chunk_index=chunk_index,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Start new chunk with overlap tokens from previous
                    if self.overlap > 0 and current_chunk_tokens:
                        current_chunk_tokens = current_chunk_tokens[-self.overlap:]
                    else:
                        current_chunk_tokens = []

                # Add paragraph tokens
                current_chunk_tokens.extend(para_tokens)

            # Add remaining chunk
            if current_chunk_tokens:
                chunk_text = " ".join(current_chunk_tokens)
                chunk_id = f"{doc_hash}_{chunk_index:04d}"

                metadata = {
                    **doc.metadata,
                    "doc_type": doc.doc_type,
                    "token_count": len(current_chunk_tokens),
                    "doc_hash": doc_hash,
                    "version": 1,
                }
                if current_page:
                    metadata["page"] = current_page

                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=doc.source,
                    chunk_index=chunk_index,
                    metadata=metadata
                )
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} semantic chunks with token awareness")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            try:
                sent_tokenize("test")
            except LookupError:
                nltk.download('punkt', quiet=True)

            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]

        except ImportError:
            logger.warning("NLTK not available, falling back to regex")
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
