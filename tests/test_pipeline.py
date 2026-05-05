import pytest

pytest.importorskip("loguru")
pytest.importorskip("pydantic_settings")

from src.ingestion.document_loader import Document
from src.ingestion.chunker import FixedChunker, SemanticChunker

pytest.importorskip("sentence_transformers")
from src.embeddings.embedding_model import EmbeddingModel


class TestDocumentLoader:
    """Test document loading"""

    def test_document_creation(self):
        doc = Document("Test content", "test.txt", "text")
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.doc_type == "text"


class TestChunking:
    """Test document chunking"""

    def test_fixed_chunker(self):
        doc = Document("This is a test document. " * 100, "test.txt", "text")
        chunker = FixedChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk([doc])
        assert len(chunks) > 0

    def test_semantic_chunker(self):
        doc = Document("This is a test. This is another sentence. And another one.", "test.txt", "text")
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk([doc])
        assert len(chunks) > 0


class TestEmbeddings:
    """Test embedding generation"""

    def test_embedding_dimension(self):
        embedder = EmbeddingModel()
        dim = embedder.get_embedding_dimension()
        assert dim > 0

    def test_single_embedding(self):
        embedder = EmbeddingModel()
        embedding = embedder.embed_single("Test text")
        assert embedding.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
