import sys
import types

# Minimal stubs so core unit tests can run in constrained environments.
if "loguru" not in sys.modules:
    fake_loguru = types.ModuleType("loguru")

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    fake_loguru.logger = _Logger()
    sys.modules["loguru"] = fake_loguru

if "pydantic_settings" not in sys.modules:
    fake_ps = types.ModuleType("pydantic_settings")

    class BaseSettings:
        def __init__(self, **data):
            # Populate instance from class attributes, then override with kwargs.
            for name, value in self.__class__.__dict__.items():
                if name.startswith("_") or callable(value):
                    continue
                setattr(self, name, value)
            for k, v in data.items():
                setattr(self, k, v)

    fake_ps.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_ps

import pytest
from src.ingestion.document_loader import Document
from src.ingestion.chunker import FixedChunker, SemanticChunker


class TestDocumentLoader:
    def test_document_creation(self):
        doc = Document("Test content", "test.txt", "text")
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.doc_type == "text"


class TestChunking:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
