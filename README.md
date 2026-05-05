# RAG-Lines: Retrieval-Augmented Generation System

A comprehensive RAG system built in Python for Q&A and customer support, supporting multiple document types with local LLM inference.

## 🎯 Features

- **Multi-Format Document Support**: PDFs, Markdown, Plain Text, Web Content, Databases
- **Local LLM Inference**: No external API dependencies
- **Vector Database**: Chroma for semantic search
- **Hybrid Retrieval**: Semantic + BM25 search
- **Quality Metrics**: RAGAS evaluation framework
- **Scalable**: Local development → Server deployment

## 📁 Project Structure

```
rag-lines/
├── config/
│   ├── __init__.py
│   └── config.py                  # Configuration management
├── docs/
│   └── PROJECT_ARCHITECTURE.md    # Architecture + feature roadmap
├── scripts/
│   └── start_app.sh               # One-command launcher (Ollama + API + UI)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py     # Multi-format document loading
│   │   └── chunker.py             # Semantic chunking
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py     # Local embedding generation
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_db.py           # Chroma vector database setup
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py           # Hybrid retrieval logic
│   │   └── reranker.py            # Result reranking
│   ├── llm/
│   │   ├── __init__.py
│   │   └── local_llm.py           # Local LLM integration (Ollama)
│   ├── qa/
│   │   ├── __init__.py
│   │   └── qa_pipeline.py         # End-to-end Q&A pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # RAGAS/basic evaluation metrics
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   ├── routes.py              # API endpoints
│   │   └── scheduler.py           # In-process one-off scheduler
│   └── ui/
│       └── streamlit_app.py       # MVP Streamlit UI
├── tests/
│   └── test_pipeline.py           # Unit tests
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── main.py                        # Entry point
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/SAIRAMPATEL123/rag-lines.git
cd rag-lines
git checkout rag-development
pip install -r requirements.txt
```

### Python Version

- Recommended: **Python 3.11**
- Supported for local development: **Python 3.13.3** (with updated dependency ranges in `requirements.txt`)


### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start Local LLM (Ollama)
```bash
# Install Ollama: https://ollama.ai
ollama pull mistral  # or other model
ollama serve
```

### 4. Ingest Documents
```python
from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.chroma_db import ChromaVectorStore

loader = DocumentLoader()
docs = loader.load_documents("path/to/documents")
vectorstore = ChromaVectorStore()
vectorstore.add_documents(docs)
```

### 5. Run Q&A Pipeline
```python
from src.qa.qa_pipeline import RAGPipeline

rag = RAGPipeline()
answer = rag.query("What is customer support?")
print(answer)
```

### 6. Evaluate System
```python
from src.evaluation.evaluator import RAGEvaluator

evaluator = RAGEvaluator()
metrics = evaluator.evaluate(answers, ground_truth)
print(metrics)
```

## 📊 Performance Metrics

The system tracks:
- **Retrieval Accuracy**: % of relevant documents retrieved
- **Answer Quality**: ROUGE, BLEU, semantic similarity scores
- **Latency**: Query response time
- **Token Efficiency**: Cost per query

## 🔧 Configuration

See `config/config.py` for:
- Embedding model selection
- LLM parameters (temperature, top_k, etc.)
- Chunk size & overlap
- Retrieval top-K settings
- Database paths

## 📚 Document Types Supported

| Format | Status | Notes |
|--------|--------|-------|
| PDF | ✅ | Text extraction via pdfplumber |
| Markdown | ✅ | Direct parsing |
| Plain Text | ✅ | UTF-8 compatible |
| Web Content | ✅ | BeautifulSoup scraping |
| Databases | ✅ | SQLAlchemy support |
| DOCX | 🔄 | Upcoming |
| JSON | 🔄 | Upcoming |

## 🤖 Supported Local LLMs

- **Ollama Models**: Mistral, Llama 2, Neural Chat, etc.
- **Hugging Face**: Any GGML-quantized model
- **Custom**: Easy integration with custom models

## 🧪 Testing

```bash
python -m pytest tests/
```

## 📖 Documentation

Detailed docs coming soon:
- API Reference
- Configuration Guide
- Deployment Guide
- Troubleshooting

## 🤝 Contributing

Branch workflow:
- `main` - Production stable
- `rag-development` - Active development
- Feature branches from `rag-development`

## 📝 License

MIT License

## 💬 Support

For issues & questions, create GitHub issues.


## 🖥️ MVP Web UI

Run backend API first:
```bash
python main.py api
```

Then run Streamlit UI:
```bash
streamlit run src/ui/streamlit_app.py
```

The UI supports:
- Single query
- Batch query
- One-off scheduled query

## 🧭 Architecture Map

See `docs/PROJECT_ARCHITECTURE.md` for maintained architecture, feature matrix, and roadmap.


## ⚡ One-Command Run (Beginner Friendly)

After installing dependencies once:
```bash
pip install -r requirements.txt
```

Run everything with one command:
```bash
./scripts/start_app.sh
```

This script starts:
- Ollama server
- pulls configured model (default: `mistral`)
- FastAPI backend
- Streamlit UI

Open:
- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### About LLM location

LLM weights are managed by Ollama and stored in Ollama's model store (not inside this Git repository by default).
You can select model with:
```bash
RAG_LLM_MODEL=mistral ./scripts/start_app.sh
```
