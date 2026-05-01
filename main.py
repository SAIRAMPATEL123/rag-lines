#!/usr/bin/env python3
"""
RAG-Lines: Main entry point for the RAG system
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from config.config import get_config
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker
from src.vectorstore.chroma_db import ChromaVectorStore
from src.qa.qa_pipeline import RAGPipeline

def setup_logging():
    """Setup logging configuration"""
    config = get_config()
    logger.remove()  # Remove default handler
    logger.add(
        config.log_file,
        level=config.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.add(sys.stderr, level=config.log_level)
    logger.info("Logging configured")

def ingest_documents(document_path: str):
    """Ingest documents into vector store"""
    logger.info(f"Starting document ingestion from: {document_path}")
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents(document_path)
    
    if not documents:
        logger.warning("No documents loaded")
        return
    
    # Chunk documents
    chunker = SemanticChunker()
    chunks = chunker.chunk(documents)
    
    # Add to vector store
    vectorstore = ChromaVectorStore()
    vectorstore.add_documents(chunks)
    
    logger.info(f"Ingestion complete: {len(chunks)} chunks added")
    print(f"✓ Ingested {len(documents)} documents into {len(chunks)} chunks")

def interactive_qa():
    """Start interactive Q&A session"""
    logger.info("Starting interactive Q&A session")
    print("\n🚀 RAG-Lines Interactive Q&A")
    print("Type 'exit' or 'quit' to end session\n")
    
    pipeline = RAGPipeline()
    
    while True:
        try:
            question = input("❓ Question: ").strip()
            if question.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            result = pipeline.query(question)
            print(f"\n✅ Answer: {result['answer']}")
            print(f"📊 Retrieved {result['document_count']} documents\n")
        except KeyboardInterrupt:
            print("\n\n👋 Session ended")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"❌ Error: {e}\n")

def start_api():
    """Start FastAPI server"""
    logger.info("Starting FastAPI server")
    from src.api.app import create_app
    import uvicorn
    from config.config import get_config
    
    config = get_config()
    app = create_app()
    print(f"\n🚀 Starting RAG-Lines API")
    print(f"📍 Server: http://{config.api_host}:{config.api_port}")
    print(f"📚 Docs: http://{config.api_host}:{config.api_port}/docs\n")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower()
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG-Lines: Retrieval-Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest ./documents
  python main.py qa
  python main.py api
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="Path to documents directory or file")
    
    # QA command
    subparsers.add_parser("qa", help="Start interactive Q&A")
    
    # API command
    subparsers.add_parser("api", help="Start FastAPI server")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info(f"RAG-Lines started with command: {args.command}")
    
    if args.command == "ingest":
        ingest_documents(args.path)
    elif args.command == "qa":
        interactive_qa()
    elif args.command == "api":
        start_api()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
