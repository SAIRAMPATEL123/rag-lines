import os
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import markdown2
from bs4 import BeautifulSoup
import requests
from loguru import logger
from config.config import get_config

class Document:
    """Document object with metadata"""
    def __init__(self, content: str, source: str, doc_type: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.source = source
        self.doc_type = doc_type
        self.metadata = metadata or {}

class DocumentLoader:
    """Load and parse multiple document formats"""
    
    def __init__(self):
        self.config = get_config()
        logger.info("DocumentLoader initialized")
    
    def load_documents(self, path: str) -> List[Document]:
        """Load documents from directory or single file"""
        documents = []
        path_obj = Path(path)
        
        if path_obj.is_file():
            documents.append(self._load_file(path))
        elif path_obj.is_dir():
            for file_path in path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.config.supported_formats:
                    try:
                        doc = self._load_file(str(file_path))
                        documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _load_file(self, file_path: str) -> Document:
        """Load single file based on type"""
        suffix = Path(file_path).suffix.lower()
        
        if suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.md':
            return self._load_markdown(file_path)
        elif suffix == '.txt':
            return self._load_text(file_path)
        elif suffix == '.html':
            return self._load_html(file_path)
        elif suffix == '.json':
            return self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _load_pdf(self, file_path: str) -> Document:
        """Extract text from PDF"""
        try:
            text = ""
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
            logger.info(f"Loaded PDF: {file_path}")
            return Document(text, file_path, "pdf", {"pages": len(reader.pages)})
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def _load_markdown(self, file_path: str) -> Document:
        """Load markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded Markdown: {file_path}")
        return Document(content, file_path, "markdown")
    
    def _load_text(self, file_path: str) -> Document:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded Text: {file_path}")
        return Document(content, file_path, "text")
    
    def _load_html(self, file_path: str) -> Document:
        """Load HTML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n', strip=True)
        logger.info(f"Loaded HTML: {file_path}")
        return Document(text, file_path, "html")
    
    def _load_json(self, file_path: str) -> Document:
        """Load JSON file"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        logger.info(f"Loaded JSON: {file_path}")
        return Document(content, file_path, "json")
    
    def load_from_web(self, url: str) -> Document:
        """Load and parse web content"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            logger.info(f"Loaded web content: {url}")
            return Document(text, url, "web", {"url": url})
        except Exception as e:
            logger.error(f"Error loading web content from {url}: {e}")
            raise
