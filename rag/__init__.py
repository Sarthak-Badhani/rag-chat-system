"""
RAG (Retrieval-Augmented Generation) Package.

This package contains all the core components for building a document
question-answering system using RAG architecture.

Modules:
    - loader: PDF document loading and text extraction
    - chunker: Text splitting with overlap and metadata preservation
    - embeddings: Embedding model setup and caching
    - vector_store: FAISS vector store management
    - retriever: Semantic similarity search
    - prompt_builder: Structured prompt engineering
    - generator: LLM-based answer generation
"""

from .loader import PDFLoader
from .chunker import TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .retriever import DocumentRetriever
from .prompt_builder import PromptBuilder
from .generator import AnswerGenerator

__all__ = [
    "PDFLoader",
    "TextChunker", 
    "EmbeddingManager",
    "VectorStoreManager",
    "DocumentRetriever",
    "PromptBuilder",
    "AnswerGenerator",
]
