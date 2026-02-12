"""
Configuration module for the RAG Chat System.

This module centralizes all configuration settings, making the system
easily configurable without modifying code. Settings can be overridden
via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables from .env file
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    # Model provider: "openai" or "huggingface"
    provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    
    # OpenAI embedding model
    openai_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # HuggingFace embedding model (runs locally, no API key needed)
    huggingface_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Embedding dimension (depends on model)
    dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # MiniLM default
    
    # Cache embeddings to avoid recomputation
    cache_enabled: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    # Target chunk size in characters (approximately maps to tokens)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    
    # Overlap between chunks to maintain context
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Separators for text splitting (in order of priority)
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class RetrieverConfig:
    """Configuration for document retrieval."""
    
    # Number of chunks to retrieve
    top_k: int = int(os.getenv("RETRIEVER_TOP_K", "3"))
    
    # Similarity threshold (0-1, higher = more similar required)
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    
    # Search type: "similarity" or "mmr" (maximal marginal relevance)
    search_type: str = os.getenv("SEARCH_TYPE", "similarity")


@dataclass
class LLMConfig:
    """Configuration for the Language Model."""
    
    # LLM provider: "openai" or "huggingface"
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    
    # OpenAI model
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # HuggingFace model (for local inference)
    huggingface_model: str = os.getenv(
        "HUGGINGFACE_LLM_MODEL",
        "google/flan-t5-base"
    )
    
    # Temperature for generation (0 = deterministic, 1 = creative)
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # Maximum tokens in response
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Base directory for the project
    base_dir: Path = Path(__file__).parent
    
    # Directory for uploaded documents
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    
    # Directory for FAISS index persistence
    index_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "index")
    
    # Directory for embedding cache
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "cache")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration combining all settings."""
    
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.paths.ensure_directories()
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check API keys based on provider settings
        if self.embedding.provider == "openai" and not self.openai_api_key:
            issues.append("OpenAI API key required for OpenAI embeddings")
        
        if self.llm.provider == "openai" and not self.openai_api_key:
            issues.append("OpenAI API key required for OpenAI LLM")
        
        # Validate chunking settings
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            issues.append("Chunk overlap must be smaller than chunk size")
        
        # Validate retriever settings
        if self.retriever.top_k < 1:
            issues.append("Top-k must be at least 1")
        
        return issues


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AppConfig: The application configuration
    """
    return config


def reload_config() -> AppConfig:
    """
    Reload configuration from environment variables.
    
    Useful when environment variables are changed at runtime.
    
    Returns:
        AppConfig: Fresh configuration instance
    """
    global config
    load_dotenv(override=True)
    config = AppConfig()
    return config
