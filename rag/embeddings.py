"""
Embedding Module.

This module handles embedding model setup, text embedding generation,
and embedding caching for efficiency.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Union
from abc import ABC, abstractmethod

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputation.
    
    Uses file-based persistence with hash-based keys for efficient
    lookup and storage.
    """
    
    def __init__(self, cache_dir: Union[str, Path], enabled: bool = True):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory for cache storage
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._memory_cache: dict[str, list[float]] = {}
        
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _get_hash(self, text: str, model_name: str) -> str:
        """Generate hash key for text and model combination."""
        key = f"{model_name}:{text}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._memory_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._memory_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._memory_cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self._memory_cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def get(self, text: str, model_name: str) -> Optional[list[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: The text to look up
            model_name: Name of the embedding model
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.enabled:
            return None
        
        key = self._get_hash(text, model_name)
        return self._memory_cache.get(key)
    
    def set(self, text: str, model_name: str, embedding: list[float]) -> None:
        """
        Cache an embedding.
        
        Args:
            text: The original text
            model_name: Name of the embedding model
            embedding: The embedding vector
        """
        if not self.enabled:
            return
        
        key = self._get_hash(text, model_name)
        self._memory_cache[key] = embedding
    
    def save(self) -> None:
        """Persist cache to disk."""
        self._save_cache()
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache = {}
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()


class HuggingFaceEmbedding(BaseEmbedding):
    """
    HuggingFace embedding model using sentence-transformers.
    
    This runs locally and doesn't require an API key.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize HuggingFace embedding model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self._dimension: Optional[int] = None
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Get dimension by embedding a test text
            test_embedding = self._model.embed_query("test")
            self._dimension = len(test_embedding)
            
            logger.info(f"Loaded HuggingFace model: {model_name} (dim={self._dimension})")
            
        except ImportError:
            logger.error("langchain-huggingface not installed")
            raise ImportError(
                "Please install langchain-huggingface: pip install langchain-huggingface"
            )
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._model.embed_query(text)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return self._model.embed_documents(texts)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
    
    @property
    def langchain_embeddings(self):
        """Return the underlying LangChain embeddings object."""
        return self._model


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI embedding model.
    
    Requires OpenAI API key.
    """
    
    # Dimension mapping for known OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        api_key: str,
        model_name: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI embedding model
        """
        self.model_name = model_name
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)
        
        try:
            from langchain_openai import OpenAIEmbeddings
            
            self._model = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=model_name
            )
            
            logger.info(f"Initialized OpenAI embeddings: {model_name}")
            
        except ImportError:
            logger.error("langchain-openai not installed")
            raise ImportError(
                "Please install langchain-openai: pip install langchain-openai"
            )
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._model.embed_query(text)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return self._model.embed_documents(texts)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
    
    @property
    def langchain_embeddings(self):
        """Return the underlying LangChain embeddings object."""
        return self._model


class EmbeddingManager:
    """
    High-level manager for embedding operations.
    
    This class provides a unified interface for embedding generation
    with support for caching and multiple embedding providers.
    
    Attributes:
        provider: The embedding provider ("openai" or "huggingface")
        model_name: Name of the embedding model
        cache_enabled: Whether embedding caching is enabled
    
    Example:
        >>> manager = EmbeddingManager(provider="huggingface")
        >>> embedding = manager.embed_text("Hello world")
        >>> embeddings = manager.embed_texts(["Hello", "World"])
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the embedding manager.
        
        Args:
            provider: "openai" or "huggingface"
            model_name: Model name (uses default if not specified)
            api_key: API key (required for OpenAI)
            cache_dir: Directory for caching embeddings
            cache_enabled: Whether to cache embeddings
        """
        self.provider = provider.lower()
        self.cache_enabled = cache_enabled
        
        # Set default model names
        if model_name is None:
            if self.provider == "openai":
                model_name = "text-embedding-3-small"
            else:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        self.model_name = model_name
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data" / "cache"
        
        self._cache = EmbeddingCache(cache_dir, enabled=cache_enabled)
        
        # Initialize embedding model
        if self.provider == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI embeddings")
            self._embedding = OpenAIEmbedding(api_key, model_name)
        else:
            self._embedding = HuggingFaceEmbedding(model_name)
        
        logger.info(f"EmbeddingManager initialized: {provider}/{model_name}")
    
    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cached = self._cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        # Generate embedding
        embedding = self._embedding.embed_text(text)
        
        # Cache result
        if use_cache and self.cache_enabled:
            self._cache.set(text, self.model_name, embedding)
        
        return embedding
    
    def embed_texts(
        self, 
        texts: list[str], 
        use_cache: bool = True,
        show_progress: bool = False
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            if use_cache and self.cache_enabled:
                cached = self._cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue
            texts_to_embed.append(text)
            indices_to_embed.append(i)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            new_embeddings = self._embedding.embed_texts(texts_to_embed)
            
            for i, (original_idx, embedding) in enumerate(zip(indices_to_embed, new_embeddings)):
                embeddings.append((original_idx, embedding))
                
                # Cache new embedding
                if use_cache and self.cache_enabled:
                    self._cache.set(texts_to_embed[i], self.model_name, embedding)
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = [e[1] for e in embeddings]
        
        logger.info(
            f"Generated {len(texts_to_embed)} embeddings "
            f"({len(texts) - len(texts_to_embed)} from cache)"
        )
        
        return result
    
    def save_cache(self) -> None:
        """Persist embedding cache to disk."""
        self._cache.save()
        logger.info("Embedding cache saved")
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding.dimension
    
    @property
    def langchain_embeddings(self):
        """Get underlying LangChain embeddings for FAISS integration."""
        return self._embedding.langchain_embeddings
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "cache_enabled": self.cache_enabled
        }
