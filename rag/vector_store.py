"""
Vector Store Module.

This module handles FAISS vector store operations including:
- Creating indexes from documents
- Persisting and loading indexes
- Managing document metadata
"""

import logging
from pathlib import Path
from typing import Optional, Union

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .chunker import TextChunk
from .embeddings import EmbeddingManager

# Configure logging
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manager for FAISS vector store operations.
    
    This class handles the creation, persistence, and management of
    FAISS vector indexes for efficient similarity search.
    
    Features:
        - Create index from documents or text chunks
        - Persist index to disk for fast reloading
        - Load existing index if available
        - Add new documents to existing index
        - Delete and recreate index
    
    Attributes:
        index_dir: Directory for index persistence
        index_name: Name of the index (used for file naming)
        embedding_manager: Manager for generating embeddings
    
    Example:
        >>> manager = VectorStoreManager(embedding_manager)
        >>> manager.create_index(documents)
        >>> manager.save()
        >>> # Later...
        >>> manager.load()
        >>> results = manager.similarity_search("query", k=3)
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        index_dir: Optional[Union[str, Path]] = None,
        index_name: str = "document_index"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_manager: Manager for generating embeddings
            index_dir: Directory for storing the index
            index_name: Name identifier for the index
        """
        self.embedding_manager = embedding_manager
        self.index_name = index_name
        
        # Set default index directory
        if index_dir is None:
            index_dir = Path(__file__).parent.parent / "data" / "index"
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS index (will be created or loaded)
        self._index: Optional[FAISS] = None
        
        # Track indexed documents
        self._document_count: int = 0
        
        logger.info(f"VectorStoreManager initialized: {index_name}")
    
    @property
    def index_path(self) -> Path:
        """Get the path for the index files."""
        return self.index_dir / self.index_name
    
    @property
    def is_loaded(self) -> bool:
        """Check if an index is currently loaded."""
        return self._index is not None
    
    @property
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        return self._document_count
    
    def create_index(self, documents: list[Document]) -> None:
        """
        Create a new FAISS index from documents.
        
        Args:
            documents: List of LangChain Document objects to index
            
        Note:
            This will overwrite any existing index.
        """
        if not documents:
            raise ValueError("Cannot create index from empty document list")
        
        logger.info(f"Creating index from {len(documents)} documents...")
        
        # Get LangChain embeddings object
        embeddings = self.embedding_manager.langchain_embeddings
        
        # Create FAISS index
        self._index = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        self._document_count = len(documents)
        
        logger.info(f"Index created with {self._document_count} documents")
    
    def create_index_from_chunks(self, chunks: list[TextChunk]) -> None:
        """
        Create index from TextChunk objects.
        
        Args:
            chunks: List of TextChunk objects to index
        """
        # Convert chunks to LangChain documents
        documents = [chunk.to_langchain_document() for chunk in chunks]
        self.create_index(documents)
    
    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to an existing index.
        
        Args:
            documents: List of documents to add
            
        Raises:
            RuntimeError: If no index is loaded
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Create or load an index first.")
        
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to index...")
        
        # Add to existing index
        self._index.add_documents(documents)
        self._document_count += len(documents)
        
        logger.info(f"Index now contains {self._document_count} documents")
    
    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """
        Add TextChunk objects to an existing index.
        
        Args:
            chunks: List of TextChunk objects to add
        """
        documents = [chunk.to_langchain_document() for chunk in chunks]
        self.add_documents(documents)
    
    def save(self) -> None:
        """
        Persist the index to disk.
        
        The index is saved to the configured index_dir with the index_name.
        """
        if not self.is_loaded:
            raise RuntimeError("No index to save. Create an index first.")
        
        logger.info(f"Saving index to {self.index_path}...")
        
        self._index.save_local(str(self.index_path))
        
        # Save metadata
        metadata_path = self.index_path / "metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"document_count={self._document_count}\n")
            f.write(f"embedding_model={self.embedding_manager.model_name}\n")
            f.write(f"embedding_dimension={self.embedding_manager.dimension}\n")
        
        logger.info("Index saved successfully")
    
    def load(self, allow_dangerous_deserialization: bool = True) -> bool:
        """
        Load an existing index from disk.
        
        Args:
            allow_dangerous_deserialization: Allow loading pickled data
                (required for FAISS, but only use with trusted data)
            
        Returns:
            True if index was loaded, False if no index exists
        """
        index_file = self.index_path / "index.faiss"
        
        if not index_file.exists():
            logger.info("No existing index found")
            return False
        
        logger.info(f"Loading index from {self.index_path}...")
        
        embeddings = self.embedding_manager.langchain_embeddings
        
        self._index = FAISS.load_local(
            str(self.index_path),
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        # Load metadata
        metadata_path = self.index_path / "metadata.txt"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                for line in f:
                    if line.startswith("document_count="):
                        self._document_count = int(line.split("=")[1].strip())
        
        logger.info(f"Index loaded: {self._document_count} documents")
        return True
    
    def exists(self) -> bool:
        """Check if a persisted index exists."""
        index_file = self.index_path / "index.faiss"
        return index_file.exists()
    
    def delete(self) -> None:
        """Delete the persisted index."""
        import shutil
        
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
            logger.info(f"Deleted index at {self.index_path}")
        
        self._index = None
        self._document_count = 0
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 3
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples, sorted by similarity
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Create or load an index first.")
        
        results = self._index.similarity_search_with_score(query, k=k)
        
        # Results are (document, distance) - lower distance = more similar
        # Convert to (document, similarity_score) where higher = better
        # Using 1/(1+distance) transformation
        scored_results = [
            (doc, 1.0 / (1.0 + score)) for doc, score in results
        ]
        
        return scored_results
    
    def similarity_search_simple(
        self, 
        query: str, 
        k: int = 3
    ) -> list[Document]:
        """
        Search for similar documents (without scores).
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of documents
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Create or load an index first.")
        
        return self._index.similarity_search(query, k=k)
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a LangChain retriever interface.
        
        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 5})
            
        Returns:
            LangChain Retriever object
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Create or load an index first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        return self._index.as_retriever(search_kwargs=search_kwargs)
    
    def get_index_info(self) -> dict:
        """
        Get information about the current index.
        
        Returns:
            Dictionary with index information
        """
        return {
            "is_loaded": self.is_loaded,
            "document_count": self._document_count,
            "index_name": self.index_name,
            "index_path": str(self.index_path),
            "exists_on_disk": self.exists(),
            "embedding_model": self.embedding_manager.model_name,
            "embedding_dimension": self.embedding_manager.dimension
        }
