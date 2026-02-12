"""
Document Retrieval Module.

This module handles retrieving relevant document chunks based on
semantic similarity to user queries.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from .vector_store import VectorStoreManager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    A retrieved document chunk with its metadata and relevance score.
    
    Attributes:
        content: The text content of the chunk
        page_number: Source page number
        file_name: Source file name
        chunk_index: Index of the chunk
        similarity_score: Relevance score (0-1, higher is better)
        metadata: Additional metadata
    """
    
    content: str
    page_number: int
    file_name: str
    chunk_index: int = 0
    similarity_score: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_citation(self) -> str:
        """Format as a citation string."""
        return f"[{self.file_name}, Page {self.page_number}]"
    
    def to_display_string(self, max_length: int = 200) -> str:
        """Format for display in UI."""
        content_preview = self.content[:max_length]
        if len(self.content) > max_length:
            content_preview += "..."
        
        return (
            f"**Source**: {self.file_name}, Page {self.page_number}\n"
            f"**Score**: {self.similarity_score:.3f}\n"
            f"**Content**: {content_preview}"
        )


@dataclass
class RetrievalResult:
    """
    Complete result of a retrieval operation.
    
    Attributes:
        query: The original query
        chunks: List of retrieved chunks
        total_chunks_searched: Total number of chunks in index
    """
    
    query: str
    chunks: list[RetrievedChunk]
    total_chunks_searched: int = 0
    
    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.chunks) > 0
    
    def get_context_text(self, separator: str = "\n\n---\n\n") -> str:
        """
        Combine all chunks into a single context string.
        
        Args:
            separator: Separator between chunks
            
        Returns:
            Combined context text
        """
        return separator.join(chunk.content for chunk in self.chunks)
    
    def get_all_citations(self) -> list[str]:
        """Get unique citations from all chunks."""
        citations = []
        seen = set()
        
        for chunk in self.chunks:
            citation = chunk.to_citation()
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)
        
        return citations
    
    def get_unique_sources(self) -> list[dict]:
        """Get unique source documents with page numbers."""
        sources = {}
        
        for chunk in self.chunks:
            key = chunk.file_name
            if key not in sources:
                sources[key] = {
                    "file_name": chunk.file_name,
                    "pages": set()
                }
            sources[key]["pages"].add(chunk.page_number)
        
        # Convert pages sets to sorted lists
        return [
            {
                "file_name": s["file_name"],
                "pages": sorted(s["pages"])
            }
            for s in sources.values()
        ]


class DocumentRetriever:
    """
    High-level document retrieval using semantic search.
    
    This class provides an easy-to-use interface for retrieving
    relevant document chunks based on semantic similarity.
    
    Features:
        - Configurable top-k retrieval
        - Similarity threshold filtering
        - Rich metadata in results
        - Citation generation
    
    Attributes:
        vector_store: The underlying vector store
        top_k: Default number of results to retrieve
        similarity_threshold: Minimum similarity score (0-1)
    
    Example:
        >>> retriever = DocumentRetriever(vector_store, top_k=5)
        >>> result = retriever.retrieve("What is machine learning?")
        >>> for chunk in result.chunks:
        ...     print(f"{chunk.to_citation()}: {chunk.content[:100]}...")
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        top_k: int = 3,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: Vector store manager instance
            top_k: Default number of results to retrieve
            similarity_threshold: Minimum similarity score for results
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"DocumentRetriever initialized: top_k={top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of results (overrides default)
            similarity_threshold: Minimum score (overrides default)
            
        Returns:
            RetrievalResult containing matched chunks
        """
        k = top_k if top_k is not None else self.top_k
        threshold = (
            similarity_threshold 
            if similarity_threshold is not None 
            else self.similarity_threshold
        )
        
        logger.info(f"Retrieving top-{k} chunks for query: {query[:50]}...")
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)
        
        # Convert to RetrievedChunk objects
        chunks = []
        for doc, score in results:
            # Apply similarity threshold
            if score < threshold:
                continue
            
            chunk = RetrievedChunk(
                content=doc.page_content,
                page_number=doc.metadata.get("page", 0),
                file_name=doc.metadata.get("file_name", doc.metadata.get("source", "unknown")),
                chunk_index=doc.metadata.get("chunk_index", 0),
                similarity_score=score,
                metadata=dict(doc.metadata)
            )
            chunks.append(chunk)
        
        result = RetrievalResult(
            query=query,
            chunks=chunks,
            total_chunks_searched=self.vector_store.document_count
        )
        
        logger.info(f"Retrieved {len(chunks)} chunks above threshold {threshold}")
        
        return result
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with raw similarity scores.
        
        This is a lower-level method that returns LangChain
        Document objects with their scores.
        
        Args:
            query: The search query
            top_k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        k = top_k if top_k is not None else self.top_k
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve_simple(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> list[Document]:
        """
        Simple retrieval returning just documents.
        
        Args:
            query: The search query
            top_k: Number of results
            
        Returns:
            List of Document objects
        """
        k = top_k if top_k is not None else self.top_k
        return self.vector_store.similarity_search_simple(query, k=k)
    
    def find_similar_chunks(
        self,
        chunk_content: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> RetrievalResult:
        """
        Find chunks similar to a given chunk.
        
        Useful for finding related content within the document.
        
        Args:
            chunk_content: Content of the reference chunk
            top_k: Number of similar chunks to find
            exclude_self: Whether to exclude the exact same content
            
        Returns:
            RetrievalResult with similar chunks
        """
        # Retrieve more if we need to exclude self
        k = top_k + 1 if exclude_self else top_k
        
        result = self.retrieve(chunk_content, top_k=k)
        
        if exclude_self:
            # Remove chunks with very high similarity (likely self)
            result.chunks = [
                c for c in result.chunks 
                if c.similarity_score < 0.99 or c.content != chunk_content
            ][:top_k]
        
        return result
    
    def batch_retrieve(
        self,
        queries: list[str],
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of RetrievalResult objects
        """
        results = []
        for query in queries:
            result = self.retrieve(query, top_k=top_k)
            results.append(result)
        return results
    
    def get_retriever_config(self) -> dict:
        """Get current retriever configuration."""
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "index_loaded": self.vector_store.is_loaded,
            "document_count": self.vector_store.document_count
        }
