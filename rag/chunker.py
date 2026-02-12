"""
Text Chunking Module.

This module handles splitting documents into smaller chunks for embedding.
It preserves metadata and supports configurable chunk sizes with overlap.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .loader import LoadedDocument

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    
    source: str
    page_number: int
    chunk_index: int
    total_chunks: int
    file_name: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    extra: dict = field(default_factory=dict)


@dataclass
class TextChunk:
    """A chunk of text with associated metadata."""
    
    content: str
    metadata: ChunkMetadata
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata={
                "source": self.metadata.source,
                "page": self.metadata.page_number,
                "chunk_index": self.metadata.chunk_index,
                "total_chunks": self.metadata.total_chunks,
                "file_name": self.metadata.file_name,
                **self.metadata.extra
            }
        )


class TextChunker:
    """
    Text chunking with overlap and metadata preservation.
    
    This class splits documents into smaller chunks suitable for embedding,
    while preserving important metadata like page numbers and source files.
    
    The overlap ensures that context is maintained across chunk boundaries,
    which is crucial for accurate retrieval.
    
    Attributes:
        chunk_size: Target size for each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to use when splitting (priority order)
    
    Example:
        >>> chunker = TextChunker(chunk_size=800, chunk_overlap=150)
        >>> chunks = chunker.chunk_documents(documents)
        >>> print(f"Created {len(chunks)} chunks")
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[list[str]] = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            chunk_overlap: Overlap between consecutive chunks
            separators: Custom separators for splitting (optional)
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        # Initialize LangChain text splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(
            f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_text(
        self, 
        text: str,
        source: str = "unknown",
        page_number: int = 1,
        file_name: str = "unknown"
    ) -> list[TextChunk]:
        """
        Split a single text into chunks.
        
        Args:
            text: The text to split
            source: Source identifier for metadata
            page_number: Page number for metadata
            file_name: File name for metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Use LangChain splitter
        splits = self._splitter.split_text(text)
        
        if not splits:
            return []
        
        chunks = []
        current_pos = 0
        
        for i, split in enumerate(splits):
            # Find position in original text
            start_char = text.find(split, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(split)
            current_pos = start_char + 1  # Move past this occurrence
            
            chunk = TextChunk(
                content=split,
                metadata=ChunkMetadata(
                    source=source,
                    page_number=page_number,
                    chunk_index=i,
                    total_chunks=len(splits),
                    file_name=file_name,
                    start_char=start_char,
                    end_char=end_char
                )
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, document: LoadedDocument) -> list[TextChunk]:
        """
        Split a single LoadedDocument into chunks.
        
        Args:
            document: The document to split
            
        Returns:
            List of TextChunk objects
        """
        return self.chunk_text(
            text=document.content,
            source=document.metadata.source,
            page_number=document.metadata.page_number,
            file_name=document.metadata.file_name
        )
    
    def chunk_documents(
        self, 
        documents: list[LoadedDocument]
    ) -> list[TextChunk]:
        """
        Split multiple documents into chunks.
        
        This method processes multiple pages/documents and combines
        all chunks into a single list with proper metadata.
        
        Args:
            documents: List of LoadedDocument objects
            
        Returns:
            List of all TextChunk objects from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        # Update chunk indices to be global across all documents
        for i, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = i
        
        # Update total_chunks for all
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata.total_chunks = total
        
        logger.info(f"Created {total} chunks from {len(documents)} documents")
        return all_chunks
    
    def chunk_langchain_documents(
        self, 
        documents: list[Document]
    ) -> list[Document]:
        """
        Split LangChain Document objects directly.
        
        Convenience method for working directly with LangChain's
        Document objects.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked LangChain Document objects
        """
        all_chunks = self._splitter.split_documents(documents)
        
        # Add chunk indices to metadata
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    @staticmethod
    def get_chunk_statistics(chunks: list[TextChunk]) -> dict:
        """
        Get statistics about the created chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_characters": 0
            }
        
        sizes = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes)
        }
    
    def estimate_chunks(self, text_length: int) -> int:
        """
        Estimate number of chunks for given text length.
        
        Args:
            text_length: Total length of text in characters
            
        Returns:
            Estimated number of chunks
        """
        if text_length <= self.chunk_size:
            return 1
        
        effective_length = self.chunk_size - self.chunk_overlap
        return max(1, (text_length - self.chunk_overlap) // effective_length + 1)
