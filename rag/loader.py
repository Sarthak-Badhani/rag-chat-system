"""
PDF Loading Module.

This module handles loading and text extraction from PDF documents.
It supports multi-page PDFs and preserves page number metadata for citations.
"""

import logging
from pathlib import Path
from typing import Union, BinaryIO
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata associated with a loaded document."""
    
    source: str
    page_number: int
    total_pages: int
    file_name: str
    extra: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """A loaded document with content and metadata."""
    
    content: str
    metadata: DocumentMetadata
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata={
                "source": self.metadata.source,
                "page": self.metadata.page_number,
                "total_pages": self.metadata.total_pages,
                "file_name": self.metadata.file_name,
                **self.metadata.extra
            }
        )


class PDFLoader:
    """
    PDF document loader with metadata extraction.
    
    This class handles loading PDF files and extracting text content
    along with page-level metadata for citation purposes.
    
    Attributes:
        supported_extensions: List of supported file extensions
    
    Example:
        >>> loader = PDFLoader()
        >>> documents = loader.load("document.pdf")
        >>> for doc in documents:
        ...     print(f"Page {doc.metadata.page_number}: {doc.content[:100]}...")
    """
    
    supported_extensions = [".pdf"]
    
    def __init__(self):
        """Initialize the PDF loader."""
        logger.info("PDFLoader initialized")
    
    def load(self, file_path: Union[str, Path]) -> list[LoadedDocument]:
        """
        Load a PDF file and extract text from all pages.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LoadedDocument objects, one per page
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        # Validate file extension
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {self.supported_extensions}"
            )
        
        logger.info(f"Loading PDF: {path}")
        
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            
            total_pages = len(pages)
            documents = []
            
            for page in pages:
                # Extract page number (PyPDFLoader uses 0-indexed pages)
                page_num = page.metadata.get("page", 0) + 1
                
                doc = LoadedDocument(
                    content=page.page_content,
                    metadata=DocumentMetadata(
                        source=str(path),
                        page_number=page_num,
                        total_pages=total_pages,
                        file_name=path.name
                    )
                )
                documents.append(doc)
            
            logger.info(f"Loaded {total_pages} pages from {path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {path}: {e}")
            raise
    
    def load_from_bytes(
        self, 
        file_bytes: BinaryIO, 
        file_name: str,
        temp_dir: Union[str, Path] = None
    ) -> list[LoadedDocument]:
        """
        Load a PDF from bytes (e.g., from file upload).
        
        Args:
            file_bytes: File-like object containing PDF data
            file_name: Original filename for metadata
            temp_dir: Directory to store temporary file (optional)
            
        Returns:
            List of LoadedDocument objects, one per page
        """
        import tempfile
        
        # Create temporary file
        if temp_dir:
            temp_path = Path(temp_dir) / file_name
        else:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=".pdf"
            )
            temp_path = Path(temp_file.name)
            temp_file.close()
        
        try:
            # Write bytes to temporary file
            with open(temp_path, "wb") as f:
                f.write(file_bytes.read())
            
            # Load using standard method
            documents = self.load(temp_path)
            
            # Update metadata to use original filename
            for doc in documents:
                doc.metadata.file_name = file_name
                doc.metadata.source = file_name
            
            return documents
            
        finally:
            # Clean up temporary file if we created it
            if not temp_dir and temp_path.exists():
                temp_path.unlink()
    
    def load_multiple(
        self, 
        file_paths: list[Union[str, Path]]
    ) -> dict[str, list[LoadedDocument]]:
        """
        Load multiple PDF files.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping file names to their documents
        """
        results = {}
        
        for path in file_paths:
            try:
                documents = self.load(path)
                results[Path(path).name] = documents
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                results[Path(path).name] = []
        
        return results
    
    @staticmethod
    def get_text_statistics(documents: list[LoadedDocument]) -> dict:
        """
        Get statistics about loaded documents.
        
        Args:
            documents: List of LoadedDocument objects
            
        Returns:
            Dictionary with text statistics
        """
        if not documents:
            return {
                "total_pages": 0,
                "total_characters": 0,
                "total_words": 0,
                "avg_chars_per_page": 0,
                "avg_words_per_page": 0
            }
        
        total_chars = sum(len(doc.content) for doc in documents)
        total_words = sum(len(doc.content.split()) for doc in documents)
        num_pages = len(documents)
        
        return {
            "total_pages": num_pages,
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_page": total_chars // num_pages,
            "avg_words_per_page": total_words // num_pages
        }
