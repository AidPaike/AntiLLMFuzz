"""Base class for token extractors."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import logging

from src.data_models import Token

logger = logging.getLogger(__name__)


class BaseTokenExtractor(ABC):
    """Abstract base class for token extraction from various sources.
    
    This class defines the interface that all token extractors must implement.
    Each language or document type should have its own extractor that inherits
    from this base class.
    """
    
    def __init__(self, language: str, supported_extensions: List[str]):
        """Initialize the base extractor.
        
        Args:
            language: Language or document type name (e.g., 'java', 'python', 'markdown')
            supported_extensions: List of supported file extensions (e.g., ['.java', '.py'])
        """
        self.language = language
        self.supported_extensions = supported_extensions
        logger.debug(f"Initialized {language} extractor with extensions: {supported_extensions}")
    
    def can_extract(self, file_path: str) -> bool:
        """Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this extractor supports the file type
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        can_handle = extension in self.supported_extensions
        
        if can_handle:
            logger.debug(f"{self.language} extractor can handle {file_path}")
        
        return can_handle
    
    def validate_file(self, file_path: str) -> bool:
        """Validate that the file exists and is readable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid and readable
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if not path.is_file():
            logger.error(f"Not a file: {file_path}")
            return False
        
        if not self.can_extract(file_path):
            logger.warning(f"File extension not supported by {self.language} extractor: {file_path}")
            return False
        
        return True
    
    @abstractmethod
    def extract_tokens(self, file_path: str) -> List[Token]:
        """Extract tokens from the given file.
        
        This is the main method that must be implemented by all subclasses.
        
        Args:
            file_path: Path to the file to extract tokens from
            
        Returns:
            List of Token objects extracted from the file
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def parse_content(self, content: str, source_file: str) -> List[Token]:
        """Parse content and extract tokens.
        
        This method should be implemented to parse the actual content.
        
        Args:
            content: File content as string
            source_file: Source file path for token metadata
            
        Returns:
            List of Token objects
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def get_statistics(self, tokens: List[Token]) -> dict:
        """Get statistics about extracted tokens.
        
        Args:
            tokens: List of extracted tokens
            
        Returns:
            Dictionary with token statistics
        """
        stats = {
            'total_tokens': len(tokens),
            'by_type': {},
            'language': self.language
        }
        
        # Count by token type
        for token in tokens:
            token_type = token.token_type
            stats['by_type'][token_type] = stats['by_type'].get(token_type, 0) + 1
        
        return stats
    
    def __str__(self) -> str:
        return f"{self.language.capitalize()}TokenExtractor(extensions={self.supported_extensions})"
    
    def __repr__(self) -> str:
        return self.__str__()
