"""File reading utilities with error handling."""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def read_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """Read file content with error handling.
    
    Args:
        file_path: Path to the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File content as string, or None if error occurs
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if not path.is_file():
            logger.error(f"Not a file: {file_path}")
            return None
        
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        logger.debug(f"Successfully read {len(content)} characters from {file_path}")
        return content
        
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {file_path}: {e}")
        return None
    except PermissionError as e:
        logger.error(f"Permission denied reading {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {e}")
        return None


def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """Write content to file with error handling.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding (default: utf-8)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.debug(f"Successfully wrote {len(content)} characters to {file_path}")
        return True
        
    except PermissionError as e:
        logger.error(f"Permission denied writing {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing {file_path}: {e}")
        return False


def ensure_directory(dir_path: str) -> bool:
    """Ensure directory exists, create if needed.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes, or None if error occurs
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        size = path.stat().st_size
        logger.debug(f"File size of {file_path}: {size} bytes")
        return size
        
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return None
