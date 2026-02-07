"""File utility functions."""

from pathlib import Path
from datetime import datetime
from typing import Optional


def create_output_directory(base_dir: str = "output", 
                           prefix: str = "perturbations") -> Path:
    """Create timestamped output directory.
    
    Args:
        base_dir: Base output directory
        prefix: Directory name prefix
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use in filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of sanitized text
        
    Returns:
        Sanitized filename-safe text
    """
    # Remove newlines and special characters
    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    clean_text = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in clean_text)
    
    # Truncate to max length
    if len(clean_text) > max_length:
        clean_text = clean_text[:max_length]
    
    return clean_text


def generate_output_filename(base_name: str, 
                            strategy_name: str,
                            timestamp: Optional[str] = None,
                            extension: str = ".md") -> str:
    """Generate output filename with strategy and timestamp.
    
    Args:
        base_name: Base filename (without extension)
        strategy_name: Perturbation strategy name
        timestamp: Optional timestamp (generated if not provided)
        extension: File extension
        
    Returns:
        Generated filename
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    clean_strategy = sanitize_filename(strategy_name)
    return f"{base_name}_{clean_strategy}_{timestamp}{extension}"
