"""Logging utility for the LLM Fuzzer Semantic Disruptor."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any



class Logger:
    """Centralized logging utility."""
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "fuzzer_disruptor", 
                   level: str = "INFO",
                   log_to_file: bool = True,
                   log_dir: str = "logs") -> logging.Logger:
        """Get or create a logger instance.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_dir: Directory for log files
            
        Returns:
            Configured logger instance
        """
        if cls._instance is not None:
            return cls._instance
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"fuzzer_disruptor_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        
        cls._instance = logger
        return logger
    
    @classmethod
    def reset(cls):
        """Reset the logger instance."""
        cls._instance = None


    @classmethod
    def format_token(cls, token: Any, max_length: int = 80) -> str:
        """Format a token for logging."""
        if token is None:
            return "<none>"
        text = getattr(token, "text", "")
        token_type = getattr(token, "token_type", "")
        line = getattr(token, "line", None)
        column = getattr(token, "column", None)
        score = getattr(token, "priority_score", None)
        scs_score = getattr(token, "scs_score", None)
        source = getattr(token, "source_file", None)

        short_text = text.replace("\n", " ")
        if len(short_text) > max_length:
            short_text = short_text[:max_length] + "..."

        parts = [short_text]
        if token_type:
            parts.append(f"type={token_type}")
        if line is not None:
            parts.append(f"line={line}")
        if column is not None:
            parts.append(f"col={column}")
        if score is not None:
            parts.append(f"priority={score:.2f}")
        if scs_score is not None:
            parts.append(f"scs={scs_score:.2f}")
        if source:
            parts.append(f"src={source}")

        return " | ".join(parts)


    @classmethod
    def format_strategy_summary(
        cls,
        strategy: Any,
        targets: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a strategy summary for logging."""
        if strategy is None:
            return "<no strategy>"
        name = getattr(strategy, "name", "unknown")
        category = getattr(strategy, "category", "unknown")
        supported_targets = getattr(strategy, "supported_targets", None)
        supported_languages = getattr(strategy, "supported_languages", None)
        code_safety = getattr(strategy, "code_safety", None)

        parts = [f"{name} ({category})"]
        if supported_targets:
            parts.append(f"targets={sorted(list(supported_targets))}")
        if supported_languages:
            parts.append(f"languages={sorted(list(supported_languages))}")
        if code_safety:
            parts.append(f"code_safety={code_safety}")
        if targets:
            parts.append(
                "context=" + ",".join(f"{k}={v}" for k, v in targets.items() if v is not None)
            )

        return " | ".join(parts)


    @classmethod
    def format_scs_summary(cls, stats: Dict[str, Any]) -> str:
        """Format SCS statistics for logging."""
        if not stats:
            return "SCS stats unavailable"
        return (
            f"mean={stats.get('mean', 0):.2f} "
            f"median={stats.get('median', 0):.2f} "
            f"max={stats.get('max', 0):.2f} "
            f"min={stats.get('min', 0):.2f} "
            f"std={stats.get('std_dev', 0):.2f}"
        )


def get_logger(name: str = "fuzzer_disruptor", **kwargs) -> logging.Logger:

    """Convenience function to get logger.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for Logger.get_logger()
        
    Returns:
        Logger instance
    """
    return Logger.get_logger(name, **kwargs)
