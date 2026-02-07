"""Focused constants organization following project guidelines."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DisplayConstants:
    """Display and formatting constants."""
    MAX_TOKEN_DISPLAY_LENGTH: int = 50
    SEPARATOR_LENGTH: int = 80
    JSON_INDENT: int = 2


@dataclass(frozen=True)
class ValidationLimits:
    """Validation thresholds and limits."""
    MAX_TOP_N: int = 100
    MIN_TOP_N: int = 1
    MAX_TOKENS_FOR_PROCESSING: int = 10000
    LARGE_FILE_SIZE_MB: int = 10
    MAX_METHOD_LINES: int = 25
    MAX_CLASS_LINES: int = 300


@dataclass(frozen=True)
class ApplicationDefaults:
    """Default configuration values."""
    INPUT_FILE: str = "data/00java_std.md"
    OUTPUT_DIR: str = "output"
    TOP_N: int = 5
    STRATEGY: str = "tokenization_drift"
    LOG_LEVEL: str = "INFO"


@dataclass(frozen=True)
class ApplicationMetadata:
    """Application metadata and system constants."""
    VERSION: str = "LLM Fuzzer Semantic Disruptor v1.0.0"
    EXIT_SUCCESS: int = 0
    EXIT_FAILURE: int = 1
    
    @property
    def supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.md', '.txt', '.rst', '.adoc', '.java', '.py']
    
    @property
    def pipeline_steps(self) -> List[str]:
        """Get pipeline step names."""
        return [
            "Selecting token extractor",
            "Extracting tokens", 
            "Prioritizing tokens",
            "Initializing perturbation strategies",
            "Reading original content",
            "Creating output directory",
            "Applying perturbations",
            "Generating metadata report"
        ]


# Singleton instances for easy access
DISPLAY = DisplayConstants()
VALIDATION = ValidationLimits()
DEFAULTS = ApplicationDefaults()
APP = ApplicationMetadata()