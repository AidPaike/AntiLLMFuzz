"""LLM-Assisted Fuzzer Simulator Module.

This module provides a comprehensive simulation of LLM-assisted fuzzing tools
like Fuzz4All. It processes documentation/code, uses LLM to generate test cases,
executes them against simulated target systems, and collects detailed feedback.

The module follows the project's architecture patterns:
- Strategy Pattern for different fuzzing approaches
- Extractor Pattern for document processing
- Comprehensive data models for type safety
"""

__version__ = "1.0.0"

# Data Models
from src.fuzzer.data_models import (
    APISpec,
    ParameterSpec,
    TestCase,
    ExecutionResult,
    FeedbackReport,
    CoverageMetrics,
    Defect,
    FuzzerConfig,
    DocumentStructure,
    ValidationResult,
    ComparisonReport
)

# Base Interfaces
from src.fuzzer.base_interfaces import (
    BaseDocumentProcessor,
    BaseTestGenerator,
    BaseTargetSystem,
    BaseTestExecutor,
    BaseFeedbackCollector,
    BaseFuzzerSimulator,
    ComponentFactory,
    FuzzerError,
    DocumentProcessingError,
    TestGenerationError,
    TestExecutionError,
    TargetSystemError,
    ConfigurationError,
    ValidationError
)

# Configuration Management
from src.fuzzer.config_manager import (
    FuzzerConfigManager,
    get_fuzzer_config_manager,
    get_fuzzer_config
)

# Core Components
from src.fuzzer.document_processor import DocumentProcessor
from src.fuzzer.llm_test_generator import LLMTestGenerator
from src.fuzzer.target_system_simulator import TargetSystemSimulator
from src.fuzzer.test_executor import TestExecutor
from src.fuzzer.feedback_collector import FeedbackCollector

# Main Simulator
from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator

# High-level API
from src.fuzzer.api import (
    FuzzerAPI,
    WorkflowBuilder,
    create_fuzzer_api,
    quick_fuzz,
    quick_compare,
    quick_batch_fuzz
)

# Integration Helpers
from src.fuzzer.integration import (
    PerturbationFuzzerIntegrator,
    LLMFeedbackSimulator
)

# Convenience function for quick setup
def create_fuzzer_simulator(config_dict: dict = None) -> LLMFuzzerSimulator:
    """Create a fuzzer simulator with optional configuration.
    
    Args:
        config_dict: Optional configuration dictionary
        
    Returns:
        Configured LLMFuzzerSimulator instance
        
    Example:
        >>> fuzzer = create_fuzzer_simulator({'cases_per_api': 10})
        >>> report = fuzzer.run_fuzzing_session(document_content)
    """
    if config_dict:
        return LLMFuzzerSimulator.from_config_dict(config_dict)
    return LLMFuzzerSimulator()

__all__ = [
    # Version
    '__version__',
    
    # Data Models
    'APISpec',
    'ParameterSpec', 
    'TestCase',
    'ExecutionResult',
    'FeedbackReport',
    'CoverageMetrics',
    'Defect',
    'FuzzerConfig',
    'DocumentStructure',
    'ValidationResult',
    'ComparisonReport',
    
    # Base Interfaces
    'BaseDocumentProcessor',
    'BaseTestGenerator',
    'BaseTargetSystem',
    'BaseTestExecutor',
    'BaseFeedbackCollector',
    'BaseFuzzerSimulator',
    'ComponentFactory',
    
    # Exceptions
    'FuzzerError',
    'DocumentProcessingError',
    'TestGenerationError',
    'TestExecutionError',
    'TargetSystemError',
    'ConfigurationError',
    'ValidationError',
    
    # Configuration Management
    'FuzzerConfigManager',
    'get_fuzzer_config_manager',
    'get_fuzzer_config',
    
    # Core Components
    'DocumentProcessor',
    'LLMTestGenerator',
    'TargetSystemSimulator',
    'TestExecutor',
    'FeedbackCollector',
    
    # Main Simulator
    'LLMFuzzerSimulator',
    
    # High-level API
    'FuzzerAPI',
    'WorkflowBuilder',
    'create_fuzzer_api',
    'quick_fuzz',
    'quick_compare',
    'quick_batch_fuzz',
    
    # Integration Helpers
    'PerturbationFuzzerIntegrator',
    'LLMFeedbackSimulator',
    
    # Convenience Functions
    'create_fuzzer_simulator'
]