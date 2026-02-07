"""Base interfaces and abstract classes for LLM fuzzer simulator components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.fuzzer.data_models import (
    DocumentStructure, APISpec, TestCase, ExecutionResult, 
    FeedbackReport, CoverageMetrics, ValidationResult, FuzzerConfig
)


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def parse_document(self, content: str) -> DocumentStructure:
        """Parse document content into structured format.
        
        Args:
            content: Raw document content
            
        Returns:
            DocumentStructure with parsed information
        """
        pass
    
    @abstractmethod
    def extract_api_specs(self, structure: DocumentStructure) -> List[APISpec]:
        """Extract API specifications from document structure.
        
        Args:
            structure: Parsed document structure
            
        Returns:
            List of APISpec objects
        """
        pass
    
    @abstractmethod
    def identify_functions(self, content: str) -> List[str]:
        """Identify function names in content.
        
        Args:
            content: Content to search
            
        Returns:
            List of function names
        """
        pass


class BaseTestGenerator(ABC):
    """Abstract base class for test case generators."""
    
    @abstractmethod
    def generate_test_cases(self, api_specs: List[APISpec], count: int) -> List[TestCase]:
        """Generate test cases for given API specifications.
        
        Args:
            api_specs: List of API specifications
            count: Number of test cases to generate
            
        Returns:
            List of generated test cases
        """
        pass
    
    @abstractmethod
    def validate_test_case(self, test_case: TestCase) -> ValidationResult:
        """Validate a test case for correctness.
        
        Args:
            test_case: Test case to validate
            
        Returns:
            ValidationResult with validation status
        """
        pass


class BaseTargetSystem(ABC):
    """Abstract base class for target system simulators."""
    
    @abstractmethod
    def execute_test(self, test_case: TestCase) -> ExecutionResult:
        """Execute a test case against the target system.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            ExecutionResult with test results
        """
        pass
    
    @abstractmethod
    def get_coverage_info(self) -> CoverageMetrics:
        """Get current coverage information.
        
        Returns:
            CoverageMetrics with current coverage data
        """
        pass
    
    @abstractmethod
    def reset_state(self) -> None:
        """Reset target system state for new test session."""
        pass


class BaseTestExecutor(ABC):
    """Abstract base class for test executors."""
    
    @abstractmethod
    def execute_test_suite(self, test_cases: List[TestCase]) -> List[ExecutionResult]:
        """Execute a suite of test cases.
        
        Args:
            test_cases: List of test cases to execute
            
        Returns:
            List of execution results
        """
        pass
    
    @abstractmethod
    def execute_single_test(self, test_case: TestCase) -> ExecutionResult:
        """Execute a single test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            ExecutionResult with test results
        """
        pass


class BaseFeedbackCollector(ABC):
    """Abstract base class for feedback collectors."""
    
    @abstractmethod
    def collect_feedback(self, execution_results: List[ExecutionResult]) -> FeedbackReport:
        """Collect and analyze feedback from execution results.
        
        Args:
            execution_results: List of test execution results
            
        Returns:
            FeedbackReport with analysis results
        """
        pass
    
    @abstractmethod
    def calculate_validity_rate(self, results: List[ExecutionResult]) -> float:
        """Calculate validity rate from execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            Validity rate as float between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def measure_coverage(self, results: List[ExecutionResult]) -> CoverageMetrics:
        """Measure code coverage from execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            CoverageMetrics with coverage information
        """
        pass


class BaseFuzzerSimulator(ABC):
    """Abstract base class for fuzzer simulators."""
    
    def __init__(self, config: FuzzerConfig):
        """Initialize fuzzer simulator with configuration.
        
        Args:
            config: Fuzzer configuration
        """
        self.config = config
    
    @abstractmethod
    def run_fuzzing_session(self, document_content: str) -> FeedbackReport:
        """Run a complete fuzzing session on document.
        
        Args:
            document_content: Input document content
            
        Returns:
            FeedbackReport with session results
        """
        pass
    
    @abstractmethod
    def process_document(self, content: str) -> DocumentStructure:
        """Process document to extract structure and APIs.
        
        Args:
            content: Document content
            
        Returns:
            DocumentStructure with parsed information
        """
        pass
    
    @abstractmethod
    def generate_tests(self, api_specs: List[APISpec]) -> List[TestCase]:
        """Generate test cases for API specifications.
        
        Args:
            api_specs: List of API specifications
            
        Returns:
            List of generated test cases
        """
        pass
    
    @abstractmethod
    def execute_tests(self, test_cases: List[TestCase]) -> List[ExecutionResult]:
        """Execute test cases against target system.
        
        Args:
            test_cases: List of test cases to execute
            
        Returns:
            List of execution results
        """
        pass


class BaseVulnerabilityInjector(ABC):
    """Abstract base class for vulnerability injectors."""
    
    @abstractmethod
    def inject_vulnerability(self, vuln_type: str, trigger_condition: str) -> None:
        """Inject a vulnerability into the target system.
        
        Args:
            vuln_type: Type of vulnerability to inject
            trigger_condition: Condition that triggers the vulnerability
        """
        pass
    
    @abstractmethod
    def remove_vulnerability(self, vuln_id: str) -> None:
        """Remove a previously injected vulnerability.
        
        Args:
            vuln_id: ID of vulnerability to remove
        """
        pass
    
    @abstractmethod
    def list_vulnerabilities(self) -> List[Dict[str, Any]]:
        """List all currently injected vulnerabilities.
        
        Returns:
            List of vulnerability information dictionaries
        """
        pass


class BasePerformanceSimulator(ABC):
    """Abstract base class for performance simulators."""
    
    @abstractmethod
    def simulate_llm_generation_time(self, prompt_length: int, model: str) -> float:
        """Simulate LLM test generation time.
        
        Args:
            prompt_length: Length of generation prompt
            model: LLM model being used
            
        Returns:
            Simulated generation time in seconds
        """
        pass
    
    @abstractmethod
    def simulate_test_execution_time(self, test_complexity: float) -> float:
        """Simulate test case execution time.
        
        Args:
            test_complexity: Complexity factor of the test
            
        Returns:
            Simulated execution time in seconds
        """
        pass
    
    @abstractmethod
    def simulate_response_time(self, api_complexity: float, load_factor: float) -> float:
        """Simulate API response time.
        
        Args:
            api_complexity: Complexity of the API call
            load_factor: System load factor
            
        Returns:
            Simulated response time in seconds
        """
        pass


class BaseRandomSeedManager(ABC):
    """Abstract base class for random seed management."""
    
    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results.
        
        Args:
            seed: Random seed value
        """
        pass
    
    @abstractmethod
    def get_seed(self) -> Optional[int]:
        """Get current random seed.
        
        Returns:
            Current seed value or None if not set
        """
        pass
    
    @abstractmethod
    def generate_deterministic_value(self, key: str) -> float:
        """Generate deterministic random value for given key.
        
        Args:
            key: Key for deterministic generation
            
        Returns:
            Deterministic random value between 0.0 and 1.0
        """
        pass


class BaseConfigurationValidator(ABC):
    """Abstract base class for configuration validators."""
    
    @abstractmethod
    def validate_config(self, config: FuzzerConfig) -> ValidationResult:
        """Validate fuzzer configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation status
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> FuzzerConfig:
        """Get default configuration with safe values.
        
        Returns:
            Default FuzzerConfig instance
        """
        pass
    
    @abstractmethod
    def merge_configs(self, base_config: FuzzerConfig, override_config: Dict[str, Any]) -> FuzzerConfig:
        """Merge configuration with overrides.
        
        Args:
            base_config: Base configuration
            override_config: Configuration overrides
            
        Returns:
            Merged FuzzerConfig instance
        """
        pass


# Component factory interface
class ComponentFactory(ABC):
    """Abstract factory for creating fuzzer components."""
    
    @abstractmethod
    def create_document_processor(self) -> BaseDocumentProcessor:
        """Create document processor instance."""
        pass
    
    @abstractmethod
    def create_test_generator(self, config: FuzzerConfig) -> BaseTestGenerator:
        """Create test generator instance."""
        pass
    
    @abstractmethod
    def create_target_system(self, config: FuzzerConfig) -> BaseTargetSystem:
        """Create target system simulator instance."""
        pass
    
    @abstractmethod
    def create_test_executor(self, config: FuzzerConfig) -> BaseTestExecutor:
        """Create test executor instance."""
        pass
    
    @abstractmethod
    def create_feedback_collector(self, config: FuzzerConfig) -> BaseFeedbackCollector:
        """Create feedback collector instance."""
        pass
    
    @abstractmethod
    def create_fuzzer_simulator(self, config: FuzzerConfig) -> BaseFuzzerSimulator:
        """Create complete fuzzer simulator instance."""
        pass


# Exception classes for fuzzer components
class FuzzerError(Exception):
    """Base exception for fuzzer errors."""
    pass


class DocumentProcessingError(FuzzerError):
    """Error during document processing."""
    pass


class TestGenerationError(FuzzerError):
    """Error during test case generation."""
    pass


class TestExecutionError(FuzzerError):
    """Error during test execution."""
    pass


class TargetSystemError(FuzzerError):
    """Error in target system simulation."""
    pass


class ConfigurationError(FuzzerError):
    """Error in configuration."""
    pass


class ValidationError(FuzzerError):
    """Error during validation."""
    pass