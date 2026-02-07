"""Data models for LLM-assisted fuzzer simulator."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING, Protocol
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from typing import Self


class TestType(Enum):
    """Types of test cases."""
    NORMAL = "normal"
    EDGE = "edge"
    SECURITY = "security"
    MALFORMED = "malformed"


class DefectType(Enum):
    """Types of defects that can be detected."""
    SECURITY = "security"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"


class DefectSeverity(Enum):
    """Severity levels for defects."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Constants for validation
class ValidationConstants:
    """Constants used in validation."""
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    VALID_REPORT_FORMATS = {"json", "csv", "yaml"}
    VALID_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    RATIO_TOLERANCE = 0.01  # Tolerance for floating point ratio comparisons


class Validatable(Protocol):
    """Protocol for objects that can be validated."""
    
    def validate(self) -> 'ValidationResult':
        """Validate the object and return validation result."""
        ...


@dataclass
class ValidationResult:
    """Result of validation with errors, warnings, and confidence score."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult', prefix: str = "") -> None:
        """Merge another validation result into this one."""
        prefix_str = f"{prefix}: " if prefix else ""
        self.errors.extend([f"{prefix_str}{error}" for error in other.errors])
        self.warnings.extend([f"{prefix_str}{warning}" for warning in other.warnings])
        if not other.is_valid:
            self.is_valid = False
        self.confidence = min(self.confidence, other.confidence)


class ValidationMixin:
    """Mixin class providing common validation utilities."""
    
    @staticmethod
    def _validate_non_empty_string(value: Optional[str], field_name: str) -> List[str]:
        """Validate that a string field is not empty."""
        errors = []
        if not value or not value.strip():
            errors.append(f"{field_name} cannot be empty")
        return errors
    
    @staticmethod
    def _validate_range(value: float, min_val: float, max_val: float, field_name: str) -> List[str]:
        """Validate that a numeric value is within range."""
        errors = []
        if not (min_val <= value <= max_val):
            errors.append(f"{field_name} must be between {min_val} and {max_val}")
        return errors
    
    @staticmethod
    def _validate_positive(value: float, field_name: str) -> List[str]:
        """Validate that a numeric value is positive."""
        errors = []
        if value <= 0:
            errors.append(f"{field_name} must be positive")
        return errors


@dataclass
class ParameterSpec(ValidationMixin):
    """Specification for an API parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    constraints: List[str] = field(default_factory=list)
    examples: List[Any] = field(default_factory=list)
    default_value: Optional[Any] = None
    
    def __str__(self) -> str:
        """String representation of parameter specification."""
        req_str = "required" if self.required else "optional"
        default_str = f" = {self.default_value}" if self.default_value is not None else ""
        return f"{self.name}: {self.type} ({req_str}){default_str}"
    
    def validate(self) -> ValidationResult:
        """Validate parameter specification."""
        result = ValidationResult(is_valid=True)
        
        # Validate required string fields
        for error in self._validate_non_empty_string(self.name, "Parameter name"):
            result.add_error(error)
        
        for error in self._validate_non_empty_string(self.type, "Parameter type"):
            result.add_error(error)
        
        # Validate description (warning only)
        if not self.description or not self.description.strip():
            result.add_warning("Parameter description is empty")
        
        # Validate logical consistency
        if self.required and self.default_value is not None:
            result.add_warning("Required parameter has default value")
        
        # Validate parameter name format (should be valid identifier)
        if self.name and not self.name.replace('_', '').isalnum():
            result.add_warning("Parameter name should be a valid identifier")
        
        # Set confidence based on validation results
        result.confidence = 1.0 if result.is_valid else 0.0
        
        return result


@dataclass
class APISpec(ValidationMixin):
    """API specification extracted from documentation."""
    name: str
    description: str
    parameters: List[ParameterSpec] = field(default_factory=list)
    return_type: str = "Any"
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None
    method: str = "POST"
    
    def __str__(self) -> str:
        """String representation of API specification."""
        param_str = f"({len(self.parameters)} params)" if self.parameters else "(no params)"
        endpoint_str = f" @ {self.endpoint}" if self.endpoint else ""
        return f"{self.method} {self.name}{param_str}{endpoint_str}"
    
    def validate(self) -> ValidationResult:
        """Validate API specification."""
        result = ValidationResult(is_valid=True)
        
        # Validate required string fields
        for error in self._validate_non_empty_string(self.name, "API name"):
            result.add_error(error)
        
        # Validate description (warning only)
        if not self.description or not self.description.strip():
            result.add_warning("API description is empty")
        
        # Validate HTTP method
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if self.method.upper() not in valid_methods:
            result.add_warning(f"Unusual HTTP method: {self.method}")
        
        # Validate parameters
        for i, param in enumerate(self.parameters):
            param_validation = param.validate()
            result.merge(param_validation, f"Parameter {i} ({param.name})")
        
        # Check for duplicate parameter names
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            result.add_error("Duplicate parameter names found")
        
        # Set confidence based on validation results
        result.confidence = 1.0 if result.is_valid else 0.5
        
        return result


@dataclass
class TestCase:
    """Generated test case for API testing."""
    id: str
    api_name: str
    parameters: Dict[str, Any]
    test_type: TestType
    expected_result: Optional[str] = None
    generation_prompt: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> ValidationResult:
        """Validate test case."""
        errors = []
        warnings = []
        
        if not self.id or not self.id.strip():
            errors.append("Test case ID cannot be empty")
        
        if not self.api_name or not self.api_name.strip():
            errors.append("API name cannot be empty")
        
        if not isinstance(self.parameters, dict):
            errors.append("Parameters must be a dictionary")
        
        if not isinstance(self.test_type, TestType):
            errors.append("Test type must be a valid TestType enum")
        
        if not self.generation_prompt or not self.generation_prompt.strip():
            warnings.append("Generation prompt is empty")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else 0.0
        )


@dataclass
class CoverageMetrics:
    """Code coverage information."""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    api_endpoint_coverage: float = 0.0
    covered_lines: Set[int] = field(default_factory=set)
    total_lines: int = 0
    covered_branches: Set[str] = field(default_factory=set)
    total_branches: int = 0


@dataclass
class Defect:
    """Information about a detected defect."""
    id: str
    type: DefectType
    severity: DefectSeverity
    description: str
    trigger_test_case: str
    location: Optional[str] = None
    impact_assessment: str = ""
    stack_trace: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of executing a single test case."""
    test_case_id: str
    success: bool
    response: Any = None
    execution_time: float = 0.0
    coverage_delta: Optional[CoverageMetrics] = None
    errors: List[str] = field(default_factory=list)
    detected_defects: List[Defect] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)



@dataclass
class PerformanceMetrics:
    """Performance metrics for fuzzing session."""
    total_execution_time: float = 0.0
    average_test_time: float = 0.0
    llm_generation_time: float = 0.0
    test_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    tests_per_second: float = 0.0


@dataclass
class FeedbackReport:
    """Comprehensive feedback from a fuzzing session."""
    session_id: str
    document_hash: str
    total_test_cases: int
    validity_rate: float
    coverage_metrics: CoverageMetrics
    defects_found: List[Defect]
    performance_metrics: PerformanceMetrics
    api_specs: List[APISpec] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "document_hash": self.document_hash,
            "total_test_cases": self.total_test_cases,
            "validity_rate": self.validity_rate,
            "coverage_metrics": self.coverage_metrics.__dict__,
            "defects_found": [d.__dict__ for d in self.defects_found],
            "performance_metrics": self.performance_metrics.__dict__,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    
    def get_validity_rate(self) -> float:
        """Calculate validity rate from execution results."""
        if not self.execution_results:
            return 0.0
        
        successful_tests = sum(1 for result in self.execution_results if result.success)
        return successful_tests / len(self.execution_results)
    
    def get_coverage_percentage(self) -> float:
        """Get overall coverage percentage."""
        return self.coverage_metrics.line_coverage
    
    def get_defect_count(self) -> int:
        """Get total number of defects found."""
        return len(self.defects_found)

    def get_crash_count(self) -> int:
        """Get number of failed test executions."""
        return sum(1 for result in self.execution_results if not result.success)
    
    def get_security_defect_count(self) -> int:
        """Get number of security defects found."""
        return sum(1 for defect in self.defects_found 
                  if defect.type == DefectType.SECURITY)
    
    def get_defects_by_severity(self, severity: DefectSeverity) -> List[Defect]:
        """Get defects filtered by severity level."""
        return [defect for defect in self.defects_found if defect.severity == severity]
    
    def get_critical_defects(self) -> List[Defect]:
        """Get all critical defects."""
        return self.get_defects_by_severity(DefectSeverity.CRITICAL)
    
    def get_test_success_rate(self) -> float:
        """Get the success rate of test execution (alias for validity rate)."""
        return self.get_validity_rate()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        return {
            'total_time': self.performance_metrics.total_execution_time,
            'avg_test_time': self.performance_metrics.average_test_time,
            'throughput': self.performance_metrics.tests_per_second,
            'memory_usage_mb': self.performance_metrics.memory_usage_mb
        }
    
    def is_successful_session(self, min_validity: float = 0.8, min_coverage: float = 50.0) -> bool:
        """Check if the fuzzing session meets success criteria."""
        return (
            self.get_validity_rate() >= min_validity and
            self.get_coverage_percentage() >= min_coverage and
            len(self.execution_results) > 0
        )
    
    def validate(self) -> ValidationResult:
        """Validate feedback report."""
        errors = []
        warnings = []
        
        if not self.session_id or not self.session_id.strip():
            errors.append("Session ID cannot be empty")
        
        if not self.document_hash or not self.document_hash.strip():
            errors.append("Document hash cannot be empty")
        
        if self.total_test_cases < 0:
            errors.append("Total test cases cannot be negative")
        
        if not (0.0 <= self.validity_rate <= 1.0):
            errors.append("Validity rate must be between 0.0 and 1.0")
        
        if self.total_test_cases != len(self.test_cases):
            warnings.append("Total test cases count doesn't match test cases list length")
        
        # Validate API specs
        for i, api_spec in enumerate(self.api_specs):
            api_validation = api_spec.validate()
            if not api_validation.is_valid:
                errors.extend([f"API spec {i}: {error}" for error in api_validation.errors])
        
        # Validate test cases
        for i, test_case in enumerate(self.test_cases):
            test_validation = test_case.validate()
            if not test_validation.is_valid:
                errors.extend([f"Test case {i}: {error}" for error in test_validation.errors])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else 0.7
        )


@dataclass
class FuzzerConfig(ValidationMixin):
    """Configuration for LLM fuzzer simulator."""
    
    # LLM Configuration
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_timeout: int = 30
    llm_api_key: Optional[str] = None

    # Document summarization (optional)
    summary_enabled: bool = False
    summary_model: str = "gpt-4o"
    summary_temperature: float = 0.2
    summary_max_tokens: int = 800
    summary_timeout: int = 60
    summary_api_key: Optional[str] = None
    summary_endpoint: Optional[str] = None
    
    # Test Generation
    cases_per_api: int = 20
    security_test_ratio: float = 0.3
    edge_case_ratio: float = 0.2
    normal_case_ratio: float = 0.5

    # Document-based generation
    document_generation_enabled: bool = False
    document_generation_language: str = "java"
    cases_per_document: int = 20
    document_prompt_trigger: str = "/* Please create a very short program which uses new Java features in a complex way */"
    document_prompt_hint: str = "import java.lang.Object;"
    document_generation_case_file: Optional[str] = None
    document_generation_case_mode: str = "fixed"
    document_generation_max_attempts: int = 60
    document_generation_max_seconds: int = 300
    
    # Target System Simulation
    base_error_rate: float = 0.05
    response_time_base: float = 0.1
    vulnerability_injection: bool = True
    coverage_tracking: bool = True

    # Real target execution (javac)
    target_mode: str = "simulated"
    javac_home: Optional[str] = None
    javac_source_root: Optional[str] = None
    jacoco_cli_path: Optional[str] = None
    jacoco_agent_path: Optional[str] = None
    coverage_output_dir: str = "coverage"
    coverage_scope: str = "javac"

    # Execution Parameters
    timeout_per_test: float = 10.0
    parallel_execution: bool = True
    max_workers: int = 4
    retry_failed_tests: bool = True

    
    # Reproducibility
    random_seed: Optional[int] = None

    def get_timeout_per_test(self) -> float:
        return float(self.timeout_per_test)

    
    # Output Configuration
    detailed_logs: bool = True
    save_test_cases: bool = True
    export_coverage: bool = True
    report_format: str = "json"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FuzzerConfig':
        """Create FuzzerConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
    
    def validate(self) -> ValidationResult:
        """Validate fuzzer configuration."""
        result = ValidationResult(is_valid=True)
        
        # Validate LLM configuration
        for error in self._validate_non_empty_string(self.llm_model, "LLM model"):
            result.add_error(error)
        
        for error in self._validate_range(
            self.llm_temperature, 
            ValidationConstants.MIN_TEMPERATURE, 
            ValidationConstants.MAX_TEMPERATURE, 
            "LLM temperature"
        ):
            result.add_error(error)
        
        for error in self._validate_positive(self.llm_max_tokens, "LLM max_tokens"):
            result.add_error(error)
        
        for error in self._validate_positive(self.llm_timeout, "LLM timeout"):
            result.add_error(error)
        
        if not self.llm_api_key:
            result.add_warning("No API key specified for LLM")

        if self.summary_enabled:
            for error in self._validate_non_empty_string(self.summary_model, "Summary model"):
                result.add_error(error)
            for error in self._validate_range(
                self.summary_temperature,
                ValidationConstants.MIN_TEMPERATURE,
                ValidationConstants.MAX_TEMPERATURE,
                "Summary temperature",
            ):
                result.add_error(error)
            for error in self._validate_positive(self.summary_max_tokens, "Summary max_tokens"):
                result.add_error(error)
            for error in self._validate_positive(self.summary_timeout, "Summary timeout"):
                result.add_error(error)
            if not self.summary_api_key:
                result.add_warning("No API key specified for summary model")
        
        # Validate test generation ratios
        total_ratio = self.security_test_ratio + self.edge_case_ratio + self.normal_case_ratio
        if abs(total_ratio - 1.0) > ValidationConstants.RATIO_TOLERANCE:
            result.add_warning(f"Test generation ratios sum to {total_ratio:.3f}, expected 1.0")
        
        for error in self._validate_positive(self.cases_per_api, "cases_per_api"):
            result.add_error(error)

        if self.document_generation_enabled:
            for error in self._validate_non_empty_string(
                self.document_generation_language, "document_generation_language"
            ):
                result.add_error(error)
            for error in self._validate_positive(self.cases_per_document, "cases_per_document"):
                result.add_error(error)
        
        # Validate execution parameters
        for error in self._validate_positive(self.timeout_per_test, "timeout_per_test"):
            result.add_error(error)

        if self.max_workers < 1:
            result.add_error("max_workers must be at least 1")

        # Validate target system parameters
        for error in self._validate_range(self.base_error_rate, 0.0, 1.0, "base_error_rate"):
            result.add_error(error)

        if self.response_time_base < 0:
            result.add_error("response_time_base cannot be negative")

        if self.target_mode not in {"simulated", "javac"}:
            result.add_error("target_mode must be 'simulated' or 'javac'")

        if self.target_mode == "javac":
            if not self.javac_home or not str(self.javac_home).strip():
                result.add_error("javac_home is required for javac target")
            if not self.javac_source_root or not str(self.javac_source_root).strip():
                result.add_error("javac_source_root is required for javac target")
            if not self.jacoco_cli_path or not str(self.jacoco_cli_path).strip():
                result.add_error("jacoco_cli_path is required for javac target")
            if not self.jacoco_agent_path or not str(self.jacoco_agent_path).strip():
                result.add_error("jacoco_agent_path is required for javac target")

        
        # Validate output format
        if self.report_format not in ValidationConstants.VALID_REPORT_FORMATS:
            result.add_warning(
                f"Unknown report format: {self.report_format}. "
                f"Valid formats: {ValidationConstants.VALID_REPORT_FORMATS}"
            )
        
        # Set confidence based on validation results
        result.confidence = 1.0 if result.is_valid else 0.5
        
        return result
    
    @classmethod
    def get_default_config(cls) -> 'FuzzerConfig':
        """Get default configuration with safe values."""
        return cls()  # Uses dataclass defaults
    
    @classmethod
    def get_fast_config(cls) -> 'FuzzerConfig':
        """Get configuration optimized for speed."""
        return cls(
            cases_per_api=5,
            parallel_execution=True,
            max_workers=8,
            timeout_per_test=5.0,
            vulnerability_injection=False,
            detailed_logs=False,
            save_test_cases=False
        )
    
    @classmethod
    def get_thorough_config(cls) -> 'FuzzerConfig':
        """Get configuration optimized for thoroughness."""
        return cls(
            cases_per_api=50,
            security_test_ratio=0.4,
            edge_case_ratio=0.3,
            normal_case_ratio=0.3,
            parallel_execution=False,  # Sequential for deterministic results
            timeout_per_test=30.0,
            vulnerability_injection=True,
            detailed_logs=True,
            save_test_cases=True
        )
    
    @classmethod
    def get_security_focused_config(cls) -> 'FuzzerConfig':
        """Get configuration focused on security testing."""
        return cls(
            cases_per_api=30,
            security_test_ratio=0.6,
            edge_case_ratio=0.2,
            normal_case_ratio=0.2,
            vulnerability_injection=True,
            detailed_logs=True
        )


@dataclass
class DocumentStructure:
    """Parsed structure of input documentation."""
    title: str = ""
    sections: List[Dict[str, Any]] = field(default_factory=list)
    api_specs: List[APISpec] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate document structure."""
        errors = []
        warnings = []
        
        if not self.title or not self.title.strip():
            warnings.append("Document title is empty")
        
        if not self.sections:
            warnings.append("No sections found in document")
        
        if not self.api_specs:
            warnings.append("No API specifications found in document")
        
        # Validate API specs
        for i, api_spec in enumerate(self.api_specs):
            api_validation = api_spec.validate()
            if not api_validation.is_valid:
                errors.extend([f"API spec {i}: {error}" for error in api_validation.errors])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else 0.8
        )


@dataclass
class ComparisonReport:
    """Report comparing fuzzing results between original and perturbed documents."""
    original_report: FeedbackReport
    perturbed_reports: List[FeedbackReport]
    validity_impact: float = 0.0
    coverage_impact: float = 0.0
    defect_impact: float = 0.0
    overall_impact: float = 0.0
    statistical_significance: bool = False
    
    def calculate_impacts(self) -> None:
        """Calculate impact metrics from comparison."""
        if not self.perturbed_reports:
            return
            
        # Calculate average metrics for perturbed documents
        avg_validity = sum(r.get_validity_rate() for r in self.perturbed_reports) / len(self.perturbed_reports)
        avg_coverage = sum(r.get_coverage_percentage() for r in self.perturbed_reports) / len(self.perturbed_reports)
        avg_defects = sum(r.get_defect_count() for r in self.perturbed_reports) / len(self.perturbed_reports)
        
        # Calculate impacts (negative means degradation)
        self.validity_impact = (avg_validity - self.original_report.get_validity_rate()) / self.original_report.get_validity_rate()
        self.coverage_impact = (avg_coverage - self.original_report.get_coverage_percentage()) / self.original_report.get_coverage_percentage()
        self.defect_impact = (avg_defects - self.original_report.get_defect_count()) / max(1, self.original_report.get_defect_count())
        
        # Overall impact (weighted average)
        self.overall_impact = (
            0.4 * abs(self.validity_impact) +
            0.35 * abs(self.coverage_impact) +
            0.25 * abs(self.defect_impact)
        )
