"""Target system simulator for realistic API behavior simulation."""

import random
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.fuzzer.base_interfaces import BaseTargetSystem, BaseVulnerabilityInjector, BasePerformanceSimulator
from src.fuzzer.data_models import (
    TestCase, ExecutionResult, CoverageMetrics, Defect, DefectType, DefectSeverity,
    TestType, FuzzerConfig, ValidationResult
)
from src.fuzzer.vulnerability_injector import AdvancedVulnerabilityInjector
from src.fuzzer.coverage_tracker import AdvancedCoverageTracker
from src.fuzzer.random_seed_manager import get_seed_manager, RandomSeedManager


class VulnerabilityType(Enum):
    """Types of vulnerabilities that can be injected."""
    SQL_INJECTION = "sql_injection"
    BUFFER_OVERFLOW = "buffer_overflow"
    AUTH_BYPASS = "auth_bypass"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"


@dataclass
class InjectedVulnerability:
    """Information about an injected vulnerability."""
    id: str
    vuln_type: VulnerabilityType
    trigger_condition: str
    parameter_name: Optional[str] = None
    trigger_value: Optional[str] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class APIResponse:
    """Response from simulated API call."""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    response_time: float = 0.0
    error_message: Optional[str] = None


class PerformanceSimulator(BasePerformanceSimulator):
    """Simulator for realistic performance characteristics."""
    
    def __init__(self, config: FuzzerConfig, seed_manager: Optional[RandomSeedManager] = None):
        """Initialize performance simulator.
        
        Args:
            config: Fuzzer configuration
            seed_manager: Random seed manager for deterministic behavior
        """
        self.config = config
        self.base_response_time = config.response_time_base
        self.seed_manager = seed_manager or get_seed_manager()
        
        # Model-specific base times (seconds)
        self.llm_base_times = {
            "gpt-4": 2.5,
            "gpt-3.5-turbo": 1.2,
            "claude-3": 2.0,
            "llama-2": 1.8
        }
    
    def simulate_llm_generation_time(self, prompt_length: int, model: str) -> float:
        """Simulate LLM test generation time.
        
        Args:
            prompt_length: Length of generation prompt
            model: LLM model being used
            
        Returns:
            Simulated generation time in seconds
        """
        base_time = self.llm_base_times.get(model, 2.0)
        length_factor = prompt_length / 1000  # Scale by prompt length
        
        # Use deterministic jitter
        jitter = self.seed_manager.generate_deterministic_value(
            "performance_simulator", "llm_generation_jitter", -0.2, 0.2
        )
        
        return max(0.1, base_time + length_factor * 0.5 + jitter)
    
    def simulate_test_execution_time(self, test_complexity: float) -> float:
        """Simulate test case execution time.
        
        Args:
            test_complexity: Complexity factor of the test (0.0-1.0)
            
        Returns:
            Simulated execution time in seconds
        """
        base_time = 0.05  # 50ms base execution time
        complexity_factor = test_complexity * 0.1  # Up to 100ms for complex tests
        
        # Use deterministic jitter
        jitter = self.seed_manager.generate_deterministic_value(
            "performance_simulator", "test_execution_jitter", -0.01, 0.01
        )
        
        return max(0.01, base_time + complexity_factor + jitter)
    
    def simulate_response_time(self, api_complexity: float, load_factor: float = 1.0) -> float:
        """Simulate API response time.
        
        Args:
            api_complexity: Complexity of the API call (0.0-1.0)
            load_factor: System load factor (1.0 = normal load)
            
        Returns:
            Simulated response time in seconds
        """
        base_time = self.base_response_time
        complexity_penalty = api_complexity * 0.05  # Up to 50ms for complex APIs
        load_penalty = (load_factor - 1.0) * 0.02  # 20ms per load unit above normal
        
        # Use deterministic jitter
        jitter = self.seed_manager.generate_deterministic_value(
            "performance_simulator", "response_time_jitter", -0.01, 0.01
        )
        
        return max(0.005, base_time + complexity_penalty + load_penalty + jitter)


class VulnerabilityInjector(BaseVulnerabilityInjector):
    """Injector for realistic vulnerability simulation."""
    
    def __init__(self, seed_manager: Optional[RandomSeedManager] = None):
        """Initialize vulnerability injector.
        
        Args:
            seed_manager: Random seed manager for deterministic behavior
        """
        self.vulnerabilities: Dict[str, InjectedVulnerability] = {}
        self._vulnerability_counter = 0
        self.seed_manager = seed_manager or get_seed_manager()
    
    def inject_vulnerability(self, vuln_type: str, trigger_condition: str, 
                           parameter_name: Optional[str] = None) -> str:
        """Inject a vulnerability into the target system.
        
        Args:
            vuln_type: Type of vulnerability to inject
            trigger_condition: Condition that triggers the vulnerability
            parameter_name: Parameter that triggers the vulnerability
            
        Returns:
            ID of the injected vulnerability
        """
        self._vulnerability_counter += 1
        vuln_id = f"vuln_{self._vulnerability_counter}"
        
        # Convert string to enum if needed
        if isinstance(vuln_type, str):
            try:
                vuln_type_enum = VulnerabilityType(vuln_type)
            except ValueError:
                vuln_type_enum = VulnerabilityType.SQL_INJECTION  # Default
        else:
            vuln_type_enum = vuln_type
        
        vulnerability = InjectedVulnerability(
            id=vuln_id,
            vuln_type=vuln_type_enum,
            trigger_condition=trigger_condition,
            parameter_name=parameter_name,
            trigger_value=self._generate_trigger_value(vuln_type_enum)
        )
        
        self.vulnerabilities[vuln_id] = vulnerability
        return vuln_id
    
    def remove_vulnerability(self, vuln_id: str) -> None:
        """Remove a previously injected vulnerability.
        
        Args:
            vuln_id: ID of vulnerability to remove
        """
        if vuln_id in self.vulnerabilities:
            del self.vulnerabilities[vuln_id]
    
    def list_vulnerabilities(self) -> List[Dict[str, Any]]:
        """List all currently injected vulnerabilities.
        
        Returns:
            List of vulnerability information dictionaries
        """
        return [
            {
                "id": vuln.id,
                "type": vuln.vuln_type.value,
                "trigger_condition": vuln.trigger_condition,
                "parameter_name": vuln.parameter_name,
                "trigger_value": vuln.trigger_value,
                "active": vuln.active,
                "created_at": vuln.created_at.isoformat()
            }
            for vuln in self.vulnerabilities.values()
        ]
    
    def check_vulnerability_trigger(self, test_case: TestCase) -> List[InjectedVulnerability]:
        """Check if test case triggers any vulnerabilities.
        
        Args:
            test_case: Test case to check
            
        Returns:
            List of triggered vulnerabilities
        """
        triggered = []
        
        for vuln in self.vulnerabilities.values():
            if not vuln.active:
                continue
                
            if self._is_vulnerability_triggered(vuln, test_case):
                triggered.append(vuln)
        
        return triggered
    
    def _generate_trigger_value(self, vuln_type: VulnerabilityType) -> str:
        """Generate a trigger value for the vulnerability type.
        
        Args:
            vuln_type: Type of vulnerability
            
        Returns:
            Trigger value that would exploit the vulnerability
        """
        triggers = {
            VulnerabilityType.SQL_INJECTION: "'; DROP TABLE users; --",
            VulnerabilityType.BUFFER_OVERFLOW: "A" * 1000,
            VulnerabilityType.AUTH_BYPASS: "admin' OR '1'='1",
            VulnerabilityType.XSS: "<script>alert('xss')</script>",
            VulnerabilityType.PATH_TRAVERSAL: "../../../etc/passwd",
            VulnerabilityType.COMMAND_INJECTION: "; rm -rf /"
        }
        return triggers.get(vuln_type, "malicious_input")
    
    def _is_vulnerability_triggered(self, vuln: InjectedVulnerability, test_case: TestCase) -> bool:
        """Check if a vulnerability is triggered by the test case.
        
        Args:
            vuln: Vulnerability to check
            test_case: Test case to evaluate
            
        Returns:
            True if vulnerability is triggered
        """
        # Check if specific parameter is targeted
        if vuln.parameter_name and vuln.parameter_name in test_case.parameters:
            param_value = str(test_case.parameters[vuln.parameter_name])
            
            # Check for trigger value or pattern
            if vuln.trigger_value and vuln.trigger_value in param_value:
                return True
            
            # Check for vulnerability-specific patterns
            if vuln.vuln_type == VulnerabilityType.SQL_INJECTION:
                sql_patterns = ["'", "DROP", "SELECT", "UNION", "INSERT", "--", ";"]
                return any(pattern.lower() in param_value.lower() for pattern in sql_patterns)
            
            elif vuln.vuln_type == VulnerabilityType.XSS:
                xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
                return any(pattern.lower() in param_value.lower() for pattern in xss_patterns)
            
            elif vuln.vuln_type == VulnerabilityType.BUFFER_OVERFLOW:
                return len(param_value) > 500  # Trigger on long inputs
            
            elif vuln.vuln_type == VulnerabilityType.PATH_TRAVERSAL:
                return ".." in param_value or "/etc/" in param_value
            
            elif vuln.vuln_type == VulnerabilityType.COMMAND_INJECTION:
                cmd_patterns = [";", "|", "&", "`", "$", "rm ", "cat "]
                return any(pattern in param_value for pattern in cmd_patterns)
        
        # Check general trigger condition
        return vuln.trigger_condition in str(test_case.parameters)


class CoverageTracker:
    """Tracker for code coverage metrics."""
    
    def __init__(self, seed_manager: Optional[RandomSeedManager] = None):
        """Initialize coverage tracker.
        
        Args:
            seed_manager: Random seed manager for deterministic behavior
        """
        self.covered_lines: Set[int] = set()
        self.covered_branches: Set[str] = set()
        self.covered_functions: Set[str] = set()
        self.covered_endpoints: Set[str] = set()
        self.seed_manager = seed_manager or get_seed_manager()
        
        # Simulated total counts
        self.total_lines = 1000
        self.total_branches = 200
        self.total_functions = 50
        self.total_endpoints = 20
    
    def update_coverage(self, test_case: TestCase) -> CoverageMetrics:
        """Update coverage based on test case execution.
        
        Args:
            test_case: Executed test case
            
        Returns:
            Updated coverage metrics
        """
        # Simulate coverage based on test case characteristics
        coverage_factor = self._calculate_coverage_factor(test_case)
        
        # Add new lines based on coverage factor
        new_lines = int(coverage_factor * 50)  # Up to 50 new lines per test
        for _ in range(new_lines):
            line_num = self.seed_manager.generate_deterministic_int(
                "coverage_tracker", "line_selection", 1, self.total_lines + 1
            )
            self.covered_lines.add(line_num)
        
        # Add new branches
        new_branches = int(coverage_factor * 10)  # Up to 10 new branches per test
        for _ in range(new_branches):
            branch_num = self.seed_manager.generate_deterministic_int(
                "coverage_tracker", "branch_selection", 1, self.total_branches + 1
            )
            branch_id = f"branch_{branch_num}"
            self.covered_branches.add(branch_id)
        
        # Add functions
        func_threshold = self.seed_manager.generate_deterministic_value(
            "coverage_tracker", "function_threshold", 0.0, 1.0
        )
        if func_threshold < coverage_factor:
            func_num = self.seed_manager.generate_deterministic_int(
                "coverage_tracker", "function_selection", 1, self.total_functions + 1
            )
            func_name = f"function_{func_num}"
            self.covered_functions.add(func_name)
        
        # Add endpoint
        self.covered_endpoints.add(test_case.api_name)
        
        return self.get_current_metrics()
    
    def get_current_metrics(self) -> CoverageMetrics:
        """Get current coverage metrics.
        
        Returns:
            Current coverage metrics
        """
        line_coverage = len(self.covered_lines) / self.total_lines
        branch_coverage = len(self.covered_branches) / self.total_branches
        function_coverage = len(self.covered_functions) / self.total_functions
        endpoint_coverage = len(self.covered_endpoints) / self.total_endpoints
        
        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            api_endpoint_coverage=endpoint_coverage,
            covered_lines=self.covered_lines.copy(),
            total_lines=self.total_lines,
            covered_branches=self.covered_branches.copy(),
            total_branches=self.total_branches
        )
    
    def _calculate_coverage_factor(self, test_case: TestCase) -> float:
        """Calculate coverage factor based on test case characteristics.
        
        Args:
            test_case: Test case to analyze
            
        Returns:
            Coverage factor between 0.0 and 1.0
        """
        base_factor = 0.3  # Base coverage for any test
        
        # Adjust based on test type
        type_factors = {
            TestType.NORMAL: 0.4,
            TestType.EDGE: 0.6,
            TestType.SECURITY: 0.8,
            TestType.MALFORMED: 0.2
        }
        
        type_factor = type_factors.get(test_case.test_type, 0.3)
        
        # Adjust based on parameter complexity
        param_factor = min(0.3, len(test_case.parameters) * 0.05)
        
        return min(1.0, base_factor + type_factor + param_factor)


class TargetSystemSimulator(BaseTargetSystem):
    """Simulator for realistic target system behavior."""
    
    def __init__(self, config: FuzzerConfig, seed_manager: Optional[RandomSeedManager] = None):
        """Initialize target system simulator.
        
        Args:
            config: Fuzzer configuration
            seed_manager: Random seed manager for deterministic behavior
        """
        self.config = config
        self.seed_manager = seed_manager or get_seed_manager()
        self.performance_sim = PerformanceSimulator(config, self.seed_manager)
        
        # Use advanced components
        self.vuln_injector = AdvancedVulnerabilityInjector()
        session_id = f"session_{int(time.time())}"
        self.coverage_tracker = AdvancedCoverageTracker(session_id)
        
        # Legacy compatibility with deterministic behavior
        self.legacy_vuln_injector = VulnerabilityInjector(self.seed_manager)
        self.legacy_coverage_tracker = CoverageTracker(self.seed_manager)
        
        # System state
        self.system_state: Dict[str, Any] = {}
        self.request_count = 0
        self.session_start_time = time.time()
        
        # Inject default vulnerabilities if enabled
        if config.vulnerability_injection:
            self._inject_default_vulnerabilities()
    
    def execute_test(self, test_case: TestCase) -> ExecutionResult:
        """Execute a test case against the target system.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            ExecutionResult with test results
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Simulate API call
            response = self._simulate_api_call(test_case)
            
            # Update coverage using advanced tracker
            coverage_delta = None
            if self.config.coverage_tracking:
                coverage_delta = self.coverage_tracker.update_coverage(test_case)
            
            # Check for vulnerabilities
            detected_defects = self._check_for_defects(test_case, response)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Determine success
            success = response.status_code < 400 and not response.error_message
            
            return ExecutionResult(
                test_case_id=test_case.id,
                success=success,
                response=response.data,
                execution_time=execution_time,
                coverage_delta=coverage_delta,
                errors=[response.error_message] if response.error_message else [],
                detected_defects=detected_defects,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                test_case_id=test_case.id,
                success=False,
                response=None,
                execution_time=execution_time,
                coverage_delta=None,
                errors=[str(e)],
                detected_defects=[],
                timestamp=datetime.now()
            )
    
    def get_coverage_info(self) -> CoverageMetrics:
        """Get current coverage information.
        
        Returns:
            CoverageMetrics with current coverage data
        """
        # Get metrics from advanced tracker
        report = self.coverage_tracker.get_coverage_report()
        from src.fuzzer.coverage_tracker import CoverageType
        
        return CoverageMetrics(
            line_coverage=report.coverage_by_type.get(CoverageType.LINE, 0.0),
            branch_coverage=report.coverage_by_type.get(CoverageType.BRANCH, 0.0),
            function_coverage=report.coverage_by_type.get(CoverageType.FUNCTION, 0.0),
            api_endpoint_coverage=report.coverage_by_type.get(CoverageType.API_ENDPOINT, 0.0),
            covered_lines=set(),  # Simplified for compatibility
            total_lines=1000,
            covered_branches=set(),
            total_branches=200
        )
    
    def reset_state(self) -> None:
        """Reset target system state for new test session."""
        self.system_state.clear()
        self.request_count = 0
        self.session_start_time = time.time()
        
        # Reset advanced coverage tracker
        self.coverage_tracker.reset_coverage()
        
        # Reset legacy tracker for compatibility
        self.legacy_coverage_tracker = CoverageTracker()
    
    def get_call_count(self) -> int:
        """Get the number of API calls made to this target system.
        
        Returns:
            Number of API calls executed
        """
        return self.request_count
    
    def inject_vulnerability(self, vuln_type: str, trigger_condition: str, 
                           parameter_name: Optional[str] = None) -> str:
        """Inject a vulnerability into the target system.
        
        Args:
            vuln_type: Type of vulnerability to inject
            trigger_condition: Condition that triggers the vulnerability
            parameter_name: Parameter that triggers the vulnerability
            
        Returns:
            ID of the injected vulnerability
        """
        # Use legacy injector for backward compatibility
        return self.legacy_vuln_injector.inject_vulnerability(
            vuln_type, trigger_condition, parameter_name
        )
    
    def _simulate_api_call(self, test_case: TestCase) -> APIResponse:
        """Simulate an API call for the test case.
        
        Args:
            test_case: Test case to simulate
            
        Returns:
            APIResponse with simulated response
        """
        # Calculate response time
        api_complexity = self._calculate_api_complexity(test_case)
        load_factor = self._calculate_load_factor()
        response_time = self.performance_sim.simulate_response_time(api_complexity, load_factor)
        
        # Simulate actual delay
        time.sleep(min(response_time, 0.1))  # Cap at 100ms for simulation
        
        # Determine if error should occur
        should_error = self._should_simulate_error(test_case)
        
        if should_error:
            return self._generate_error_response(test_case, response_time)
        else:
            return self._generate_success_response(test_case, response_time)
    
    def _calculate_api_complexity(self, test_case: TestCase) -> float:
        """Calculate API complexity factor.
        
        Args:
            test_case: Test case to analyze
            
        Returns:
            Complexity factor between 0.0 and 1.0
        """
        base_complexity = 0.3
        param_complexity = min(0.4, len(test_case.parameters) * 0.1)
        
        # Security tests are more complex
        type_complexity = 0.3 if test_case.test_type == TestType.SECURITY else 0.1
        
        return min(1.0, base_complexity + param_complexity + type_complexity)
    
    def _calculate_load_factor(self) -> float:
        """Calculate current system load factor.
        
        Returns:
            Load factor (1.0 = normal load)
        """
        # Simulate increasing load over time
        session_duration = time.time() - self.session_start_time
        load_increase = min(0.5, session_duration / 300)  # Increase over 5 minutes
        
        # Add request-based load
        request_load = min(0.3, self.request_count / 100)
        
        return 1.0 + load_increase + request_load
    
    def _should_simulate_error(self, test_case: TestCase) -> bool:
        """Determine if test should result in an error.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            True if error should be simulated
        """
        # Base error rates by test type
        error_rates = {
            TestType.NORMAL: self.config.base_error_rate,
            TestType.EDGE: self.config.base_error_rate * 2,
            TestType.SECURITY: self.config.base_error_rate * 3,
            TestType.MALFORMED: 0.8  # High error rate for malformed inputs
        }
        
        error_rate = error_rates.get(test_case.test_type, self.config.base_error_rate)
        
        # Use deterministic random value
        random_value = self.seed_manager.generate_deterministic_value(
            "target_system", f"error_decision_{test_case.id}", 0.0, 1.0
        )
        
        return random_value < error_rate
    
    def _generate_success_response(self, test_case: TestCase, response_time: float) -> APIResponse:
        """Generate a successful API response.
        
        Args:
            test_case: Test case being executed
            response_time: Simulated response time
            
        Returns:
            APIResponse with success data
        """
        # Generate realistic response data
        response_data = self._generate_response_data(test_case)
        
        return APIResponse(
            status_code=200,
            data=response_data,
            headers={"Content-Type": "application/json"},
            response_time=response_time
        )
    
    def _generate_error_response(self, test_case: TestCase, response_time: float) -> APIResponse:
        """Generate an error API response.
        
        Args:
            test_case: Test case being executed
            response_time: Simulated response time
            
        Returns:
            APIResponse with error information
        """
        # Choose error type based on test case
        if test_case.test_type == TestType.MALFORMED:
            status_code = 400
            error_message = "Bad Request: Invalid input format"
        elif test_case.test_type == TestType.SECURITY:
            status_code = 403
            error_message = "Forbidden: Security violation detected"
        else:
            status_code = 500
            error_message = "Internal Server Error"
        
        return APIResponse(
            status_code=status_code,
            data={"error": error_message},
            headers={"Content-Type": "application/json"},
            response_time=response_time,
            error_message=error_message
        )
    
    def _generate_response_data(self, test_case: TestCase) -> Dict[str, Any]:
        """Generate realistic response data for successful calls.
        
        Args:
            test_case: Test case being executed
            
        Returns:
            Dictionary with response data
        """
        # Generate data based on API name and parameters
        api_name = test_case.api_name.lower()
        
        if "user" in api_name:
            user_id = self.seed_manager.generate_deterministic_int(
                "target_system", f"user_id_{test_case.id}", 1, 10001
            )
            username_num = self.seed_manager.generate_deterministic_int(
                "target_system", f"username_{test_case.id}", 1, 1001
            )
            email_num = self.seed_manager.generate_deterministic_int(
                "target_system", f"email_{test_case.id}", 1, 1001
            )
            return {
                "id": user_id,
                "username": f"user_{username_num}",
                "email": f"user{email_num}@example.com",
                "created_at": datetime.now().isoformat()
            }
        elif "search" in api_name:
            result_count = self.seed_manager.generate_deterministic_int(
                "target_system", f"search_results_{test_case.id}", 0, 11
            )
            total_count = self.seed_manager.generate_deterministic_int(
                "target_system", f"search_total_{test_case.id}", 0, 101
            )
            return {
                "results": [
                    {
                        "id": i, 
                        "title": f"Result {i}", 
                        "score": self.seed_manager.generate_deterministic_value(
                            "target_system", f"search_score_{test_case.id}_{i}", 0.0, 1.0
                        )
                    }
                    for i in range(result_count)
                ],
                "total": total_count
            }
        else:
            # Generic response
            return {
                "status": "success",
                "data": test_case.parameters,
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_for_defects(self, test_case: TestCase, response: APIResponse) -> List[Defect]:
        """Check for defects based on test execution.
        
        Args:
            test_case: Executed test case
            response: API response
            
        Returns:
            List of detected defects
        """
        defects = []
        
        # Check for vulnerability triggers using advanced injector
        advanced_defects = self.vuln_injector.check_vulnerabilities(test_case)
        defects.extend(advanced_defects)
        
        # Also check legacy vulnerabilities for backward compatibility
        legacy_triggered_vulns = self.legacy_vuln_injector.check_vulnerability_trigger(test_case)
        
        for vuln in legacy_triggered_vulns:
            defect = Defect(
                id=f"defect_{vuln.id}_{test_case.id}",
                type=DefectType.SECURITY,
                severity=self._get_vulnerability_severity(vuln.vuln_type),
                description=f"{vuln.vuln_type.value} vulnerability triggered",
                trigger_test_case=test_case.id,
                location=vuln.parameter_name,
                impact_assessment=f"Vulnerability in {vuln.parameter_name} parameter"
            )
            defects.append(defect)
        
        # Check for functional defects based on response
        if response.status_code >= 500:
            defect = Defect(
                id=f"defect_functional_{test_case.id}",
                type=DefectType.FUNCTIONAL,
                severity=DefectSeverity.HIGH,
                description="Server error during test execution",
                trigger_test_case=test_case.id,
                impact_assessment="System instability detected"
            )
            defects.append(defect)
        
        # Check for performance defects
        if response.response_time > 5.0:  # Slow response
            defect = Defect(
                id=f"defect_performance_{test_case.id}",
                type=DefectType.PERFORMANCE,
                severity=DefectSeverity.MEDIUM,
                description="Slow response time detected",
                trigger_test_case=test_case.id,
                impact_assessment=f"Response time: {response.response_time:.2f}s"
            )
            defects.append(defect)
        
        return defects
    
    def _get_vulnerability_severity(self, vuln_type: VulnerabilityType) -> DefectSeverity:
        """Get severity level for vulnerability type.
        
        Args:
            vuln_type: Type of vulnerability
            
        Returns:
            Severity level for the vulnerability
        """
        severity_map = {
            VulnerabilityType.SQL_INJECTION: DefectSeverity.CRITICAL,
            VulnerabilityType.BUFFER_OVERFLOW: DefectSeverity.CRITICAL,
            VulnerabilityType.AUTH_BYPASS: DefectSeverity.CRITICAL,
            VulnerabilityType.COMMAND_INJECTION: DefectSeverity.CRITICAL,
            VulnerabilityType.XSS: DefectSeverity.HIGH,
            VulnerabilityType.PATH_TRAVERSAL: DefectSeverity.HIGH
        }
        return severity_map.get(vuln_type, DefectSeverity.MEDIUM)
    
    def _inject_default_vulnerabilities(self) -> None:
        """Inject default vulnerabilities for testing."""
        # Legacy vulnerabilities for backward compatibility
        # SQL injection in user_id parameter
        self.legacy_vuln_injector.inject_vulnerability(
            VulnerabilityType.SQL_INJECTION,
            "user_id parameter contains SQL",
            "user_id"
        )
        
        # Buffer overflow in message parameter
        self.legacy_vuln_injector.inject_vulnerability(
            VulnerabilityType.BUFFER_OVERFLOW,
            "message parameter too long",
            "message"
        )
        
        # XSS in content parameter
        self.legacy_vuln_injector.inject_vulnerability(
            VulnerabilityType.XSS,
            "content parameter contains script",
            "content"
        )
        
        # Auth bypass in admin parameter
        self.legacy_vuln_injector.inject_vulnerability(
            VulnerabilityType.AUTH_BYPASS,
            "admin parameter bypass",
            "admin"
        )
        
        # Advanced vulnerability injector already has default rules initialized