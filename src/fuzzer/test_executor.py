"""Test executor for running test cases against target system."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from src.fuzzer.data_models import TestCase, ExecutionResult, FuzzerConfig
from src.fuzzer.base_interfaces import BaseTargetSystem

from src.utils.logger import get_logger


class TestExecutor:
    """Executes test cases against target system with parallel processing."""
    
    def __init__(self, config: FuzzerConfig):
        """Initialize test executor.
        
        Args:
            config: Fuzzer configuration
        """
        self.config = config
        self.logger = get_logger("TestExecutor")
        self.last_execution_time = 0.0
    
    def run_test_suite(self, test_cases: List[TestCase], target_system: BaseTargetSystem) -> List[ExecutionResult]:

        """Run a complete test suite.
        
        Args:
            test_cases: List of test cases to execute
            target_system: Target system simulator
            
        Returns:
            List of execution results
        """
        if not test_cases:
            self.logger.warning("No test cases to execute")
            return []
        
        self.logger.info(f"Executing {len(test_cases)} test cases")
        start_time = time.time()
        
        if self.config.parallel_execution and len(test_cases) > 1:
            results = self._run_parallel(test_cases, target_system)
        else:
            results = self._run_sequential(test_cases, target_system)
        
        self.last_execution_time = time.time() - start_time
        
        # Log summary
        successful_tests = sum(1 for result in results if result.success)
        failed_tests = len(results) - successful_tests
        total_defects = sum(len(result.detected_defects) for result in results)
        
        self.logger.info(f"Test execution completed in {self.last_execution_time:.2f}s")
        self.logger.info(f"Results: {successful_tests} passed, {failed_tests} failed, {total_defects} defects")
        
        return results
    
    def execute_single_test(self, test_case: TestCase, target_system: BaseTargetSystem) -> ExecutionResult:

        """Execute a single test case with timeout and error handling.
        
        Args:
            test_case: Test case to execute
            target_system: Target system simulator
            
        Returns:
            ExecutionResult with execution details
        """
        self.logger.debug(f"Executing test case: {test_case.id}")
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(test_case, target_system)
            
            # Retry failed tests if configured
            if not result.success and self.config.retry_failed_tests:
                self.logger.debug(f"Retrying failed test: {test_case.id}")
                retry_result = self._execute_with_timeout(test_case, target_system)
                
                # Use retry result if it succeeded
                if retry_result.success:
                    result = retry_result
                else:
                    # Combine error information
                    result.errors.extend([f"Retry also failed: {err}" for err in retry_result.errors])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Test execution failed for {test_case.id}: {e}")
            
            return ExecutionResult(
                test_case_id=test_case.id,
                success=False,
                execution_time=0.0,
                errors=[f"Execution exception: {str(e)}"]
            )
    
    def collect_metrics(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Collect execution metrics from results.
        
        Args:
            results: List of execution results
            
        Returns:
            Dictionary with execution metrics
        """
        if not results:
            return {}
        
        total_tests = len(results)
        successful_tests = sum(1 for result in results if result.success)
        failed_tests = total_tests - successful_tests
        
        total_execution_time = sum(result.execution_time for result in results)
        average_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        total_defects = sum(len(result.detected_defects) for result in results)
        
        # Collect error types
        error_types = {}
        for result in results:
            for error in result.errors:
                error_type = error.split(':')[0] if ':' in error else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': average_execution_time,
            'total_defects': total_defects,
            'defects_per_test': total_defects / total_tests if total_tests > 0 else 0,
            'error_types': error_types
        }
    
    def detect_anomalies(self, results: List[ExecutionResult]) -> List[Dict[str, Any]]:
        """Detect anomalies in test execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not results:
            return anomalies
        
        # Calculate baseline metrics
        execution_times = [result.execution_time for result in results if result.execution_time > 0]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            
            # Detect unusually slow tests
            for result in results:
                if result.execution_time > avg_time * 3:  # 3x slower than average
                    anomalies.append({
                        'type': 'slow_execution',
                        'test_case_id': result.test_case_id,
                        'execution_time': result.execution_time,
                        'average_time': avg_time,
                        'description': f"Test took {result.execution_time:.2f}s (avg: {avg_time:.2f}s)"
                    })
        
        # Detect tests with many defects
        defect_counts = [len(result.detected_defects) for result in results]
        if defect_counts:
            avg_defects = sum(defect_counts) / len(defect_counts)
            
            for result in results:
                defect_count = len(result.detected_defects)
                if defect_count > avg_defects * 2 and defect_count > 3:
                    anomalies.append({
                        'type': 'high_defect_count',
                        'test_case_id': result.test_case_id,
                        'defect_count': defect_count,
                        'average_defects': avg_defects,
                        'description': f"Test found {defect_count} defects (avg: {avg_defects:.1f})"
                    })
        
        # Detect repeated failures
        error_patterns = {}
        for result in results:
            if not result.success and result.errors:
                error_key = result.errors[0]  # Use first error as key
                if error_key not in error_patterns:
                    error_patterns[error_key] = []
                error_patterns[error_key].append(result.test_case_id)
        
        for error, test_ids in error_patterns.items():
            if len(test_ids) > len(results) * 0.2:  # More than 20% of tests
                anomalies.append({
                    'type': 'repeated_failure',
                    'error_message': error,
                    'affected_tests': test_ids,
                    'count': len(test_ids),
                    'description': f"Error '{error}' occurred in {len(test_ids)} tests"
                })
        
        self.logger.info(f"Detected {len(anomalies)} anomalies in test execution")
        return anomalies
    
    def _run_parallel(self, test_cases: List[TestCase], target_system: BaseTargetSystem) -> List[ExecutionResult]:

        """Run test cases in parallel."""
        self.logger.debug(f"Running {len(test_cases)} tests in parallel with {self.config.max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all test cases
            future_to_test = {
                executor.submit(self.execute_single_test, test_case, target_system): test_case
                for test_case in test_cases
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Parallel execution failed for {test_case.id}: {e}")
                    
                    # Create error result
                    error_result = ExecutionResult(
                        test_case_id=test_case.id,
                        success=False,
                        execution_time=0.0,
                        errors=[f"Parallel execution error: {str(e)}"]
                    )
                    results.append(error_result)
        
        # Sort results by test case ID to maintain order
        results.sort(key=lambda r: r.test_case_id)
        return results
    
    def _run_sequential(self, test_cases: List[TestCase], target_system: BaseTargetSystem) -> List[ExecutionResult]:

        """Run test cases sequentially."""
        self.logger.debug(f"Running {len(test_cases)} tests sequentially")
        
        results = []
        
        for test_case in test_cases:
            result = self.execute_single_test(test_case, target_system)
            results.append(result)
        
        return results
    
    def _execute_with_timeout(self, test_case: TestCase, target_system: BaseTargetSystem) -> ExecutionResult:

        """Execute test case with timeout handling."""
        import signal
        import threading
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test execution timed out after {self.config.timeout_per_test}s")
        
        # Set up timeout (Unix-like systems only)
        if threading.current_thread() is not threading.main_thread():
            return target_system.execute_test(test_case)

        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout_per_test))

            try:
                result = target_system.execute_test(test_case)
                return result

            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)

        except (AttributeError, OSError):
            # signal.SIGALRM not available (Windows) or other OS error
            # Fall back to simple execution without timeout
            return target_system.execute_test(test_case)
