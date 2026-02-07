"""Feedback collector for aggregating and analyzing test results."""

import hashlib
from typing import List, Dict, Any
from datetime import datetime
from src.fuzzer.data_models import (
    FeedbackReport, APISpec, TestCase, ExecutionResult,
    CoverageMetrics, Defect, DefectType, DefectSeverity,
    PerformanceMetrics, FuzzerConfig
)
from src.utils.logger import get_logger


class FeedbackCollector:
    """Collects and analyzes feedback from test execution."""
    
    def __init__(self, config: FuzzerConfig):
        """Initialize feedback collector.
        
        Args:
            config: Fuzzer configuration
        """
        self.config = config
        self.logger = get_logger("FeedbackCollector")
    
    def generate_report(
        self,
        session_id: str,
        document_content: str,
        api_specs: List[APISpec],
        test_cases: List[TestCase],
        execution_results: List[ExecutionResult]
    ) -> FeedbackReport:
        """Generate comprehensive feedback report.
        
        Args:
            session_id: Unique session identifier
            document_content: Original document content
            api_specs: API specifications extracted from document
            test_cases: Generated test cases
            execution_results: Results from test execution
            
        Returns:
            FeedbackReport with comprehensive analysis
        """
        self.logger.info(f"Generating feedback report for session {session_id}")
        
        # Calculate basic metrics
        validity_rate = self.calculate_validity_rate(execution_results)
        coverage_metrics = self.measure_coverage(execution_results)
        defects_found = self.classify_defects(execution_results)
        
        # Generate performance metrics
        performance_metrics = self._calculate_performance_metrics(execution_results)
        
        # Create report
        report = FeedbackReport(
            session_id=session_id,
            document_hash=self._hash_content(document_content),
            total_test_cases=len(test_cases),
            validity_rate=validity_rate,
            coverage_metrics=coverage_metrics,
            defects_found=defects_found,
            performance_metrics=performance_metrics,
            api_specs=api_specs,
            test_cases=test_cases if self.config.save_test_cases else [],
            execution_results=execution_results
        )
        
        self.logger.info(f"Report generated: {validity_rate:.2%} validity, "
                        f"{coverage_metrics.line_coverage:.1f}% coverage, "
                        f"{len(defects_found)} defects")
        
        return report
    
    def calculate_validity_rate(self, results: List[ExecutionResult]) -> float:
        """Calculate validity rate from execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            Validity rate (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        successful_tests = sum(1 for result in results if result.success)
        return successful_tests / len(results)
    
    def measure_coverage(self, results: List[ExecutionResult]) -> CoverageMetrics:
        """Measure code coverage from execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            CoverageMetrics with aggregated coverage information
        """
        if not results:
            return CoverageMetrics()
        
        # Aggregate coverage from all results
        all_covered_lines = set()
        all_covered_branches = set()
        total_lines = 0
        total_branches = 0
        
        for result in results:
            if result.coverage_delta:
                all_covered_lines.update(result.coverage_delta.covered_lines)
                all_covered_branches.update(result.coverage_delta.covered_branches)
                total_lines = max(total_lines, result.coverage_delta.total_lines)
                total_branches = max(total_branches, result.coverage_delta.total_branches)
        
        # Use reasonable defaults if no coverage data
        if total_lines == 0:
            total_lines = 1000  # Default total lines
        if total_branches == 0:
            total_branches = 200  # Default total branches
        
        # Calculate coverage percentages
        line_coverage = len(all_covered_lines) / total_lines * 100 if total_lines > 0 else 0
        branch_coverage = len(all_covered_branches) / total_branches * 100 if total_branches > 0 else 0
        
        # Estimate function and API coverage
        function_coverage = min(line_coverage * 0.8, 100)  # Approximate
        api_coverage = min(len(results) / 10 * 100, 100)  # Based on test count
        
        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            api_endpoint_coverage=api_coverage,
            covered_lines=all_covered_lines,
            total_lines=total_lines,
            covered_branches=all_covered_branches,
            total_branches=total_branches
        )
    
    def classify_defects(self, results: List[ExecutionResult]) -> List[Defect]:
        """Classify and aggregate defects from execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            List of classified defects
        """
        all_defects = []
        
        for result in results:
            all_defects.extend(result.detected_defects)
        
        # Deduplicate defects by description and location
        unique_defects = {}
        for defect in all_defects:
            key = f"{defect.description}_{defect.location}"
            if key not in unique_defects:
                unique_defects[key] = defect
            else:
                # Merge information from duplicate defects
                existing = unique_defects[key]
                if defect.severity.value == 'critical' and existing.severity.value != 'critical':
                    existing.severity = defect.severity
        
        classified_defects = list(unique_defects.values())
        
        # Sort by severity (critical first)
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        classified_defects.sort(key=lambda d: severity_order.get(d.severity.value, 4))
        
        self.logger.debug(f"Classified {len(classified_defects)} unique defects")
        return classified_defects
    
    def generate_statistical_summary(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Generate statistical summary of execution results.
        
        Args:
            results: List of execution results
            
        Returns:
            Dictionary with statistical information
        """
        if not results:
            return {}
        
        # Basic statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - successful_tests
        
        # Execution time statistics
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        
        # Defect statistics
        defect_counts = [len(r.detected_defects) for r in results]
        total_defects = sum(defect_counts)
        avg_defects_per_test = total_defects / total_tests if total_tests > 0 else 0
        
        # Error analysis
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
            'execution_time': {
                'average': avg_execution_time,
                'minimum': min_execution_time,
                'maximum': max_execution_time,
                'total': sum(execution_times)
            },
            'defects': {
                'total': total_defects,
                'average_per_test': avg_defects_per_test,
                'max_per_test': max(defect_counts) if defect_counts else 0
            },
            'error_types': error_types
        }
    
    def export_report(self, report: FeedbackReport, format: str = "json") -> str:
        """Export feedback report to specified format.
        
        Args:
            report: Feedback report to export
            format: Export format ("json" or "csv")
            
        Returns:
            Exported report as string
        """
        if format.lower() == "json":
            return self._export_json(report)
        elif format.lower() == "csv":
            return self._export_csv(report)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def compare_reports(self, baseline_report: FeedbackReport, current_report: FeedbackReport) -> Dict[str, Any]:
        """Compare two feedback reports to identify changes.
        
        Args:
            baseline_report: Baseline report for comparison
            current_report: Current report to compare against baseline
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'validity_change': current_report.validity_rate - baseline_report.validity_rate,
            'coverage_change': (
                current_report.coverage_metrics.line_coverage - 
                baseline_report.coverage_metrics.line_coverage
            ),
            'defect_change': len(current_report.defects_found) - len(baseline_report.defects_found),
            'performance_change': (
                current_report.performance_metrics.total_execution_time - 
                baseline_report.performance_metrics.total_execution_time
            )
        }
        
        # Calculate relative changes
        if baseline_report.validity_rate > 0:
            comparison['validity_change_percent'] = (
                comparison['validity_change'] / baseline_report.validity_rate * 100
            )
        
        if baseline_report.coverage_metrics.line_coverage > 0:
            comparison['coverage_change_percent'] = (
                comparison['coverage_change'] / baseline_report.coverage_metrics.line_coverage * 100
            )
        
        # Identify new defects
        baseline_defect_descriptions = {d.description for d in baseline_report.defects_found}
        current_defect_descriptions = {d.description for d in current_report.defects_found}
        
        comparison['new_defects'] = list(current_defect_descriptions - baseline_defect_descriptions)
        comparison['resolved_defects'] = list(baseline_defect_descriptions - current_defect_descriptions)
        
        return comparison
    
    def _calculate_performance_metrics(self, results: List[ExecutionResult]) -> PerformanceMetrics:
        """Calculate performance metrics from execution results."""
        if not results:
            return PerformanceMetrics()
        
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        total_time = sum(execution_times)
        avg_time = total_time / len(execution_times) if execution_times else 0
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            average_test_time=avg_time,
            test_execution_time=total_time,  # Same as total for now
            tests_per_second=len(results) / total_time if total_time > 0 else 0
        )
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _export_json(self, report: FeedbackReport) -> str:
        """Export report as JSON."""
        import json
        
        # Convert report to dictionary
        report_dict = {
            'session_id': report.session_id,
            'document_hash': report.document_hash,
            'timestamp': report.timestamp.isoformat(),
            'total_test_cases': report.total_test_cases,
            'validity_rate': report.validity_rate,
            'coverage': {
                'line_coverage': report.coverage_metrics.line_coverage,
                'branch_coverage': report.coverage_metrics.branch_coverage,
                'function_coverage': report.coverage_metrics.function_coverage,
                'api_endpoint_coverage': report.coverage_metrics.api_endpoint_coverage
            },
            'defects': [
                {
                    'id': d.id,
                    'type': d.type.value,
                    'severity': d.severity.value,
                    'description': d.description,
                    'location': d.location
                }
                for d in report.defects_found
            ],
            'performance': {
                'total_execution_time': report.performance_metrics.total_execution_time,
                'average_test_time': report.performance_metrics.average_test_time,
                'tests_per_second': report.performance_metrics.tests_per_second
            }
        }
        
        return json.dumps(report_dict, indent=2)
    
    def _export_csv(self, report: FeedbackReport) -> str:
        """Export report as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Session ID', 'Timestamp', 'Total Tests', 'Validity Rate',
            'Line Coverage', 'Branch Coverage', 'Total Defects',
            'Security Defects', 'Execution Time'
        ])
        
        # Write data
        security_defects = sum(1 for d in report.defects_found if d.type == DefectType.SECURITY)
        
        writer.writerow([
            report.session_id,
            report.timestamp.isoformat(),
            report.total_test_cases,
            f"{report.validity_rate:.3f}",
            f"{report.coverage_metrics.line_coverage:.1f}",
            f"{report.coverage_metrics.branch_coverage:.1f}",
            len(report.defects_found),
            security_defects,
            f"{report.performance_metrics.total_execution_time:.2f}"
        ])
        
        return output.getvalue()