"""Advanced coverage tracking system for fuzzer simulation."""

import random
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from src.fuzzer.data_models import TestCase, TestType, CoverageMetrics


class CoverageType(Enum):
    """Types of coverage metrics."""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    API_ENDPOINT = "api_endpoint"
    CONDITION = "condition"
    PATH = "path"


@dataclass
class CoveragePoint:
    """Individual coverage point (line, branch, etc.)."""
    id: str
    type: CoverageType
    location: str
    hit_count: int = 0
    first_hit_test: Optional[str] = None
    first_hit_time: Optional[datetime] = None
    complexity_weight: float = 1.0


@dataclass
class CoverageReport:
    """Detailed coverage report with analysis."""
    session_id: str
    total_coverage: float
    coverage_by_type: Dict[CoverageType, float] = field(default_factory=dict)
    coverage_points: List[CoveragePoint] = field(default_factory=list)
    uncovered_areas: List[str] = field(default_factory=list)
    coverage_trend: List[Tuple[datetime, float]] = field(default_factory=list)
    hotspots: List[str] = field(default_factory=list)  # Frequently hit areas
    cold_spots: List[str] = field(default_factory=list)  # Never hit areas
    generated_at: datetime = field(default_factory=datetime.now)


class CoverageAnalyzer:
    """Analyzer for coverage patterns and insights."""
    
    def __init__(self):
        """Initialize coverage analyzer."""
        self.coverage_history: List[CoverageReport] = []
    
    def analyze_coverage_trend(self, coverage_points: List[CoveragePoint]) -> Dict[str, Any]:
        """Analyze coverage trends and patterns.
        
        Args:
            coverage_points: List of coverage points to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if not coverage_points:
            return {"trend": "no_data", "insights": []}
        
        # Calculate hit distribution
        hit_counts = [point.hit_count for point in coverage_points]
        total_hits = sum(hit_counts)
        avg_hits = total_hits / len(hit_counts) if hit_counts else 0
        
        # Find hotspots (>2x average hits)
        hotspots = [
            point.id for point in coverage_points 
            if point.hit_count > avg_hits * 2
        ]
        
        # Find cold spots (never hit)
        cold_spots = [
            point.id for point in coverage_points 
            if point.hit_count == 0
        ]
        
        # Calculate coverage distribution by type
        type_distribution = defaultdict(list)
        for point in coverage_points:
            type_distribution[point.type].append(point.hit_count)
        
        type_coverage = {}
        for coverage_type, hits in type_distribution.items():
            covered = sum(1 for h in hits if h > 0)
            type_coverage[coverage_type.value] = covered / len(hits) if hits else 0
        
        insights = []
        
        # Generate insights
        if len(hotspots) > len(coverage_points) * 0.1:
            insights.append("High concentration of test activity in specific areas")
        
        if len(cold_spots) > len(coverage_points) * 0.3:
            insights.append("Large portions of code remain untested")
        
        if type_coverage.get("branch", 0) < type_coverage.get("line", 0) * 0.8:
            insights.append("Branch coverage significantly lower than line coverage")
        
        return {
            "trend": "improving" if avg_hits > 1 else "needs_improvement",
            "total_hits": total_hits,
            "average_hits": avg_hits,
            "hotspots": hotspots[:10],  # Top 10 hotspots
            "cold_spots": cold_spots[:20],  # Top 20 cold spots
            "type_coverage": type_coverage,
            "insights": insights
        }
    
    def suggest_test_targets(self, coverage_points: List[CoveragePoint], 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Suggest areas that need more testing.
        
        Args:
            coverage_points: Current coverage points
            limit: Maximum number of suggestions
            
        Returns:
            List of test target suggestions
        """
        suggestions = []
        
        # Find uncovered high-complexity areas
        uncovered_complex = [
            point for point in coverage_points
            if point.hit_count == 0 and point.complexity_weight > 1.5
        ]
        
        for point in sorted(uncovered_complex, key=lambda p: p.complexity_weight, reverse=True)[:limit//2]:
            suggestions.append({
                "target": point.id,
                "reason": "High complexity, never tested",
                "priority": "high",
                "type": point.type.value,
                "complexity": point.complexity_weight
            })
        
        # Find lightly covered areas
        lightly_covered = [
            point for point in coverage_points
            if 0 < point.hit_count < 3 and point.complexity_weight > 1.0
        ]
        
        for point in sorted(lightly_covered, key=lambda p: p.complexity_weight, reverse=True)[:limit//2]:
            suggestions.append({
                "target": point.id,
                "reason": "Lightly tested, moderate complexity",
                "priority": "medium",
                "type": point.type.value,
                "complexity": point.complexity_weight,
                "hit_count": point.hit_count
            })
        
        return suggestions[:limit]


class AdvancedCoverageTracker:
    """Advanced coverage tracking with detailed metrics and analysis."""
    
    def __init__(self, session_id: str):
        """Initialize coverage tracker.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.coverage_points: Dict[str, CoveragePoint] = {}
        self.analyzer = CoverageAnalyzer()
        
        # Coverage statistics
        self.test_count = 0
        self.coverage_snapshots: List[Tuple[datetime, float]] = []
        
        # Initialize simulated code structure
        self._initialize_code_structure()
    
    def update_coverage(self, test_case: TestCase) -> CoverageMetrics:
        """Update coverage based on test case execution.
        
        Args:
            test_case: Executed test case
            
        Returns:
            Updated coverage metrics
        """
        self.test_count += 1
        current_time = datetime.now()
        
        # Determine coverage impact based on test characteristics
        coverage_impact = self._calculate_coverage_impact(test_case)
        
        # Update coverage points
        hit_points = self._simulate_coverage_hits(test_case, coverage_impact)
        
        for point_id in hit_points:
            if point_id in self.coverage_points:
                point = self.coverage_points[point_id]
                point.hit_count += 1
                
                # Record first hit
                if point.hit_count == 1:
                    point.first_hit_test = test_case.id
                    point.first_hit_time = current_time
        
        # Calculate current metrics
        current_metrics = self._calculate_current_metrics()
        
        # Record coverage snapshot
        self.coverage_snapshots.append((current_time, current_metrics.line_coverage))
        
        return current_metrics
    
    def get_coverage_report(self) -> CoverageReport:
        """Generate comprehensive coverage report.
        
        Returns:
            Detailed coverage report
        """
        current_metrics = self._calculate_current_metrics()
        analysis = self.analyzer.analyze_coverage_trend(list(self.coverage_points.values()))
        
        return CoverageReport(
            session_id=self.session_id,
            total_coverage=current_metrics.line_coverage,
            coverage_by_type={
                CoverageType.LINE: current_metrics.line_coverage,
                CoverageType.BRANCH: current_metrics.branch_coverage,
                CoverageType.FUNCTION: current_metrics.function_coverage,
                CoverageType.API_ENDPOINT: current_metrics.api_endpoint_coverage
            },
            coverage_points=list(self.coverage_points.values()),
            uncovered_areas=analysis.get("cold_spots", []),
            coverage_trend=self.coverage_snapshots.copy(),
            hotspots=analysis.get("hotspots", []),
            cold_spots=analysis.get("cold_spots", [])
        )
    
    def get_incremental_coverage(self, previous_metrics: Optional[CoverageMetrics] = None) -> CoverageMetrics:
        """Get incremental coverage since last measurement.
        
        Args:
            previous_metrics: Previous coverage metrics for comparison
            
        Returns:
            Incremental coverage metrics
        """
        current_metrics = self._calculate_current_metrics()
        
        if previous_metrics is None:
            return current_metrics
        
        # Calculate incremental coverage
        incremental_lines = current_metrics.covered_lines - previous_metrics.covered_lines
        incremental_branches = current_metrics.covered_branches - previous_metrics.covered_branches
        
        return CoverageMetrics(
            line_coverage=len(incremental_lines) / max(1, current_metrics.total_lines),
            branch_coverage=len(incremental_branches) / max(1, current_metrics.total_branches),
            function_coverage=current_metrics.function_coverage - previous_metrics.function_coverage,
            api_endpoint_coverage=current_metrics.api_endpoint_coverage - previous_metrics.api_endpoint_coverage,
            covered_lines=incremental_lines,
            total_lines=current_metrics.total_lines,
            covered_branches=incremental_branches,
            total_branches=current_metrics.total_branches
        )
    
    def export_coverage_data(self, format: str = "json") -> str:
        """Export coverage data in specified format.
        
        Args:
            format: Export format ("json", "csv", "xml")
            
        Returns:
            Formatted coverage data
        """
        report = self.get_coverage_report()
        
        if format.lower() == "json":
            return self._export_json(report)
        elif format.lower() == "csv":
            return self._export_csv(report)
        elif format.lower() == "xml":
            return self._export_xml(report)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_coverage(self) -> None:
        """Reset all coverage data."""
        for point in self.coverage_points.values():
            point.hit_count = 0
            point.first_hit_test = None
            point.first_hit_time = None
        
        self.test_count = 0
        self.coverage_snapshots.clear()
    
    def _initialize_code_structure(self) -> None:
        """Initialize simulated code structure with coverage points."""
        
        # Create line coverage points
        for i in range(1, 1001):  # 1000 lines of code
            complexity = self._calculate_line_complexity(i)
            self.coverage_points[f"line_{i}"] = CoveragePoint(
                id=f"line_{i}",
                type=CoverageType.LINE,
                location=f"file.py:{i}",
                complexity_weight=complexity
            )
        
        # Create branch coverage points
        for i in range(1, 201):  # 200 branches
            complexity = random.uniform(1.0, 3.0)
            self.coverage_points[f"branch_{i}"] = CoveragePoint(
                id=f"branch_{i}",
                type=CoverageType.BRANCH,
                location=f"branch_{i}",
                complexity_weight=complexity
            )
        
        # Create function coverage points
        for i in range(1, 51):  # 50 functions
            complexity = random.uniform(1.0, 4.0)
            self.coverage_points[f"function_{i}"] = CoveragePoint(
                id=f"function_{i}",
                type=CoverageType.FUNCTION,
                location=f"function_{i}",
                complexity_weight=complexity
            )
        
        # Create API endpoint coverage points
        endpoints = [
            "user_create", "user_update", "user_delete", "user_get",
            "auth_login", "auth_logout", "auth_refresh",
            "data_search", "data_filter", "data_export",
            "admin_config", "admin_users", "admin_logs",
            "file_upload", "file_download", "file_delete",
            "report_generate", "report_schedule", "report_view",
            "notification_send", "notification_list"
        ]
        
        for endpoint in endpoints:
            complexity = random.uniform(1.5, 3.5)
            self.coverage_points[f"endpoint_{endpoint}"] = CoveragePoint(
                id=f"endpoint_{endpoint}",
                type=CoverageType.API_ENDPOINT,
                location=endpoint,
                complexity_weight=complexity
            )
    
    def _calculate_line_complexity(self, line_number: int) -> float:
        """Calculate complexity weight for a line of code.
        
        Args:
            line_number: Line number
            
        Returns:
            Complexity weight
        """
        # Simulate different complexity patterns
        if line_number % 100 < 10:  # Error handling code
            return random.uniform(2.0, 4.0)
        elif line_number % 50 < 5:  # Complex logic
            return random.uniform(1.5, 3.0)
        else:  # Regular code
            return random.uniform(0.5, 1.5)
    
    def _calculate_coverage_impact(self, test_case: TestCase) -> float:
        """Calculate how much coverage this test case should generate.
        
        Args:
            test_case: Test case to analyze
            
        Returns:
            Coverage impact factor (0.0 to 1.0)
        """
        base_impact = 0.1  # Base coverage for any test
        
        # Impact based on test type
        type_impacts = {
            TestType.NORMAL: 0.3,
            TestType.EDGE: 0.5,
            TestType.SECURITY: 0.7,
            TestType.MALFORMED: 0.2
        }
        
        type_impact = type_impacts.get(test_case.test_type, 0.3)
        
        # Impact based on parameter complexity
        param_impact = min(0.3, len(test_case.parameters) * 0.05)
        
        # Random variation
        random_factor = random.uniform(0.8, 1.2)
        
        return min(1.0, (base_impact + type_impact + param_impact) * random_factor)
    
    def _simulate_coverage_hits(self, test_case: TestCase, coverage_impact: float) -> Set[str]:
        """Simulate which coverage points are hit by this test.
        
        Args:
            test_case: Test case being executed
            coverage_impact: Coverage impact factor
            
        Returns:
            Set of coverage point IDs that were hit
        """
        hit_points = set()
        
        # Calculate number of points to hit
        total_points = len(self.coverage_points)
        points_to_hit = int(total_points * coverage_impact * random.uniform(0.5, 1.5))
        
        # Select points to hit based on various factors
        available_points = list(self.coverage_points.keys())
        
        # Prefer API endpoint related to test
        api_related = [
            point_id for point_id in available_points
            if f"endpoint_{test_case.api_name}" in point_id
        ]
        
        if api_related:
            hit_points.update(random.sample(api_related, min(len(api_related), max(1, points_to_hit // 10))))
        
        # Add random selection of other points
        remaining_points = points_to_hit - len(hit_points)
        if remaining_points > 0:
            other_points = [p for p in available_points if p not in hit_points]
            if other_points:
                hit_points.update(random.sample(other_points, min(len(other_points), remaining_points)))
        
        return hit_points
    
    def _calculate_current_metrics(self) -> CoverageMetrics:
        """Calculate current coverage metrics.
        
        Returns:
            Current coverage metrics
        """
        # Count covered points by type
        line_points = [p for p in self.coverage_points.values() if p.type == CoverageType.LINE]
        branch_points = [p for p in self.coverage_points.values() if p.type == CoverageType.BRANCH]
        function_points = [p for p in self.coverage_points.values() if p.type == CoverageType.FUNCTION]
        endpoint_points = [p for p in self.coverage_points.values() if p.type == CoverageType.API_ENDPOINT]
        
        covered_lines = {int(p.id.split('_')[1]) for p in line_points if p.hit_count > 0}
        covered_branches = {p.id for p in branch_points if p.hit_count > 0}
        
        line_coverage = len(covered_lines) / len(line_points) if line_points else 0
        branch_coverage = len(covered_branches) / len(branch_points) if branch_points else 0
        function_coverage = sum(1 for p in function_points if p.hit_count > 0) / len(function_points) if function_points else 0
        endpoint_coverage = sum(1 for p in endpoint_points if p.hit_count > 0) / len(endpoint_points) if endpoint_points else 0
        
        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            api_endpoint_coverage=endpoint_coverage,
            covered_lines=covered_lines,
            total_lines=len(line_points),
            covered_branches=covered_branches,
            total_branches=len(branch_points)
        )
    
    def _export_json(self, report: CoverageReport) -> str:
        """Export coverage report as JSON.
        
        Args:
            report: Coverage report to export
            
        Returns:
            JSON formatted string
        """
        export_data = {
            "session_id": report.session_id,
            "generated_at": report.generated_at.isoformat(),
            "total_coverage": report.total_coverage,
            "coverage_by_type": {k.value: v for k, v in report.coverage_by_type.items()},
            "test_count": self.test_count,
            "coverage_points": [
                {
                    "id": point.id,
                    "type": point.type.value,
                    "location": point.location,
                    "hit_count": point.hit_count,
                    "complexity_weight": point.complexity_weight,
                    "first_hit_test": point.first_hit_test,
                    "first_hit_time": point.first_hit_time.isoformat() if point.first_hit_time else None
                }
                for point in report.coverage_points
            ],
            "hotspots": report.hotspots,
            "cold_spots": report.cold_spots,
            "coverage_trend": [
                {"timestamp": ts.isoformat(), "coverage": cov}
                for ts, cov in report.coverage_trend
            ]
        }
        
        return json.dumps(export_data, indent=2)
    
    def _export_csv(self, report: CoverageReport) -> str:
        """Export coverage report as CSV.
        
        Args:
            report: Coverage report to export
            
        Returns:
            CSV formatted string
        """
        lines = [
            "point_id,type,location,hit_count,complexity_weight,first_hit_test,first_hit_time"
        ]
        
        for point in report.coverage_points:
            first_hit_time = point.first_hit_time.isoformat() if point.first_hit_time else ""
            lines.append(
                f"{point.id},{point.type.value},{point.location},"
                f"{point.hit_count},{point.complexity_weight},"
                f"{point.first_hit_test or ''},{first_hit_time}"
            )
        
        return "\n".join(lines)
    
    def _export_xml(self, report: CoverageReport) -> str:
        """Export coverage report as XML.
        
        Args:
            report: Coverage report to export
            
        Returns:
            XML formatted string
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<coverage_report>',
            f'  <session_id>{report.session_id}</session_id>',
            f'  <generated_at>{report.generated_at.isoformat()}</generated_at>',
            f'  <total_coverage>{report.total_coverage}</total_coverage>',
            f'  <test_count>{self.test_count}</test_count>',
            '  <coverage_points>'
        ]
        
        for point in report.coverage_points:
            first_hit_time = point.first_hit_time.isoformat() if point.first_hit_time else ""
            lines.extend([
                '    <point>',
                f'      <id>{point.id}</id>',
                f'      <type>{point.type.value}</type>',
                f'      <location>{point.location}</location>',
                f'      <hit_count>{point.hit_count}</hit_count>',
                f'      <complexity_weight>{point.complexity_weight}</complexity_weight>',
                f'      <first_hit_test>{point.first_hit_test or ""}</first_hit_test>',
                f'      <first_hit_time>{first_hit_time}</first_hit_time>',
                '    </point>'
            ])
        
        lines.extend([
            '  </coverage_points>',
            '</coverage_report>'
        ])
        
        return "\n".join(lines)