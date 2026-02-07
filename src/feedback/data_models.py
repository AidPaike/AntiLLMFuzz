"""Data models for feedback analysis module."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class DefectInfo:
    """Information about a defect triggered by the fuzzer.
    
    Attributes:
        defect_id: Unique identifier for the defect
        defect_type: Type of defect (crash, assertion, timeout, memory_leak, etc.)
        test_case: Test case that triggered the defect
        error_message: Error message from the fuzzer
        stack_trace: Stack trace if available
        severity: Severity level (critical, high, medium, low)
        timestamp: When the defect was detected
        metadata: Additional metadata about the defect
    """
    defect_id: str
    defect_type: str
    test_case: str
    error_message: str
    stack_trace: Optional[str] = None
    severity: str = "medium"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'defect_id': self.defect_id,
            'defect_type': self.defect_type,
            'test_case': self.test_case,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class CriticalLocation:
    """A critical location in the document identified by LLM analysis.
    
    Attributes:
        line_number: Line number in the document
        token_text: Text of the token at this location
        token_type: Type of token (noun, verb, phrase, etc.)
        relevance_score: How relevant this location is (0.0-1.0)
        reason: Explanation of why this location is critical
        suggested_strategies: Recommended perturbation strategies
    """
    line_number: int
    token_text: str
    token_type: str
    relevance_score: float
    reason: str
    suggested_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'line_number': self.line_number,
            'token_text': self.token_text,
            'token_type': self.token_type,
            'relevance_score': self.relevance_score,
            'reason': self.reason,
            'suggested_strategies': self.suggested_strategies
        }


@dataclass
class FeedbackAnalysis:
    """Result of LLM analysis on a defect.
    
    Attributes:
        defect_id: ID of the defect being analyzed
        root_cause: Root cause analysis from LLM
        critical_locations: List of critical locations identified
        reasoning: Detailed reasoning from LLM
        confidence: Overall confidence in the analysis (0.0-1.0)
        suggestions: List of suggestions for perturbation
        timestamp: When the analysis was performed
    """
    defect_id: str
    root_cause: str
    critical_locations: List[CriticalLocation]
    reasoning: str
    confidence: float
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'defect_id': self.defect_id,
            'root_cause': self.root_cause,
            'critical_locations': [loc.to_dict() for loc in self.critical_locations],
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp
        }
    
    def get_top_locations(self, n: int = 5) -> List[CriticalLocation]:
        """Get top N critical locations by relevance score.
        
        Args:
            n: Number of locations to return
            
        Returns:
            List of top N critical locations
        """
        sorted_locations = sorted(
            self.critical_locations,
            key=lambda loc: loc.relevance_score,
            reverse=True
        )
        return sorted_locations[:n]


@dataclass
class PriorityAdjustment:
    """Record of a priority adjustment made to a token.
    
    Attributes:
        token_text: Text of the token
        token_type: Type of token
        line_number: Line number of the token
        old_priority: Priority before adjustment
        new_priority: Priority after adjustment
        boost_factor: Factor by which priority was boosted
        reason: Reason for the adjustment
    """
    token_text: str
    token_type: str
    line_number: int
    old_priority: float
    new_priority: float
    boost_factor: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'token_text': self.token_text,
            'token_type': self.token_type,
            'line_number': self.line_number,
            'old_priority': self.old_priority,
            'new_priority': self.new_priority,
            'boost_factor': self.boost_factor,
            'reason': self.reason
        }


@dataclass
class FeedbackReport:
    """Complete report of feedback analysis and priority adjustments.
    
    Attributes:
        analysis_id: Unique identifier for this analysis
        defect_info: Information about the defect
        llm_analysis: LLM analysis result
        priority_adjustments: List of priority adjustments made
        timestamp: When the report was generated
        metadata: Additional metadata
    """
    analysis_id: str
    defect_info: DefectInfo
    llm_analysis: FeedbackAnalysis
    priority_adjustments: List[PriorityAdjustment]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'analysis_id': self.analysis_id,
            'defect_info': self.defect_info.to_dict(),
            'llm_analysis': self.llm_analysis.to_dict(),
            'priority_adjustments': [adj.to_dict() for adj in self.priority_adjustments],
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
