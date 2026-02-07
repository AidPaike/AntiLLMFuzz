"""Feedback analysis module for defect-driven perturbation optimization.

This module analyzes fuzzer defects using LLM agents to identify critical
document locations and dynamically adjust perturbation priorities.
"""

from .data_models import DefectInfo, FeedbackAnalysis, CriticalLocation
from .defect_analyzer import DefectAnalyzer
from .feedback_agent import FeedbackAgent
from .priority_adjuster import PriorityAdjuster
from .feedback_loop import FeedbackLoop

__all__ = [
    'DefectInfo',
    'FeedbackAnalysis',
    'CriticalLocation',
    'DefectAnalyzer',
    'FeedbackAgent',
    'PriorityAdjuster',
    'FeedbackLoop',
]
