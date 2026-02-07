"""Semantic Contribution Score (SCS) module for hotspot localization."""

from src.scs.data_models import FeedbackData, SCSConfig
from src.scs.llm_feedback_simulator import LLMFeedbackSimulator
from src.scs.scs_calculator import SCSCalculator
from src.scs.hotspot_analyzer import HotspotAnalyzer

# Keep old FeedbackSimulator for backward compatibility
from src.scs.feedback_simulator import FeedbackSimulator

__all__ = [
    'FeedbackData',
    'SCSConfig',
    'LLMFeedbackSimulator',  # New LLM-based simulator
    'FeedbackSimulator',     # Legacy simple simulator
    'SCSCalculator',
    'HotspotAnalyzer'
]
