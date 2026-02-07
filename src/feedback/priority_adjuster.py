"""Priority adjuster for updating token priorities based on feedback analysis.

Enhanced with dynamic priority management integration.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from src.data_models import Token
from src.feedback.data_models import FeedbackAnalysis, CriticalLocation, PriorityAdjustment
from src.utils import get_logger

if TYPE_CHECKING:
    from src.priority.dynamic_manager import DynamicPriorityManager


class PriorityAdjuster:
    """Adjusts token priorities based on LLM feedback analysis.
    
    Enhanced version with dynamic priority management integration
    and score inflation prevention.
    """
    
    def __init__(
        self,
        boost_factor: float = 1.5,
        max_boost: float = 3.0,
        confidence_threshold: float = 0.7,
        dynamic_manager: Optional['DynamicPriorityManager'] = None,
        max_feedback_score: float = 20.0,
        decay_factor: float = 0.8
    ):
        """Initialize priority adjuster.
        
        Args:
            boost_factor: Base factor for boosting priorities
            max_boost: Maximum boost factor allowed
            confidence_threshold: Minimum confidence to apply boost
            dynamic_manager: Optional dynamic priority manager
            max_feedback_score: Maximum feedback score limit
            decay_factor: Decay factor to prevent score inflation
        """
        self.boost_factor = boost_factor
        self.max_boost = max_boost
        self.confidence_threshold = confidence_threshold
        self.dynamic_manager = dynamic_manager
        self.max_feedback_score = max_feedback_score
        self.decay_factor = decay_factor
        self.logger = get_logger()
    
    def boost_priority(
        self,
        tokens: List[Token],
        analysis: FeedbackAnalysis,
        apply_confidence_scaling: bool = True
    ) -> tuple[List[Token], List[PriorityAdjustment]]:
        """Boost priorities of tokens at critical locations.
        
        Enhanced version with dynamic manager integration and score inflation prevention.
        
        Args:
            tokens: List of tokens to adjust
            analysis: Feedback analysis with critical locations
            apply_confidence_scaling: Whether to scale boost by confidence
            
        Returns:
            Tuple of (updated tokens, list of adjustments made)
        """
        adjustments = []
        
        # Skip if confidence is too low
        if analysis.confidence < self.confidence_threshold:
            self.logger.warning(
                f"Analysis confidence ({analysis.confidence:.2f}) below threshold "
                f"({self.confidence_threshold}), skipping priority boost"
            )
            return tokens, adjustments
        
        # Create a map of line numbers to critical locations
        location_map = {loc.line_number: loc for loc in analysis.critical_locations}
        
        # Adjust priorities
        for token in tokens:
            if token.line in location_map:
                location = location_map[token.line]
                
                if self.dynamic_manager:
                    # 使用动态管理器的反馈机制
                    adjustment = self._apply_dynamic_feedback_boost(
                        token, location, analysis.confidence, apply_confidence_scaling
                    )
                else:
                    # 使用增强的传统方法
                    adjustment = self._apply_enhanced_boost(
                        token, location, analysis.confidence, apply_confidence_scaling
                    )
                
                if adjustment:
                    adjustments.append(adjustment)
        
        self.logger.info(f"Applied {len(adjustments)} priority adjustments")
        
        return tokens, adjustments
    
    def _apply_dynamic_feedback_boost(
        self,
        token: Token,
        location: CriticalLocation,
        confidence: float,
        apply_confidence_scaling: bool
    ) -> Optional[PriorityAdjustment]:
        """Apply feedback boost using dynamic manager.
        
        Args:
            token: Token to boost
            location: Critical location info
            confidence: Analysis confidence
            apply_confidence_scaling: Whether to scale by confidence
            
        Returns:
            PriorityAdjustment if boost was applied, None otherwise
        """
        # Calculate raw boost
        raw_boost = self.boost_factor * location.relevance_score
        if apply_confidence_scaling:
            raw_boost *= confidence
        
        # Apply boost through dynamic manager (with decay)
        old_feedback_score = token.feedback_score
        actual_boost = self.dynamic_manager.apply_feedback_boost(token, raw_boost)
        
        if actual_boost > 0:
            adjustment = PriorityAdjustment(
                token_text=token.text,
                token_type=token.token_type,
                line_number=token.line,
                old_priority=old_feedback_score,
                new_priority=token.feedback_score,
                boost_factor=actual_boost / raw_boost if raw_boost > 0 else 1.0,
                reason=location.reason
            )
            
            self.logger.debug(
                f"Dynamic boost '{token.text}': feedback_score "
                f"{old_feedback_score:.2f} -> {token.feedback_score:.2f} "
                f"(requested={raw_boost:.2f}, actual={actual_boost:.2f})"
            )
            
            return adjustment
        
        return None
    
    def _apply_enhanced_boost(
        self,
        token: Token,
        location: CriticalLocation,
        confidence: float,
        apply_confidence_scaling: bool
    ) -> Optional[PriorityAdjustment]:
        """Apply enhanced boost with decay (fallback method).
        
        Args:
            token: Token to boost
            location: Critical location info
            confidence: Analysis confidence
            apply_confidence_scaling: Whether to scale by confidence
            
        Returns:
            PriorityAdjustment if boost was applied, None otherwise
        """
        # Calculate boost factor
        boost = self.boost_factor
        boost *= location.relevance_score
        
        if apply_confidence_scaling:
            boost *= confidence
        
        # Apply decay to prevent inflation
        current_total = token.priority_score + token.scs_score + token.feedback_score
        decay = 1.0 / (1.0 + current_total / 50.0)
        boost *= decay * self.decay_factor
        
        # Cap at max boost
        boost = min(boost, self.max_boost)
        
        # Apply to feedback_score instead of priority_score
        old_feedback_score = token.feedback_score
        token.feedback_score = min(
            token.feedback_score + boost,
            self.max_feedback_score
        )
        
        actual_boost = token.feedback_score - old_feedback_score
        
        if actual_boost > 0:
            adjustment = PriorityAdjustment(
                token_text=token.text,
                token_type=token.token_type,
                line_number=token.line,
                old_priority=old_feedback_score,
                new_priority=token.feedback_score,
                boost_factor=boost,
                reason=location.reason
            )
            
            self.logger.debug(
                f"Enhanced boost '{token.text}': feedback_score "
                f"{old_feedback_score:.2f} -> {token.feedback_score:.2f} "
                f"(boost={boost:.2f}, decay={decay:.2f})"
            )
            
            return adjustment
        
        return None
    
    def apply_multiple_analyses(
        self,
        tokens: List[Token],
        analyses: List[FeedbackAnalysis]
    ) -> tuple[List[Token], List[PriorityAdjustment]]:
        """Apply multiple feedback analyses cumulatively.
        
        Args:
            tokens: List of tokens
            analyses: List of feedback analyses
            
        Returns:
            Tuple of (updated tokens, all adjustments made)
        """
        all_adjustments = []
        
        for analysis in analyses:
            tokens, adjustments = self.boost_priority(tokens, analysis)
            all_adjustments.extend(adjustments)
        
        return tokens, all_adjustments
    
    def get_top_adjusted_tokens(
        self,
        tokens: List[Token],
        n: int = 10
    ) -> List[Token]:
        """Get top N tokens after priority adjustment.
        
        Uses dynamic priority calculation if available.
        
        Args:
            tokens: List of tokens
            n: Number of top tokens to return
            
        Returns:
            List of top N tokens by priority
        """
        if self.dynamic_manager:
            # 使用动态优先级排序
            def get_final_priority(token):
                return self.dynamic_manager.calculate_priority(token)
            
            sorted_tokens = sorted(tokens, key=get_final_priority, reverse=True)
        else:
            # 使用综合分数排序 (priority + scs + feedback)
            def get_combined_score(token):
                return token.priority_score + token.scs_score + token.feedback_score
            
            sorted_tokens = sorted(tokens, key=get_combined_score, reverse=True)
        
        return sorted_tokens[:n]
    
    def set_dynamic_manager(self, manager: 'DynamicPriorityManager'):
        """Set the dynamic priority manager.
        
        Args:
            manager: Dynamic priority manager instance
        """
        self.dynamic_manager = manager
        self.logger.info("Dynamic priority manager attached to PriorityAdjuster")
