"""Token prioritization module for security-relevant token scoring.

Enhanced with dynamic priority management capabilities.
"""

from typing import List, Optional, Dict, Any
from src.data_models import Token
from src.priority.dynamic_manager import DynamicPriorityManager
from src.priority.metrics import PriorityMetrics
from src.utils import get_logger, format_token



class TokenPrioritizer:
    """Prioritizes tokens based on security relevance and context.
    
    Enhanced version with dynamic priority management for balanced
    exploration and exploitation.
    """
    
    def __init__(self, 
                 security_keywords=None, 
                 validation_keywords=None,
                 use_dynamic_manager: bool = True,
                 dynamic_config: Optional[Dict[str, Any]] = None):
        """Initialize the prioritizer with keyword lists.
        
        Args:
            security_keywords: List of security-related keywords
            validation_keywords: List of validation-related keywords
            use_dynamic_manager: Whether to use dynamic priority management
            dynamic_config: Configuration for dynamic manager
        """
        self.security_keywords = security_keywords or [
            "validate", "auth", "authenticate", "authorization", "sanitize",
            "escape", "filter", "verify", "check", "secure", "security",
            "credential", "password", "token", "session", "permission"
        ]
        
        self.validation_keywords = validation_keywords or [
            "input", "parameter", "argument", "boundary", "range",
            "limit", "constraint", "validation", "verify", "check"
        ]
        
        # 动态优先级管理
        self.use_dynamic_manager = use_dynamic_manager
        if use_dynamic_manager:
            self.dynamic_manager = DynamicPriorityManager(dynamic_config)
            self.metrics = PriorityMetrics()
        else:
            self.dynamic_manager = None
            self.metrics = None
        
        self.logger = get_logger()
    
    def assign_scores(self, tokens: List[Token]) -> List[Token]:
        """Assign heuristic priority scores to each token.
        
        This method assigns base priority scores using heuristic rules.
        If dynamic manager is enabled, final priorities will be calculated
        dynamically during token selection.
        
        Args:
            tokens: List of Token objects
            
        Returns:
            List of Token objects with updated priority scores
        """
        for token in tokens:
            # 计算基础分数
            score = self._calculate_base_score(token)
            token.priority_score = score

        sample_tokens = tokens[: min(5, len(tokens))]
        sample_summary = ", ".join(format_token(t) for t in sample_tokens)
        if self.use_dynamic_manager:
            self.logger.info(
                f"Assigned base scores to {len(tokens)} tokens (dynamic mode). Sample: {sample_summary}"
            )
        else:
            self.logger.info(
                f"Assigned final scores to {len(tokens)} tokens (legacy mode). Sample: {sample_summary}"
            )

        return tokens

    
    def _calculate_base_score(self, token: Token) -> float:
        """Calculate base priority score using heuristic rules.
        
        Args:
            token: Token object
            
        Returns:
            Base priority score
        """
        score = 0.0
        
        # Base score by token type
        if token.token_type == "phrase":
            score += 3.0  # Technical phrases are valuable
        elif token.token_type == "noun":
            score += 2.0
        elif token.token_type == "verb":
            score += 1.5
        
        # Security-related bonus
        if self.is_security_related(token):
            score += 5.0
        
        # Validation-related bonus
        if self.is_validation_related(token):
            score += 4.0
        
        # Boundary check context bonus
        if self.is_boundary_check(token):
            score += 3.0
        
        return score
    
    def is_security_related(self, token: Token) -> bool:
        """Check if token contains security-related keywords.
        
        Args:
            token: Token object
            
        Returns:
            True if token is security-related
        """
        text_lower = token.text.lower()
        return any(keyword in text_lower for keyword in self.security_keywords)
    
    def is_validation_related(self, token: Token) -> bool:
        """Check if token is related to input validation.
        
        Args:
            token: Token object
            
        Returns:
            True if token is validation-related
        """
        text_lower = token.text.lower()
        return any(keyword in text_lower for keyword in self.validation_keywords)
    
    def is_boundary_check(self, token: Token) -> bool:
        """Check if token is in conditional or boundary-check context.
        
        Args:
            token: Token object
            
        Returns:
            True if token is in boundary check context
        """
        text_lower = token.text.lower()
        boundary_terms = ["limit", "boundary", "range", "max", "min", "threshold"]
        return any(term in text_lower for term in boundary_terms)
    
    def rank_tokens(self, tokens: List[Token]) -> List[Token]:
        """Sort tokens by priority score in descending order.
        
        If dynamic manager is enabled, uses dynamic priority calculation.
        Otherwise, uses static priority scores.
        
        Args:
            tokens: List of Token objects with priority scores
            
        Returns:
            Sorted list of Token objects
        """
        if self.use_dynamic_manager and self.dynamic_manager:
            # 使用动态优先级排序
            def get_dynamic_priority(token):
                dynamic_manager = self.dynamic_manager
                if dynamic_manager is None:
                    return token.priority_score
                calculate = getattr(dynamic_manager, "calculate_priority", None)
                if calculate is None:
                    return token.priority_score
                return calculate(token)

            
            return sorted(tokens, key=get_dynamic_priority, reverse=True)
        else:
            # 使用静态优先级排序 (原有逻辑)
            return sorted(tokens, key=lambda t: t.priority_score, reverse=True)
    
    def select_tokens_for_perturbation(self, tokens: List[Token], n: int) -> List[Token]:
        """Select tokens for perturbation using dynamic strategy.
        
        This is the main entry point for token selection with dynamic
        priority management.
        
        Args:
            tokens: List of candidate tokens
            n: Number of tokens to select
            
        Returns:
            Selected tokens for perturbation
        """
        if not self.use_dynamic_manager or not self.dynamic_manager:
            # 回退到简单的Top-N选择
            ranked = self.rank_tokens(tokens)
            return ranked[:n]
        
        # 使用动态管理器选择
        selected = self.dynamic_manager.select_tokens_for_perturbation(tokens, n)
        
        # 记录指标
        if self.metrics:
            metrics = self.metrics.track_selection(
                selected, tokens, 
                self.dynamic_manager.current_iteration,
                "dynamic"
            )
            self.logger.debug(f"Selection metrics: diversity={metrics['diversity_index']:.2f}, "
                            f"exploration={metrics['exploration_ratio']:.2f}")
        
        return selected
    
    def update_after_perturbation(self, token: Token, impact: float):
        """Update token state after perturbation.
        
        Args:
            token: Token that was perturbed
            impact: Impact score of the perturbation (0.0-1.0)
        """
        if self.use_dynamic_manager and self.dynamic_manager:
            self.dynamic_manager.update_after_perturbation(token, impact)
    
    def update_iteration(self, iteration: int, has_feedback: bool = False):
        """Update current iteration and stage.
        
        Args:
            iteration: Current iteration number
            has_feedback: Whether feedback data is available
        """
        if self.use_dynamic_manager and self.dynamic_manager:
            self.dynamic_manager.update_stage(iteration, has_feedback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        if self.use_dynamic_manager and self.dynamic_manager:
            metrics.update(self.dynamic_manager.get_metrics())
        
        if self.metrics:
            summary = self.metrics.get_summary_report()
            metrics.update(summary)
        
        return metrics
    
    def reset(self):
        """Reset prioritizer state for new experiment.
        
        Clears all history and resets to initial state.
        """
        if self.use_dynamic_manager and self.dynamic_manager:
            self.dynamic_manager.reset()
        
        if self.metrics:
            self.metrics.reset()
        
        self.logger.info("Token prioritizer reset")
