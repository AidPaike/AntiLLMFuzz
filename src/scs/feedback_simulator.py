"""Legacy feedback simulator for backward compatibility."""

import random
from typing import Optional
from src.data_models import Token
from src.scs.data_models import FeedbackData, SCSConfig
from src.utils.logger import get_logger


class FeedbackSimulator:
    """Legacy feedback simulator for backward compatibility.
    
    This is a simplified version that maintains the same interface as the original
    FeedbackSimulator but with reduced functionality. For new code, use
    LLMFeedbackSimulator instead.
    """
    
    def __init__(
        self,
        baseline_validity: float = 0.85,
        baseline_coverage: float = 65.0,
        baseline_defects: int = 10,
        variance: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """Initialize legacy feedback simulator.
        
        Args:
            baseline_validity: Baseline validity rate (0.0 to 1.0)
            baseline_coverage: Baseline coverage percentage (0.0 to 100.0)
            baseline_defects: Baseline defect count (non-negative)
            variance: Variance factor for noise (0.0 to 1.0, default 0.1 = ±10%)
            random_seed: Optional seed for reproducible results
        """
        self.baseline_validity = baseline_validity
        self.baseline_coverage = baseline_coverage
        self.baseline_defects = baseline_defects
        self.variance = variance
        
        self.logger = get_logger("FeedbackSimulator")
        self.logger.warning("Using legacy FeedbackSimulator. Consider upgrading to LLMFeedbackSimulator.")
        
        # Security and validation keywords (matching TokenPrioritizer)
        self.security_keywords = [
            "validate", "auth", "authenticate", "authorization", "sanitize",
            "escape", "filter", "verify", "check", "secure", "security",
            "credential", "password", "token", "session", "permission"
        ]
        
        self.validation_keywords = [
            "input", "parameter", "argument", "boundary", "range",
            "limit", "constraint", "validation", "verify", "check"
        ]
        
        self.boundary_keywords = [
            "limit", "boundary", "range", "max", "min", "threshold"
        ]
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def simulate_feedback(
        self,
        token: Token,
        perturbation_type: str = "generic"
    ) -> FeedbackData:
        """Simulate fuzzer feedback for a perturbed token.
        
        Args:
            token: The token that was perturbed
            perturbation_type: Type of perturbation applied
            
        Returns:
            FeedbackData with simulated metrics
        """
        # Calculate impact factor based on token characteristics
        impact_factor = self._calculate_impact_factor(token)
        
        # Simulate degraded metrics based on impact
        validity = self.baseline_validity * (1.0 - impact_factor)
        coverage = self.baseline_coverage * (1.0 - impact_factor)
        defects = int(self.baseline_defects * (1.0 - impact_factor))
        
        # Apply realistic variance
        validity = self._apply_variance(validity, self.variance)
        coverage = self._apply_variance(coverage, self.variance)
        defects = max(0, int(self._apply_variance(float(defects), self.variance)))
        
        # Clamp to valid ranges
        validity = max(0.0, min(1.0, validity))
        coverage = max(0.0, min(100.0, coverage))
        
        return FeedbackData.create_now(
            validity_rate=validity,
            coverage_percent=coverage,
            defects_found=defects
        )
    
    def _calculate_impact_factor(self, token: Token) -> float:
        """Calculate impact factor based on token characteristics."""
        text_lower = token.text.lower()
        
        # Security-related tokens have high impact (0.7-0.9)
        if self._is_security_related(text_lower):
            return random.uniform(0.7, 0.9)
        
        # Validation-related tokens have moderate impact (0.4-0.6)
        if self._is_validation_related(text_lower):
            return random.uniform(0.4, 0.6)
        
        # Boundary-check tokens have medium-high impact (0.5-0.7)
        if self._is_boundary_check(text_lower):
            return random.uniform(0.5, 0.7)
        
        # Other tokens have low impact (0.1-0.3)
        return random.uniform(0.1, 0.3)
    
    def _is_security_related(self, text_lower: str) -> bool:
        """Check if text contains security-related keywords."""
        return any(keyword in text_lower for keyword in self.security_keywords)
    
    def _is_validation_related(self, text_lower: str) -> bool:
        """Check if text contains validation-related keywords."""
        return any(keyword in text_lower for keyword in self.validation_keywords)
    
    def _is_boundary_check(self, text_lower: str) -> bool:
        """Check if text contains boundary-check keywords."""
        return any(keyword in text_lower for keyword in self.boundary_keywords)
    
    def _apply_variance(self, value: float, variance: float) -> float:
        """Apply random variance to a metric value."""
        factor = random.uniform(1.0 - variance, 1.0 + variance)
        return value * factor
    
    @classmethod
    def from_config(cls, config: SCSConfig, random_seed: Optional[int] = None) -> 'FeedbackSimulator':
        """Create FeedbackSimulator from SCSConfig."""
        return cls(
            baseline_validity=config.baseline_validity,
            baseline_coverage=config.baseline_coverage,
            baseline_defects=config.baseline_defects,
            variance=config.variance,
            random_seed=random_seed
        )