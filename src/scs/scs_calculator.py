"""SCS (Semantic Contribution Score) calculator."""

import math
from typing import Optional, Dict, List
from src.scs.data_models import FeedbackData, SCSConfig


class SCSCalculator:
    """Calculates Semantic Contribution Scores based on feedback data.
    
    The SCS score measures how much a token perturbation impacts fuzzer
    effectiveness, using a weighted combination of validity, coverage, and
    defect detection metrics.
    """
    
    # Small epsilon to avoid division by zero
    EPSILON = 1e-6
    
    def __init__(
        self,
        validity_weight: float = 0.40,
        coverage_weight: float = 0.35,
        defect_weight: float = 0.25,
        baseline_metrics: Optional[FeedbackData] = None
    ):
        """Initialize calculator with weights and baseline.
        
        Args:
            validity_weight: Weight for validity metric (default 0.40)
            coverage_weight: Weight for coverage metric (default 0.35)
            defect_weight: Weight for defect detection metric (default 0.25)
            baseline_metrics: Optional baseline feedback data
        """
        self.validity_weight = validity_weight
        self.coverage_weight = coverage_weight
        self.defect_weight = defect_weight
        self.baseline_metrics = baseline_metrics
        
        # Validate weights sum to 1.0
        weight_sum = validity_weight + coverage_weight + defect_weight
        if abs(weight_sum - 1.0) >= 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.4f}"
            )
    
    def calculate_scs(
        self,
        baseline: FeedbackData,
        perturbed: FeedbackData
    ) -> float:
        """Calculate SCS score for a token.
        
        The SCS score represents the impact of perturbation on fuzzer
        effectiveness, normalized to 0-100 range. Higher scores indicate
        greater impact.
        
        Args:
            baseline: Baseline fuzzer metrics (without perturbation)
            perturbed: Metrics after perturbation
            
        Returns:
            SCS score (0-100)
        """
        # Calculate deltas for each metric
        delta_validity = self._calculate_delta(
            baseline.validity_rate,
            perturbed.validity_rate
        )
        
        delta_coverage = self._calculate_delta(
            baseline.coverage_percent,
            perturbed.coverage_percent
        )
        
        delta_defects = self._calculate_delta(
            float(baseline.defects_found),
            float(perturbed.defects_found)
        )
        
        # Calculate weighted raw score
        raw_score = (
            delta_validity * self.validity_weight +
            delta_coverage * self.coverage_weight +
            delta_defects * self.defect_weight
        )
        
        # Normalize to 0-100 range
        scs_score = self._normalize_score(raw_score)
        
        return scs_score
    
    def _calculate_delta(
        self,
        baseline_value: float,
        perturbed_value: float
    ) -> float:
        """Calculate normalized delta between baseline and perturbed.
        
        Delta represents the relative change: (baseline - perturbed) / baseline
        A positive delta means performance degraded (perturbed < baseline).
        
        Args:
            baseline_value: Baseline metric value
            perturbed_value: Perturbed metric value
            
        Returns:
            Normalized delta (0.0 to 1.0+)
        """
        # Handle zero baseline with epsilon
        if abs(baseline_value) < self.EPSILON:
            baseline_value = self.EPSILON
        
        # Calculate relative change
        delta = (baseline_value - perturbed_value) / baseline_value
        
        # Handle NaN/Inf
        if math.isnan(delta) or math.isinf(delta):
            return 0.0
        
        # Clamp to reasonable range (allow > 1.0 for severe degradation)
        return max(0.0, delta)
    
    def _normalize_score(self, raw_score: float) -> float:
        """Normalize score to 0-100 range.
        
        Args:
            raw_score: Raw weighted score (typically 0.0 to 1.0+)
            
        Returns:
            Normalized score (0-100)
        """
        # Handle NaN/Inf
        if math.isnan(raw_score) or math.isinf(raw_score):
            return 0.0
        
        # Scale to 0-100 and clamp
        normalized = raw_score * 100.0
        return max(0.0, min(100.0, normalized))
    
    def calculate_scs_batch(
        self,
        baseline: FeedbackData,
        perturbed_list: List[FeedbackData]
    ) -> List[float]:
        """Calculate SCS scores for multiple perturbations.
        
        Args:
            baseline: Baseline fuzzer metrics
            perturbed_list: List of perturbed metrics
            
        Returns:
            List of SCS scores
        """
        return [
            self.calculate_scs(baseline, perturbed)
            for perturbed in perturbed_list
        ]
    
    def aggregate_scs_scores(
        self,
        scs_scores: List[float],
        method: str = "max"
    ) -> float:
        """Aggregate multiple SCS scores for the same token.
        
        When multiple perturbations are applied to the same token,
        we aggregate their SCS scores. Default is to use maximum.
        
        Args:
            scs_scores: List of SCS scores
            method: Aggregation method ('max', 'mean', 'median')
            
        Returns:
            Aggregated SCS score
        """
        if not scs_scores:
            return 0.0
        
        if method == "max":
            return max(scs_scores)
        elif method == "mean":
            return sum(scs_scores) / len(scs_scores)
        elif method == "median":
            sorted_scores = sorted(scs_scores)
            n = len(sorted_scores)
            if n % 2 == 0:
                return (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2.0
            else:
                return sorted_scores[n//2]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    @classmethod
    def from_config(cls, config: SCSConfig) -> 'SCSCalculator':
        """Create SCSCalculator from SCSConfig.
        
        Args:
            config: SCS configuration
            
        Returns:
            SCSCalculator instance
        """
        baseline_metrics = config.get_baseline_feedback()
        return cls(
            validity_weight=config.validity_weight,
            coverage_weight=config.coverage_weight,
            defect_weight=config.defect_weight,
            baseline_metrics=baseline_metrics
        )
