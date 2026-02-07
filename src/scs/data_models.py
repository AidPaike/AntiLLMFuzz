"""Data models for SCS calculation and feedback simulation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FeedbackData:
    """Represents fuzzer feedback metrics.
    
    Attributes:
        validity_rate: Proportion of valid test cases (0.0 to 1.0)
        coverage_percent: Code coverage percentage (0.0 to 100.0)
        defects_found: Number of defects detected (non-negative integer)
        timestamp: ISO format timestamp of feedback generation
    """
    validity_rate: float
    coverage_percent: float
    defects_found: int
    timestamp: str
    
    def __post_init__(self):
        """Validate feedback data ranges."""
        if not 0.0 <= self.validity_rate <= 1.0:
            raise ValueError(f"validity_rate must be in [0.0, 1.0], got {self.validity_rate}")
        if not 0.0 <= self.coverage_percent <= 100.0:
            raise ValueError(f"coverage_percent must be in [0.0, 100.0], got {self.coverage_percent}")
        if self.defects_found < 0:
            raise ValueError(f"defects_found must be non-negative, got {self.defects_found}")
    
    @staticmethod
    def create_now(validity_rate: float, coverage_percent: float, defects_found: int) -> 'FeedbackData':
        """Create FeedbackData with current timestamp.
        
        Args:
            validity_rate: Proportion of valid test cases
            coverage_percent: Code coverage percentage
            defects_found: Number of defects detected
            
        Returns:
            FeedbackData instance with current timestamp
        """
        return FeedbackData(
            validity_rate=validity_rate,
            coverage_percent=coverage_percent,
            defects_found=defects_found,
            timestamp=datetime.now().isoformat()
        )


@dataclass
class SCSConfig:
    """Configuration for SCS calculation.
    
    Attributes:
        validity_weight: Weight for validity metric (default 0.40)
        coverage_weight: Weight for coverage metric (default 0.35)
        defect_weight: Weight for defect detection metric (default 0.25)
        baseline_validity: Baseline validity rate for comparison (default 0.85)
        baseline_coverage: Baseline coverage percentage for comparison (default 65.0)
        baseline_defects: Baseline defect count for comparison (default 10)
        variance: Variance factor for simulation noise (default 0.1 = ±10%)
    """
    validity_weight: float = 0.40
    coverage_weight: float = 0.35
    defect_weight: float = 0.25
    baseline_validity: float = 0.85
    baseline_coverage: float = 65.0
    baseline_defects: int = 10
    variance: float = 0.1
    
    def validate(self) -> bool:
        """Validate that weights sum to 1.0 and all values are valid.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check weights sum to 1.0 (within tolerance)
        weight_sum = self.validity_weight + self.coverage_weight + self.defect_weight
        if abs(weight_sum - 1.0) >= 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.4f} "
                f"(validity={self.validity_weight}, coverage={self.coverage_weight}, "
                f"defect={self.defect_weight})"
            )
        
        # Check baseline metrics are non-negative
        if self.baseline_validity < 0:
            raise ValueError(f"baseline_validity must be non-negative, got {self.baseline_validity}")
        if self.baseline_coverage < 0:
            raise ValueError(f"baseline_coverage must be non-negative, got {self.baseline_coverage}")
        if self.baseline_defects < 0:
            raise ValueError(f"baseline_defects must be non-negative, got {self.baseline_defects}")
        
        # Check variance is in valid range
        if not 0.0 <= self.variance <= 1.0:
            raise ValueError(f"variance must be in [0.0, 1.0], got {self.variance}")
        
        return True
    
    def get_baseline_feedback(self) -> FeedbackData:
        """Get baseline metrics as FeedbackData.
        
        Returns:
            FeedbackData with baseline metrics
        """
        return FeedbackData.create_now(
            validity_rate=self.baseline_validity,
            coverage_percent=self.baseline_coverage,
            defects_found=self.baseline_defects
        )
