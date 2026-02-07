"""LLM-based feedback simulator using the fuzzer simulator."""

from typing import Optional
from src.data_models import Token
from src.scs.data_models import FeedbackData, SCSConfig
from src.fuzzer import LLMFuzzerSimulator, FuzzerConfig
from src.utils.logger import get_logger


class LLMFeedbackSimulator:
    """LLM-based feedback simulator that uses the fuzzer simulator for realistic feedback.
    
    This replaces the simple FeedbackSimulator with a more sophisticated approach
    that actually runs LLM-based test generation and execution.
    """
    
    def __init__(
        self,
        fuzzer_config: Optional[FuzzerConfig] = None,
        scs_config: Optional[SCSConfig] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize LLM feedback simulator.
        
        Args:
            fuzzer_config: Configuration for the fuzzer simulator
            scs_config: SCS configuration for compatibility
            random_seed: Random seed for reproducible results
        """
        self.logger = get_logger("LLMFeedbackSimulator")
        
        # Create fuzzer config with random seed
        if fuzzer_config is None:
            fuzzer_config = FuzzerConfig()
        
        if random_seed is not None:
            fuzzer_config.random_seed = random_seed
        
        # Initialize the LLM fuzzer simulator
        self.fuzzer_simulator = LLMFuzzerSimulator(fuzzer_config)
        
        # Store SCS config for compatibility
        self.scs_config = scs_config or SCSConfig()
        
        self.logger.info("LLM Feedback Simulator initialized")
    
    def simulate_feedback(
        self,
        token: Token,
        perturbation_type: str = "generic",
        document_content: Optional[str] = None
    ) -> FeedbackData:
        """Simulate fuzzer feedback for a perturbed token using LLM fuzzer.
        
        Args:
            token: The token that was perturbed
            perturbation_type: Type of perturbation applied
            document_content: Document content with perturbation (if available)
            
        Returns:
            FeedbackData with realistic metrics from LLM fuzzer
        """
        self.logger.debug(f"Simulating LLM feedback for token: {token.text}")
        
        try:
            # If we have the actual perturbed document content, use it
            if document_content:
                feedback_data = self.fuzzer_simulator.simulate_feedback_for_token(
                    document_content, token.text
                )
            else:
                # Otherwise, create a minimal document for testing
                minimal_doc = self._create_minimal_document(token, perturbation_type)
                feedback_data = self.fuzzer_simulator.simulate_feedback_for_token(
                    minimal_doc, token.text
                )
            
            self.logger.debug(f"LLM feedback for {token.text}: "
                            f"validity={feedback_data.validity_rate:.3f}, "
                            f"coverage={feedback_data.coverage_percent:.1f}%, "
                            f"defects={feedback_data.defects_found}")
            
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"LLM feedback simulation failed for token {token.text}: {e}")
            
            # Fallback to simple simulation based on token characteristics
            return self._fallback_simulation(token, perturbation_type)
    
    def simulate_batch_feedback(
        self,
        tokens: list[Token],
        document_content: str,
        perturbation_type: str = "generic"
    ) -> list[FeedbackData]:
        """Simulate feedback for multiple tokens in batch for efficiency.
        
        Args:
            tokens: List of tokens to simulate feedback for
            document_content: Original document content
            perturbation_type: Type of perturbation applied
            
        Returns:
            List of FeedbackData objects
        """
        self.logger.info(f"Simulating batch LLM feedback for {len(tokens)} tokens")
        
        feedback_results = []
        
        for token in tokens:
            # Create perturbed document for this token
            perturbed_doc = self._apply_token_perturbation(
                document_content, token, perturbation_type
            )
            
            # Simulate feedback
            feedback = self.simulate_feedback(token, perturbation_type, perturbed_doc)
            feedback_results.append(feedback)
        
        return feedback_results
    
    def get_baseline_feedback(self) -> FeedbackData:
        """Get baseline feedback metrics for comparison.
        
        Returns:
            FeedbackData with baseline metrics
        """
        return FeedbackData.create_now(
            validity_rate=self.scs_config.baseline_validity,
            coverage_percent=self.scs_config.baseline_coverage,
            defects_found=self.scs_config.baseline_defects
        )
    
    @classmethod
    def from_config(
        cls,
        scs_config: SCSConfig,
        fuzzer_config: Optional[FuzzerConfig] = None,
        random_seed: Optional[int] = None
    ) -> 'LLMFeedbackSimulator':
        """Create LLMFeedbackSimulator from SCS configuration.
        
        Args:
            scs_config: SCS configuration
            fuzzer_config: Optional fuzzer configuration
            random_seed: Optional seed for reproducible results
            
        Returns:
            LLMFeedbackSimulator instance
        """
        return cls(
            fuzzer_config=fuzzer_config,
            scs_config=scs_config,
            random_seed=random_seed
        )
    
    def _create_minimal_document(self, token: Token, perturbation_type: str) -> str:
        """Create a minimal document for testing when no document content is provided."""
        # Create a simple API documentation that includes the token
        doc_template = f"""# API Documentation

## Authentication API

### authenticate(username, password)

Authenticates a user with the system using {token.text} validation.

**Parameters:**
- username (string): User's username
- password (string): User's password

**Returns:**
- success (boolean): Authentication result
- token (string): Authentication token if successful

**Example:**
```python
result = authenticate("user", "pass")
if result.success:
    print(f"Token: {{result.token}}")
```

The system performs {token.text} checks to ensure security.
"""
        
        return doc_template
    
    def _apply_token_perturbation(
        self,
        document_content: str,
        token: Token,
        perturbation_type: str
    ) -> str:
        """Apply perturbation to a specific token in the document."""
        # This is a simplified perturbation - in practice, you'd use the actual
        # perturbation strategies from the strategies module
        
        if perturbation_type == "zero_width":
            # Insert zero-width space in the middle of the token
            mid_point = len(token.text) // 2
            perturbed_token = (
                token.text[:mid_point] + 
                "\u200B" +  # Zero-width space
                token.text[mid_point:]
            )
        elif perturbation_type == "homoglyph":
            # Simple homoglyph replacement (a -> а)
            perturbed_token = token.text.replace('a', 'а')  # Latin 'a' -> Cyrillic 'а'
        else:
            # Generic perturbation - add comment
            perturbed_token = f"{token.text} <!-- perturbed -->"
        
        # Replace first occurrence of the token in the document
        return document_content.replace(token.text, perturbed_token, 1)
    
    def _fallback_simulation(self, token: Token, perturbation_type: str) -> FeedbackData:
        """Fallback simulation when LLM fuzzer fails."""
        import random
        
        # Use simple heuristics based on token characteristics
        text_lower = token.text.lower()
        
        # Determine impact based on token content
        if any(keyword in text_lower for keyword in ['auth', 'password', 'secure', 'validate']):
            # Security-related tokens have high impact
            validity_impact = random.uniform(0.7, 0.9)
        elif any(keyword in text_lower for keyword in ['input', 'parameter', 'check']):
            # Validation-related tokens have medium impact
            validity_impact = random.uniform(0.4, 0.6)
        else:
            # Other tokens have low impact
            validity_impact = random.uniform(0.1, 0.3)
        
        # Calculate degraded metrics
        baseline_validity = self.scs_config.baseline_validity
        baseline_coverage = self.scs_config.baseline_coverage
        baseline_defects = self.scs_config.baseline_defects
        
        validity = baseline_validity * (1.0 - validity_impact)
        coverage = baseline_coverage * (1.0 - validity_impact)
        defects = int(baseline_defects * (1.0 - validity_impact))
        
        # Apply some variance
        variance = self.scs_config.variance
        validity *= random.uniform(1.0 - variance, 1.0 + variance)
        coverage *= random.uniform(1.0 - variance, 1.0 + variance)
        defects = max(0, int(defects * random.uniform(1.0 - variance, 1.0 + variance)))
        
        # Clamp to valid ranges
        validity = max(0.0, min(1.0, validity))
        coverage = max(0.0, min(100.0, coverage))
        
        return FeedbackData.create_now(
            validity_rate=validity,
            coverage_percent=coverage,
            defects_found=defects
        )