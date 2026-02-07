"""Feedback loop for iterative defect analysis and priority optimization."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.data_models import Token
from src.feedback.data_models import DefectInfo, FeedbackReport
from src.feedback.defect_analyzer import DefectAnalyzer
from src.feedback.feedback_agent import FeedbackAgent
from src.feedback.priority_adjuster import PriorityAdjuster
from src.utils import get_logger


class FeedbackLoop:
    """Manages iterative feedback loop for defect-driven optimization."""
    
    def __init__(
        self,
        document: str,
        tokens: List[Token],
        max_iterations: int = 3,
        convergence_threshold: float = 0.1
    ):
        """Initialize feedback loop.
        
        Args:
            document: Original document content
            tokens: Initial list of tokens
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence detection
        """
        self.document = document
        self.tokens = tokens
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize components
        self.defect_analyzer = DefectAnalyzer()
        self.feedback_agent = FeedbackAgent()
        self.priority_adjuster = PriorityAdjuster()
        
        self.logger = get_logger()
        
        # History tracking
        self.iteration_history = []
    
    def run_iteration(
        self,
        defects: List[DefectInfo],
        iteration: int
    ) -> Dict[str, Any]:
        """Run a single iteration of the feedback loop.
        
        Args:
            defects: List of defects to analyze
            iteration: Current iteration number
            
        Returns:
            Dictionary with iteration results
        """
        self.logger.info(f"Running feedback loop iteration {iteration + 1}/{self.max_iterations}")
        
        iteration_data = {
            'iteration': iteration,
            'num_defects': len(defects),
            'analyses': [],
            'adjustments': [],
            'top_tokens': []
        }
        
        # Analyze each defect
        for defect in defects:
            self.logger.info(f"Analyzing defect {defect.defect_id}...")
            
            # LLM analysis
            analysis = self.feedback_agent.analyze_defect(
                defect, 
                self.document, 
                self.tokens
            )
            
            # Adjust priorities
            self.tokens, adjustments = self.priority_adjuster.boost_priority(
                self.tokens,
                analysis
            )
            
            # Record results
            iteration_data['analyses'].append(analysis.to_dict())
            iteration_data['adjustments'].extend([adj.to_dict() for adj in adjustments])
        
        # Get top tokens after adjustment
        top_tokens = self.priority_adjuster.get_top_adjusted_tokens(self.tokens, n=10)
        iteration_data['top_tokens'] = [
            {
                'text': t.text,
                'type': t.token_type,
                'line': t.line,
                'priority': t.priority_score
            }
            for t in top_tokens
        ]
        
        self.iteration_history.append(iteration_data)
        
        return iteration_data
    
    def check_convergence(self) -> bool:
        """Check if the feedback loop has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.iteration_history) < 2:
            return False
        
        # Compare top tokens from last two iterations
        prev_top = self.iteration_history[-2]['top_tokens']
        curr_top = self.iteration_history[-1]['top_tokens']
        
        # Calculate change in priorities
        total_change = 0.0
        for prev, curr in zip(prev_top, curr_top):
            if prev['text'] == curr['text']:
                change = abs(curr['priority'] - prev['priority']) / prev['priority']
                total_change += change
        
        avg_change = total_change / len(prev_top) if prev_top else 0.0
        
        converged = avg_change < self.convergence_threshold
        
        if converged:
            self.logger.info(f"Converged: average priority change = {avg_change:.4f}")
        
        return converged
    
    def run(
        self,
        defects_per_iteration: List[List[DefectInfo]]
    ) -> tuple[List[Token], List[Dict[str, Any]]]:
        """Run the complete feedback loop.
        
        Args:
            defects_per_iteration: List of defect lists for each iteration
            
        Returns:
            Tuple of (final tokens, iteration history)
        """
        self.logger.info(f"Starting feedback loop (max {self.max_iterations} iterations)")
        
        for iteration in range(self.max_iterations):
            # Check if we have defects for this iteration
            if iteration >= len(defects_per_iteration):
                self.logger.info("No more defects to analyze, stopping")
                break
            
            defects = defects_per_iteration[iteration]
            
            if not defects:
                self.logger.info(f"No defects in iteration {iteration}, stopping")
                break
            
            # Run iteration
            self.run_iteration(defects, iteration)
            
            # Check convergence
            if self.check_convergence():
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        self.logger.info(f"Feedback loop completed after {len(self.iteration_history)} iterations")
        
        return self.tokens, self.iteration_history
    
    def save_results(self, output_path: str):
        """Save feedback loop results to file.
        
        Args:
            output_path: Path to save results
        """
        results = {
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold,
            'total_iterations': len(self.iteration_history),
            'history': self.iteration_history,
            'final_top_tokens': self.iteration_history[-1]['top_tokens'] if self.iteration_history else []
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
