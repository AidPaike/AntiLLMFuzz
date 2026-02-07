"""Pipeline orchestrator - extracted from main.py to reduce class size."""

import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..extractors.base_extractor import BaseTokenExtractor
from ..extractors import DocumentationTokenExtractor, JavaTokenExtractor, PythonTokenExtractor
from ..strategies.base_strategy import PerturbationStrategy
from ..strategies.semantic import (
    ContradictoryInfoStrategy,
    EnhancedContradictoryStrategy,
    ReasoningDistractionStrategy,
    EvasiveSuffixStrategy,
    ContextPoisonStrategy,
    SemanticEvasionStrategy,
    RiskReframeStrategy,
)
from ..token_prioritizer import TokenPrioritizer
from ..utils import get_logger


class PipelineOrchestrator:
    """Orchestrates the pipeline execution steps."""
    
    def __init__(self, config, validator, reporter, processor):
        self.config = config
        self.validator = validator
        self.reporter = reporter
        self.processor = processor
        self.logger = get_logger(level=config.log_level)
    
    def execute_pipeline(self, input_file: Path) -> int:
        """Execute the complete pipeline."""
        start_time = time.time()
        self.reporter.log_header(input_file)
        
        # Validation and setup
        if not self.validator.validate_input_file(input_file):
            return 1
        
        pipeline_data = self._setup_pipeline_data(input_file)
        if not pipeline_data:
            return 1
        
        # Execute perturbations
        output_data = self._execute_perturbations(pipeline_data)
        
        # Generate reports
        total_time = time.time() - start_time
        self._generate_final_reports(pipeline_data, output_data, total_time)
        
        return 0
    
    def _setup_pipeline_data(self, input_file: Path) -> Optional[Dict[str, Any]]:
        """Set up all pipeline components and data."""
        if not input_file.exists():
            self.logger.error(f"Input file not found: {input_file}")
            return None

        content = input_file.read_text(encoding="utf-8")
        extractor: BaseTokenExtractor
        if input_file.suffix.lower() == ".java":
            extractor = JavaTokenExtractor()
        elif input_file.suffix.lower() == ".py":
            extractor = PythonTokenExtractor()
        else:
            extractor = DocumentationTokenExtractor()

        tokens = extractor.extract_tokens(str(input_file))
        if not tokens:
            self.logger.error("No tokens extracted; aborting pipeline")
            return None

        prioritizer = TokenPrioritizer()
        ranked_tokens = prioritizer.rank_tokens(prioritizer.assign_scores(tokens))

        strategies: List[PerturbationStrategy] = [
            ContradictoryInfoStrategy(),
            EnhancedContradictoryStrategy(),
            ReasoningDistractionStrategy(),
            EvasiveSuffixStrategy(),
            ContextPoisonStrategy(),
            SemanticEvasionStrategy(),
            RiskReframeStrategy(),
        ]

        return {
            "content": content,
            "tokens": tokens,
            "ranked_tokens": ranked_tokens,
            "strategies": strategies,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "input_file": input_file,
        }

    def _execute_perturbations(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute perturbations on the data."""
        ranked_tokens: List = pipeline_data["ranked_tokens"]
        content: str = pipeline_data["content"]
        strategies: List[PerturbationStrategy] = pipeline_data["strategies"]

        all_outputs: Dict[str, str] = {}
        details: List[Dict[str, Any]] = []

        top_tokens = ranked_tokens[: max(1, min(5, len(ranked_tokens)))]
        for strat in strategies:
            variants = strat.apply_multiple(top_tokens, content, max_tokens=len(top_tokens))
            all_outputs.update(variants)
            details.append({"strategy": strat.name, "variants": list(variants.keys())})

        return {"outputs": all_outputs, "details": details}

    def _generate_final_reports(self, pipeline_data: Dict[str, Any], 
                               output_data: Dict[str, Any], 
                               execution_time: float) -> None:
        """Generate all final reports."""
        count = len(output_data.get("outputs", {}))
        msg = f"Generated {count} perturbed variants in {execution_time:.2f}s"
        if hasattr(self.reporter, "log_info"):
            try:
                self.reporter.log_info(msg)
            except Exception:
                self.logger.info(msg)
        else:
            self.logger.info(msg)
        if hasattr(self.reporter, "write_summary"):
            try:
                self.reporter.write_summary(output_data, execution_time)
            except Exception:
                pass
