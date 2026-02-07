"""High-level API interface for LLM Fuzzer Simulator.

This module provides convenient methods for common fuzzing workflows
and integration with existing perturbation systems.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime

from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator
from src.fuzzer.data_models import FuzzerConfig, FeedbackReport, ComparisonReport
from src.fuzzer.base_interfaces import FuzzerError
from src.scs.data_models import FeedbackData
from src.utils.logger import get_logger


class FuzzerAPI:
    """High-level API for LLM Fuzzer Simulator with convenience methods."""
    
    def __init__(self, config: Optional[Union[FuzzerConfig, Dict[str, Any], str]] = None):
        """Initialize the Fuzzer API.
        
        Args:
            config: Fuzzer configuration as FuzzerConfig object, dict, or path to config file
        """
        self.logger = get_logger("FuzzerAPI")
        
        # Load configuration
        if config is None:
            self.config = FuzzerConfig()
        elif isinstance(config, FuzzerConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = FuzzerConfig.from_dict(config)
        elif isinstance(config, str):
            self.config = self._load_config_from_file(config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # Initialize simulator
        self.simulator = LLMFuzzerSimulator(self.config)
        
        self.logger.info("Fuzzer API initialized successfully")
    
    def fuzz_document(
        self, 
        document: Union[str, Path], 
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> FeedbackReport:
        """Fuzz a single document and return results.
        
        Args:
            document: Document content (str) or path to document file (Path/str)
            session_id: Optional session identifier
            progress_callback: Optional progress callback function
            
        Returns:
            FeedbackReport with fuzzing results
        """
        # Load document content if path provided
        if isinstance(document, (str, Path)) and Path(document).exists():
            document_path = Path(document)
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            document_path_str = str(document_path)
        else:
            content = str(document)
            document_path_str = None
        
        return self.simulator.run_fuzzing_session(
            document_content=content,
            session_id=session_id,
            document_path=document_path_str,
            progress_callback=progress_callback
        )
    
    def fuzz_documents_batch(
        self, 
        documents: List[Union[str, Path]],
        session_prefix: str = "batch",
        parallel: bool = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[FeedbackReport]:
        """Fuzz multiple documents in batch.
        
        Args:
            documents: List of document contents or paths
            session_prefix: Prefix for session IDs
            parallel: Whether to run in parallel (uses config default if None)
            progress_callback: Optional progress callback function
            
        Returns:
            List of FeedbackReport objects
        """
        # Load document contents
        document_contents = []
        for doc in documents:
            if isinstance(doc, (str, Path)) and Path(doc).exists():
                with open(doc, 'r', encoding='utf-8') as f:
                    document_contents.append(f.read())
            else:
                document_contents.append(str(doc))
        
        return self.simulator.run_batch_fuzzing(
            documents=document_contents,
            session_prefix=session_prefix,
            parallel=parallel,
            progress_callback=progress_callback
        )
    
    def compare_perturbation_impact(
        self, 
        original_document: Union[str, Path],
        perturbed_documents: List[Union[str, Path]],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ComparisonReport:
        """Compare fuzzing results between original and perturbed documents.
        
        Args:
            original_document: Original document content or path
            perturbed_documents: List of perturbed document contents or paths
            progress_callback: Optional progress callback function
            
        Returns:
            ComparisonReport with impact analysis
        """
        # Load original document
        if isinstance(original_document, (str, Path)) and Path(original_document).exists():
            with open(original_document, 'r', encoding='utf-8') as f:
                original_content = f.read()
        else:
            original_content = str(original_document)
        
        # Load perturbed documents
        perturbed_contents = []
        for doc in perturbed_documents:
            if isinstance(doc, (str, Path)) and Path(doc).exists():
                with open(doc, 'r', encoding='utf-8') as f:
                    perturbed_contents.append(f.read())
            else:
                perturbed_contents.append(str(doc))
        
        return self.simulator.compare_fuzzing_results(
            original_document=original_content,
            perturbed_documents=perturbed_contents,
            progress_callback=progress_callback
        )
    
    def analyze_single_perturbation(
        self, 
        original_document: Union[str, Path],
        perturbed_document: Union[str, Path],
        perturbation_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze the impact of a single perturbation.
        
        Args:
            original_document: Original document content or path
            perturbed_document: Perturbed document content or path
            perturbation_metadata: Optional metadata about the perturbation
            
        Returns:
            Dictionary with detailed impact analysis
        """
        # Load documents
        if isinstance(original_document, (str, Path)) and Path(original_document).exists():
            with open(original_document, 'r', encoding='utf-8') as f:
                original_content = f.read()
        else:
            original_content = str(original_document)
        
        if isinstance(perturbed_document, (str, Path)) and Path(perturbed_document).exists():
            with open(perturbed_document, 'r', encoding='utf-8') as f:
                perturbed_content = f.read()
        else:
            perturbed_content = str(perturbed_document)
        
        return self.simulator.analyze_perturbation_impact(
            original_document=original_content,
            perturbed_document=perturbed_content,
            perturbation_metadata=perturbation_metadata
        )
    
    def simulate_feedback_for_scs(
        self, 
        document_content: str, 
        token_text: str,
        token_metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackData:
        """Simulate fuzzer feedback for SCS (Semantic Contribution Score) calculation.
        
        This method provides integration with the existing SCS system.
        
        Args:
            document_content: Document content with token perturbation
            token_text: The perturbed token text
            token_metadata: Optional metadata about the token
            
        Returns:
            FeedbackData compatible with SCS system
        """
        return self.simulator.integrate_with_feedback_system(
            document_content=document_content,
            token_text=token_text,
            token_metadata=token_metadata
        )
    
    def run_batch_perturbation_analysis(
        self, 
        original_document: Union[str, Path],
        perturbation_configs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple perturbations in batch for efficiency.
        
        Args:
            original_document: Original document content or path
            perturbation_configs: List of perturbation configurations with 'perturbed_document' and 'metadata'
            progress_callback: Optional progress callback function
            
        Returns:
            List of impact analysis results
        """
        # Load original document
        if isinstance(original_document, (str, Path)) and Path(original_document).exists():
            with open(original_document, 'r', encoding='utf-8') as f:
                original_content = f.read()
        else:
            original_content = str(original_document)
        
        return self.simulator.batch_perturbation_analysis(
            original_document=original_content,
            perturbation_configs=perturbation_configs,
            progress_callback=progress_callback
        )
    
    def validate_deterministic_behavior(
        self, 
        document: Union[str, Path], 
        runs: int = 3
    ) -> Dict[str, Any]:
        """Validate that the fuzzer produces deterministic results.
        
        Args:
            document: Document content or path to test
            runs: Number of runs to perform
            
        Returns:
            Dictionary with validation results
        """
        # Load document content
        if isinstance(document, (str, Path)) and Path(document).exists():
            with open(document, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = str(document)
        
        return self.simulator.validate_deterministic_behavior(content, runs)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics from the simulator.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.simulator.get_performance_metrics()
    
    def reset_simulator_state(self) -> None:
        """Reset simulator state for fresh sessions."""
        self.simulator.reset_state()
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for deterministic execution.
        
        Args:
            seed: Random seed to set
        """
        self.simulator.set_random_seed(seed)
    
    def get_random_seed(self) -> int:
        """Get current random seed.
        
        Returns:
            Current random seed
        """
        return self.simulator.get_random_seed()
    
    def export_experiment_state(self) -> Dict[str, Any]:
        """Export current experiment state for reproduction.
        
        Returns:
            Dictionary with complete experiment state
        """
        return self.simulator.export_experiment_state()
    
    def import_experiment_state(self, state_dict: Dict[str, Any]) -> None:
        """Import experiment state for reproduction.
        
        Args:
            state_dict: State dictionary from export_experiment_state()
        """
        self.simulator.import_experiment_state(state_dict)
    
    def save_report(
        self, 
        report: Union[FeedbackReport, ComparisonReport, Dict[str, Any]], 
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """Save a report to file.
        
        Args:
            report: Report object or dictionary to save
            output_path: Path to save the report
            format: Output format ('json', 'yaml', 'csv')
        """
        output_path = Path(output_path)
        
        # Convert report to dictionary if needed
        if hasattr(report, 'to_dict'):
            report_data = report.to_dict()
        elif isinstance(report, dict):
            report_data = report
        else:
            raise ValueError(f"Unsupported report type: {type(report)}")
        
        # Save based on format
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(report_data, f, default_flow_style=False, allow_unicode=True)
        elif format.lower() == 'csv':
            self._save_report_as_csv(report_data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_report(self, report_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a report from file.
        
        Args:
            report_path: Path to the report file
            
        Returns:
            Report data as dictionary
        """
        report_path = Path(report_path)
        
        if not report_path.exists():
            raise FileNotFoundError(f"Report file not found: {report_path}")
        
        if report_path.suffix.lower() == '.json':
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif report_path.suffix.lower() in ['.yaml', '.yml']:
            with open(report_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported report format: {report_path.suffix}")
    
    def create_workflow_builder(self) -> 'WorkflowBuilder':
        """Create a workflow builder for complex fuzzing workflows.
        
        Returns:
            WorkflowBuilder instance
        """
        return WorkflowBuilder(self)
    
    def _load_config_from_file(self, config_path: str) -> FuzzerConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Extract fuzzer config if nested
        if 'llm_fuzzer' in config_dict:
            config_dict = config_dict['llm_fuzzer']
        
        return FuzzerConfig.from_dict(config_dict)
    
    def _save_report_as_csv(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Save report data as CSV."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write basic metrics
            writer.writerow(['Metric', 'Value'])
            
            # Flatten the report data for CSV
            def flatten_dict(d, prefix=''):
                items = []
                for k, v in d.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key))
                    elif isinstance(v, list):
                        items.append((new_key, f"[{len(v)} items]"))
                    else:
                        items.append((new_key, str(v)))
                return items
            
            for key, value in flatten_dict(report_data):
                writer.writerow([key, value])


class WorkflowBuilder:
    """Builder for creating complex fuzzing workflows."""
    
    def __init__(self, api: FuzzerAPI):
        """Initialize workflow builder.
        
        Args:
            api: FuzzerAPI instance
        """
        self.api = api
        self.steps = []
        self.logger = get_logger("WorkflowBuilder")
    
    def add_single_fuzz(
        self, 
        document: Union[str, Path], 
        session_id: Optional[str] = None
    ) -> 'WorkflowBuilder':
        """Add a single document fuzzing step.
        
        Args:
            document: Document content or path
            session_id: Optional session identifier
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            'type': 'single_fuzz',
            'document': document,
            'session_id': session_id
        })
        return self
    
    def add_batch_fuzz(
        self, 
        documents: List[Union[str, Path]], 
        session_prefix: str = "batch"
    ) -> 'WorkflowBuilder':
        """Add a batch fuzzing step.
        
        Args:
            documents: List of documents
            session_prefix: Session prefix
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            'type': 'batch_fuzz',
            'documents': documents,
            'session_prefix': session_prefix
        })
        return self
    
    def add_comparison(
        self, 
        original_document: Union[str, Path],
        perturbed_documents: List[Union[str, Path]]
    ) -> 'WorkflowBuilder':
        """Add a comparison step.
        
        Args:
            original_document: Original document
            perturbed_documents: Perturbed documents
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            'type': 'comparison',
            'original_document': original_document,
            'perturbed_documents': perturbed_documents
        })
        return self
    
    def add_validation(
        self, 
        document: Union[str, Path], 
        runs: int = 3
    ) -> 'WorkflowBuilder':
        """Add a deterministic validation step.
        
        Args:
            document: Document to validate
            runs: Number of validation runs
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            'type': 'validation',
            'document': document,
            'runs': runs
        })
        return self
    
    def execute(
        self, 
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Any]:
        """Execute the workflow.
        
        Args:
            progress_callback: Optional progress callback
            
        Returns:
            List of results from each step
        """
        results = []
        total_steps = len(self.steps)
        
        for i, step in enumerate(self.steps):
            step_progress = i / total_steps
            
            if progress_callback:
                progress_callback(f"Executing step {i+1}/{total_steps}: {step['type']}", step_progress)
            
            try:
                if step['type'] == 'single_fuzz':
                    result = self.api.fuzz_document(
                        document=step['document'],
                        session_id=step['session_id']
                    )
                elif step['type'] == 'batch_fuzz':
                    result = self.api.fuzz_documents_batch(
                        documents=step['documents'],
                        session_prefix=step['session_prefix']
                    )
                elif step['type'] == 'comparison':
                    result = self.api.compare_perturbation_impact(
                        original_document=step['original_document'],
                        perturbed_documents=step['perturbed_documents']
                    )
                elif step['type'] == 'validation':
                    result = self.api.validate_deterministic_behavior(
                        document=step['document'],
                        runs=step['runs']
                    )
                else:
                    raise ValueError(f"Unknown step type: {step['type']}")
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Step {i+1} failed: {e}")
                results.append({'error': str(e), 'step': step})
        
        if progress_callback:
            progress_callback("Workflow completed", 1.0)
        
        return results
    
    def save_workflow(self, workflow_path: Union[str, Path]) -> None:
        """Save workflow definition to file.
        
        Args:
            workflow_path: Path to save workflow
        """
        workflow_data = {
            'steps': self.steps,
            'created_at': datetime.now().isoformat()
        }
        
        workflow_path = Path(workflow_path)
        
        if workflow_path.suffix.lower() == '.json':
            with open(workflow_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False, default=str)
        else:
            with open(workflow_path, 'w', encoding='utf-8') as f:
                yaml.dump(workflow_data, f, default_flow_style=False, allow_unicode=True)
    
    def load_workflow(self, workflow_path: Union[str, Path]) -> 'WorkflowBuilder':
        """Load workflow definition from file.
        
        Args:
            workflow_path: Path to workflow file
            
        Returns:
            Self for method chaining
        """
        workflow_path = Path(workflow_path)
        
        if workflow_path.suffix.lower() == '.json':
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
        else:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = yaml.safe_load(f)
        
        self.steps = workflow_data.get('steps', [])
        return self


# Convenience functions for quick access
def create_fuzzer_api(config: Optional[Union[FuzzerConfig, Dict[str, Any], str]] = None) -> FuzzerAPI:
    """Create a FuzzerAPI instance with optional configuration.
    
    Args:
        config: Optional configuration
        
    Returns:
        FuzzerAPI instance
    """
    return FuzzerAPI(config)


def quick_fuzz(
    document: Union[str, Path], 
    config: Optional[Dict[str, Any]] = None
) -> FeedbackReport:
    """Quick fuzzing of a single document with minimal setup.
    
    Args:
        document: Document content or path
        config: Optional configuration dictionary
        
    Returns:
        FeedbackReport with results
    """
    api = FuzzerAPI(config)
    return api.fuzz_document(document)


def quick_compare(
    original_document: Union[str, Path],
    perturbed_documents: List[Union[str, Path]],
    config: Optional[Dict[str, Any]] = None
) -> ComparisonReport:
    """Quick comparison between original and perturbed documents.
    
    Args:
        original_document: Original document
        perturbed_documents: Perturbed documents
        config: Optional configuration dictionary
        
    Returns:
        ComparisonReport with impact analysis
    """
    api = FuzzerAPI(config)
    return api.compare_perturbation_impact(original_document, perturbed_documents)


def quick_batch_fuzz(
    documents: List[Union[str, Path]], 
    config: Optional[Dict[str, Any]] = None
) -> List[FeedbackReport]:
    """Quick batch fuzzing of multiple documents.
    
    Args:
        documents: List of documents
        config: Optional configuration dictionary
        
    Returns:
        List of FeedbackReport objects
    """
    api = FuzzerAPI(config)
    return api.fuzz_documents_batch(documents)