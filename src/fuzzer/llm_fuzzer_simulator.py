"""Main LLM-assisted fuzzer simulator orchestrator."""

"""LLM fuzzer simulator with session management."""

# pyright: reportAttributeAccessIssue=none

import csv
import hashlib
import time

import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.fuzzer.data_models import (
    FeedbackReport, FuzzerConfig, ComparisonReport,
    PerformanceMetrics, CoverageMetrics, ValidationResult,
    APISpec, ParameterSpec,
)
from src.fuzzer.base_interfaces import BaseTargetSystem

from src.fuzzer.document_processor import DocumentProcessor
from src.fuzzer.llm_test_generator import LLMTestGenerator
from src.fuzzer.target_system_simulator import TargetSystemSimulator
from src.fuzzer.javac_target_system import JavacTargetSystem

from src.fuzzer.test_executor import TestExecutor
from src.fuzzer.feedback_collector import FeedbackCollector
from src.fuzzer.base_interfaces import FuzzerError
from src.fuzzer.random_seed_manager import get_seed_manager, RandomSeedManager
from src.fuzzer.experiment_metadata import ExperimentMetadata, ExperimentMetadataManager
from src.utils.logger import get_logger
from src.scs.data_models import FeedbackData


class SessionState:
    """Tracks state for a fuzzing session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.current_phase = "initialized"
        self.progress = 0.0
        self.errors = []
        self.warnings = []
        self.metadata = {}
    
    def update_phase(self, phase: str, progress: Optional[float] = None):
        """Update current phase and progress."""
        self.current_phase = phase
        if progress is not None:
            self.progress = progress

    
    def add_error(self, error: str):
        """Add an error to the session state."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning to the session state."""
        self.warnings.append(warning)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since session start."""
        return time.time() - self.start_time


class LLMFuzzerSimulator:
    """Main orchestrator for LLM-assisted fuzzer simulation with enhanced session management.
    
    The LLMFuzzerSimulator is the primary interface for running realistic LLM-assisted
    fuzzing simulations. It coordinates all components of the fuzzing pipeline including
    document processing, test case generation, execution, and feedback collection.
    
    Key Features:
        - Realistic LLM-based test case generation
        - Comprehensive vulnerability simulation
        - Parallel test execution with resource management
        - Detailed performance and coverage metrics
        - Deterministic execution with random seed management
        - Integration with existing SCS (Semantic Contribution Score) system
        - Batch processing for multiple documents
        - Perturbation impact analysis
        
    Architecture:
        The simulator follows a pipeline architecture:
        1. DocumentProcessor: Extracts API specifications from documentation
        2. LLMTestGenerator: Generates diverse test cases using LLM
        3. TestExecutor: Executes test cases with parallel processing
        4. TargetSystemSimulator: Simulates realistic API behavior with vulnerabilities
        5. FeedbackCollector: Aggregates results and generates comprehensive reports
        
    Example:
        Basic usage:
        >>> config = FuzzerConfig(llm_model="gpt-4", cases_per_api=20, random_seed=42)
        >>> fuzzer = LLMFuzzerSimulator(config)
        >>> report = fuzzer.run_fuzzing_session("def authenticate(user, pass): return True")
        >>> print(f"Validity: {report.get_validity_rate():.2%}")
        
        Batch processing:
        >>> documents = ["def func1():", "def func2():"]
        >>> reports = fuzzer.run_batch_fuzzing(documents, parallel=True)
        
        Perturbation analysis:
        >>> original = "def hash_data(data): return hash(data)"
        >>> perturbed = "def hash​_data(data): return hash(data)"  # zero-width space
        >>> comparison = fuzzer.compare_fuzzing_results(original, [perturbed])
        >>> print(f"Impact: {comparison.overall_impact:.2%}")
        
    Attributes:
        config (FuzzerConfig): Configuration settings for the fuzzer
        seed_manager (RandomSeedManager): Manages random seeds for deterministic execution
        document_processor (DocumentProcessor): Processes documentation to extract APIs
        test_generator (LLMTestGenerator): Generates test cases using LLM
        target_system (BaseTargetSystem): Simulates or executes target behavior

        test_executor (TestExecutor): Executes test cases with parallel processing
        feedback_collector (FeedbackCollector): Collects and analyzes results
        active_sessions (Dict[str, SessionState]): Currently active fuzzing sessions
        session_history (List[str]): History of completed session IDs
        total_sessions (int): Total number of sessions run
        total_execution_time (float): Total execution time across all sessions
        
    Thread Safety:
        The simulator is thread-safe for read operations but not for concurrent
        fuzzing sessions. Use separate instances for concurrent execution.
        
    Performance:
        - Memory usage: 50-200MB per session depending on document size
        - Execution time: 5-30 seconds per session depending on complexity
        - LLM API calls: 1-5 calls per session
        - Supports parallel batch processing for improved throughput
        
    Integration:
        - SCS System: Compatible with existing Semantic Contribution Score calculation
        - CLI: Integrated with anti_llm4fuzz CLI via --use-llm-fuzzer flag
        - Perturbation Pipeline: Seamless integration with document perturbation system
        
    See Also:
        - FuzzerConfig: Configuration options and validation
        - FeedbackReport: Comprehensive results from fuzzing sessions
        - ComparisonReport: Analysis of perturbation impacts
        - DocumentProcessor: API extraction from documentation
        - LLMTestGenerator: LLM-based test case generation
    """
    
    def __init__(self, config: Optional[FuzzerConfig] = None, seed_manager: Optional[RandomSeedManager] = None):
        """Initialize fuzzer simulator.
        
        Args:
            config: Fuzzer configuration, uses defaults if None
            seed_manager: Random seed manager for deterministic behavior
        """
        self.config = config or FuzzerConfig()
        self.logger = get_logger("LLMFuzzerSimulator")
        
        # Initialize seed manager for deterministic behavior
        self.seed_manager = seed_manager or get_seed_manager(self.config.random_seed)
        
        # Initialize experiment metadata manager
        self.metadata_manager = ExperimentMetadataManager()
        
        # Validate configuration
        config_validation = self.config.validate()
        if not config_validation.is_valid:
            self.logger.error(f"Invalid configuration: {config_validation.errors}")
            raise FuzzerError(f"Configuration validation failed: {config_validation.errors}")
        
        if config_validation.warnings:
            self.logger.warning(f"Configuration warnings: {config_validation.warnings}")
        
        # Initialize components with error handling and deterministic behavior
        try:
            self.document_processor = DocumentProcessor()
            self.test_generator = LLMTestGenerator(self.config, self.seed_manager)
            if self.config.target_mode == "javac":
                javac_home = self.config.javac_home or ""
                source_root = self.config.javac_source_root or ""
                jacoco_cli_path = self.config.jacoco_cli_path or ""
                jacoco_agent_path = self.config.jacoco_agent_path or ""
                self.target_system = JavacTargetSystem(
                    javac_home=javac_home,
                    source_root=source_root,
                    jacoco_cli_path=jacoco_cli_path,
                    jacoco_agent_path=jacoco_agent_path,
                    coverage_output_dir=self.config.coverage_output_dir,
                    coverage_scope=self.config.coverage_scope,
                    timeout=float(self.config.timeout_per_test),
                )
            else:
                self.target_system = TargetSystemSimulator(self.config, self.seed_manager)
            self.test_executor = TestExecutor(self.config)
            self.feedback_collector = FeedbackCollector(self.config)

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise FuzzerError(f"Component initialization failed: {e}")
        
        # Session management
        self.active_sessions: Dict[str, SessionState] = {}
        self.session_history: List[str] = []
        
        # Performance tracking
        self.total_sessions = 0
        self.total_execution_time = 0.0
        
        self.logger.info(f"LLM Fuzzer Simulator initialized successfully with seed: {self.seed_manager.get_master_seed()}")
    
    def run_fuzzing_session(
        self, 
        document_content: str, 
        session_id: Optional[str] = None,
        document_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        save_metadata: bool = True
    ) -> FeedbackReport:
        """Run a complete fuzzing session on a document with enhanced state tracking.
        
        Args:
            document_content: Content of the document to fuzz
            session_id: Optional session identifier
            document_path: Optional path to document file for metadata
            progress_callback: Optional callback for progress updates
            save_metadata: Whether to save experiment metadata
            
        Returns:
            FeedbackReport with comprehensive results
        """
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Create session state
        session_state = SessionState(session_id)
        self.active_sessions[session_id] = session_state
        
        # Create experiment metadata if document path is provided
        experiment_metadata = None
        if save_metadata and document_path:
            try:
                experiment_metadata = ExperimentMetadata.create_for_session(
                    session_id=session_id,
                    document_path=document_path,
                    config=self.config,
                    seed_manager=self.seed_manager
                )
                experiment_metadata.start_execution()
            except Exception as e:
                self.logger.warning(f"Failed to create experiment metadata: {e}")
        
        self.logger.info(f"Starting fuzzing session: {session_id}")
        
        try:
            # Validate input
            if not document_content or not document_content.strip():
                raise FuzzerError("Document content cannot be empty")
            
            session_state.metadata['document_length'] = len(document_content)
            
            # Step 1: Process document
            session_state.update_phase("processing_document", 0.1)
            if progress_callback:
                progress_callback("Processing document", 0.1)
            
            self.logger.info("Step 1: Processing document")
            document_structure = self.document_processor.parse_document(document_content)
            
            # Validate document structure
            doc_validation = document_structure.validate()
            if doc_validation.warnings:
                for warning in doc_validation.warnings:
                    session_state.add_warning(warning)
            
            if not document_structure.api_specs and not self.config.document_generation_enabled:
                session_state.add_warning("No API specifications found in document; using fallback spec")
                document_structure.api_specs = [
                    self._build_fallback_api_spec(document_content)
                ]
            
            session_state.metadata['api_count'] = len(document_structure.api_specs)
            
            # Step 2: Generate test cases
            session_state.update_phase("generating_tests", 0.3)
            if progress_callback:
                progress_callback("Generating test cases", 0.3)
            
            self.logger.info("Step 2: Generating test cases")
            if self.config.document_generation_enabled:
                test_cases = self.test_generator.generate_document_test_cases(
                    document_content,
                    self.config.cases_per_document
                )
            else:
                total_test_count = self.config.cases_per_api * len(document_structure.api_specs)
                test_cases = self.test_generator.generate_test_cases(
                    document_structure.api_specs,
                    total_test_count
                )
            
            if not test_cases:
                session_state.add_error("No test cases generated")
                return self._create_empty_report(session_id, document_content, session_state)
            
            session_state.metadata['test_count'] = len(test_cases)
            
            # Validate test cases
            invalid_tests = []
            for test_case in test_cases:
                validation = test_case.validate()
                if not validation.is_valid:
                    invalid_tests.append(test_case.id)
            
            if invalid_tests:
                session_state.add_warning(f"Invalid test cases: {len(invalid_tests)}")
            
            # Step 3: Execute test cases
            session_state.update_phase("executing_tests", 0.6)
            if progress_callback:
                progress_callback("Executing test cases", 0.6)
            
            self.logger.info("Step 3: Executing test cases")
            execution_results = self.test_executor.run_test_suite(
                test_cases,
                self.target_system
            )
            if self.config.target_mode == "javac":
                target_system = self.target_system
                report_dir = None
                if isinstance(target_system, JavacTargetSystem):
                    report_dir = target_system.generate_report(
                        output_dir=Path(self.config.coverage_output_dir) / "report" / session_id
                    )
                else:
                    report_dir = None
                if report_dir:
                    session_state.metadata["coverage_report_dir"] = str(report_dir)

            
            session_state.metadata['execution_count'] = len(execution_results)
            
            # Step 4: Collect feedback
            session_state.update_phase("collecting_feedback", 0.9)
            if progress_callback:
                progress_callback("Collecting feedback", 0.9)
            
            self.logger.info("Step 4: Collecting feedback")
            feedback_report = self.feedback_collector.generate_report(
                session_id=session_id,
                document_content=document_content,
                api_specs=document_structure.api_specs,
                test_cases=test_cases,
                execution_results=execution_results
            )

            if self.config.target_mode == "javac":
                report_dir_value = session_state.metadata.get("coverage_report_dir")
                if report_dir_value:
                    coverage_metrics = self._load_jacoco_coverage(Path(report_dir_value))
                    if coverage_metrics:
                        feedback_report.coverage_metrics = coverage_metrics
                        feedback_report.metadata = feedback_report.metadata or {}
                        feedback_report.metadata["jacoco_line_coverage"] = coverage_metrics.line_coverage
                        feedback_report.metadata["jacoco_branch_coverage"] = coverage_metrics.branch_coverage
                        feedback_report.metadata["jacoco_method_coverage"] = coverage_metrics.function_coverage
            
            # Add enhanced performance metrics
            total_time = session_state.get_elapsed_time()
            feedback_report.performance_metrics = PerformanceMetrics(
                total_execution_time=total_time,
                average_test_time=total_time / len(test_cases) if test_cases else 0,
                llm_generation_time=getattr(self.test_generator, 'last_generation_time', 0),
                test_execution_time=getattr(self.test_executor, 'last_execution_time', 0),
                memory_usage_mb=self._get_memory_usage(),
                tests_per_second=len(test_cases) / total_time if total_time > 0 else 0
            )
            
            # Add session metadata to report
            metadata = feedback_report.metadata or {}
            metadata.update(session_state.metadata)
            metadata['session_errors'] = session_state.errors
            metadata['session_warnings'] = session_state.warnings
            feedback_report.metadata = metadata

            
            # Complete session
            session_state.update_phase("completed", 1.0)
            if progress_callback:
                progress_callback("Completed", 1.0)

            # Update experiment metadata
            if experiment_metadata:
                try:
                    experiment_metadata.end_execution(
                        total_test_cases=len(test_cases),
                        validity_rate=feedback_report.get_validity_rate(),
                        coverage_percentage=feedback_report.get_coverage_percentage(),
                        defects_found=feedback_report.get_defect_count()
                    )
                    # Save metadata
                    metadata_path = self.metadata_manager.save_metadata(experiment_metadata)
                    metadata = feedback_report.metadata or {}
                    metadata['experiment_metadata_path'] = metadata_path
                    feedback_report.metadata = metadata
                except Exception as e:
                    self.logger.warning(f"Failed to save experiment metadata: {e}")

            # Update global statistics
            self.total_sessions += 1
            self.total_execution_time += total_time

            self.logger.info(f"Fuzzing session completed in {total_time:.2f}s")
            self.logger.info(
                f"Results: {feedback_report.get_validity_rate():.2%} validity, "
                f"{feedback_report.get_coverage_percentage():.1f}% coverage, "
                f"{feedback_report.get_defect_count()} defects"
            )

            return feedback_report

        except Exception as e:
            session_state.add_error(str(e))
            session_state.update_phase("failed", 0.0)
            self.logger.error(f"Fuzzing session failed: {e}")
            return self._create_error_report(session_id, document_content, str(e), session_state)

        
        finally:
            # Move session to history
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            self.session_history.append(session_id)
    
    def run_batch_fuzzing(
        self, 
        documents: List[str], 
        session_prefix: str = "batch",
        parallel: Optional[bool] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None

    ) -> List[FeedbackReport]:
        """Run fuzzing on multiple documents with parallel processing support.
        
        Args:
            documents: List of document contents
            session_prefix: Prefix for session IDs
            parallel: Whether to run in parallel (uses config default if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of FeedbackReport objects
        """
        if not documents:
            self.logger.warning("No documents provided for batch fuzzing")
            return []
        
        self.logger.info(f"Starting batch fuzzing for {len(documents)} documents")
        
        # Use parallel execution based on config or parameter
        use_parallel = parallel if parallel is not None else self.config.parallel_execution
        
        if use_parallel and len(documents) > 1:
            return self._run_batch_parallel(documents, session_prefix, progress_callback)
        else:
            return self._run_batch_sequential(documents, session_prefix, progress_callback)
    
    def _run_batch_sequential(
        self, 
        documents: List[str], 
        session_prefix: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[FeedbackReport]:
        """Run batch fuzzing sequentially."""
        reports = []
        
        for i, document in enumerate(documents):
            session_id = f"{session_prefix}_{i+1:03d}"
            
            # Update progress
            progress = (i + 1) / len(documents)
            if progress_callback:
                progress_callback(f"Processing document {i+1}/{len(documents)}", progress)
            
            try:
                report = self.run_fuzzing_session(document, session_id)
                reports.append(report)
                
            except Exception as e:
                self.logger.error(f"Batch fuzzing failed for document {i+1}: {e}")
                error_report = self._create_error_report(session_id, document, str(e))
                reports.append(error_report)
        
        self.logger.info(f"Sequential batch fuzzing completed: {len(reports)} reports generated")
        return reports
    
    def _run_batch_parallel(
        self, 
        documents: List[str], 
        session_prefix: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[FeedbackReport]:
        """Run batch fuzzing in parallel."""
        self.logger.info(f"Running batch fuzzing in parallel with {self.config.max_workers} workers")
        
        reports: List[FeedbackReport] = [
            self._create_empty_report(f"{session_prefix}_init", "") for _ in documents
        ]
        self.logger.debug(f"Initialized {len(reports)} placeholder reports")

        completed = 0

        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, document in enumerate(documents):
                session_id = f"{session_prefix}_{i+1:03d}"
                future = executor.submit(self.run_fuzzing_session, document, session_id)
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1
                
                # Update progress
                progress = completed / len(documents)
                if progress_callback:
                    progress_callback(f"Completed {completed}/{len(documents)} documents", progress)
                
                try:
                    report = future.result()
                    reports[index] = report
                    
                except Exception as e:
                    self.logger.error(f"Parallel batch fuzzing failed for document {index+1}: {e}")
                    session_id = f"{session_prefix}_{index+1:03d}"
                    error_report = self._create_error_report(session_id, documents[index], str(e))
                    reports[index] = error_report
        
        self.logger.info(f"Parallel batch fuzzing completed: {len(reports)} reports generated")
        return reports
    
    def compare_fuzzing_results(
        self, 
        original_document: str, 
        perturbed_documents: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ComparisonReport:
        """Compare fuzzing results between original and perturbed documents.
        
        Args:
            original_document: Original document content
            perturbed_documents: List of perturbed document contents
            progress_callback: Optional callback for progress updates
            
        Returns:
            ComparisonReport with impact analysis
        """
        self.logger.info("Comparing fuzzing results for perturbation impact analysis")
        
        total_documents = 1 + len(perturbed_documents)
        
        # Run fuzzing on original document
        if progress_callback:
            progress_callback("Processing original document", 1 / total_documents)
        
        original_report = self.run_fuzzing_session(original_document, "original")
        
        # Run fuzzing on perturbed documents
        perturbed_reports = []
        for i, perturbed_doc in enumerate(perturbed_documents):
            session_id = f"perturbed_{i+1:03d}"
            
            if progress_callback:
                progress = (i + 2) / total_documents
                progress_callback(f"Processing perturbed document {i+1}/{len(perturbed_documents)}", progress)
            
            report = self.run_fuzzing_session(perturbed_doc, session_id)
            perturbed_reports.append(report)
        
        # Create comparison report
        comparison = ComparisonReport(
            original_report=original_report,
            perturbed_reports=perturbed_reports
        )
        
        # Calculate impact metrics
        comparison.calculate_impacts()
        
        # Add statistical analysis
        self._add_statistical_analysis(comparison)
        
        self.logger.info(f"Perturbation impact analysis completed:")
        self.logger.info(f"  Validity impact: {comparison.validity_impact:.2%}")
        self.logger.info(f"  Coverage impact: {comparison.coverage_impact:.2%}")
        self.logger.info(f"  Defect impact: {comparison.defect_impact:.2%}")
        self.logger.info(f"  Overall impact: {comparison.overall_impact:.2%}")
        self.logger.info(f"  Statistical significance: {comparison.statistical_significance}")
        
        return comparison
    
    def analyze_perturbation_impact(
        self, 
        original_document: str, 
        perturbed_document: str,
        perturbation_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze the impact of a single perturbation on fuzzing performance.
        
        Args:
            original_document: Original document content
            perturbed_document: Perturbed document content
            perturbation_metadata: Optional metadata about the perturbation
            
        Returns:
            Dictionary with detailed impact analysis
        """
        self.logger.info("Analyzing single perturbation impact")
        
        # Run fuzzing on both documents
        original_report = self.run_fuzzing_session(original_document, "original_single")
        perturbed_report = self.run_fuzzing_session(perturbed_document, "perturbed_single")
        
        # Calculate detailed metrics
        impact_analysis = {
            'validity_change': perturbed_report.get_validity_rate() - original_report.get_validity_rate(),
            'coverage_change': perturbed_report.get_coverage_percentage() - original_report.get_coverage_percentage(),
            'defect_change': perturbed_report.get_defect_count() - original_report.get_defect_count(),
            'performance_change': (
                perturbed_report.performance_metrics.total_execution_time - 
                original_report.performance_metrics.total_execution_time
            ),
            'original_metrics': {
                'validity_rate': original_report.get_validity_rate(),
                'coverage_percentage': original_report.get_coverage_percentage(),
                'defect_count': original_report.get_defect_count(),
                'execution_time': original_report.performance_metrics.total_execution_time
            },
            'perturbed_metrics': {
                'validity_rate': perturbed_report.get_validity_rate(),
                'coverage_percentage': perturbed_report.get_coverage_percentage(),
                'defect_count': perturbed_report.get_defect_count(),
                'execution_time': perturbed_report.performance_metrics.total_execution_time
            }
        }
        
        # Calculate relative changes
        if original_report.get_validity_rate() > 0:
            impact_analysis['validity_change_percent'] = (
                impact_analysis['validity_change'] / original_report.get_validity_rate() * 100
            )
        
        if original_report.get_coverage_percentage() > 0:
            impact_analysis['coverage_change_percent'] = (
                impact_analysis['coverage_change'] / original_report.get_coverage_percentage() * 100
            )
        
        # Add perturbation metadata
        if perturbation_metadata:
            impact_analysis['perturbation_metadata'] = perturbation_metadata
        
        # Calculate overall impact score
        impact_analysis['overall_impact_score'] = self._calculate_impact_score(impact_analysis)
        
        return impact_analysis
    
    def integrate_with_feedback_system(
        self, 
        document_content: str, 
        token_text: str,
        token_metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackData:
        """Integrate with existing feedback analysis system for SCS calculation.
        
        This method provides compatibility with the existing SCS system by
        converting fuzzer simulation results to FeedbackData format.
        
        Args:
            document_content: Document content with token perturbation
            token_text: The perturbed token text
            token_metadata: Optional metadata about the token
            
        Returns:
            FeedbackData compatible with SCS system
        """
        self.logger.debug(f"Integrating with feedback system for token: {token_text}")
        
        try:
            # Run lightweight fuzzing session
            session_id = f"feedback_{hash(token_text) & 0x7FFFFFFF:08x}"
            report = self.run_fuzzing_session(document_content, session_id)
            
            # Convert to FeedbackData format
            feedback_data = FeedbackData.create_now(
                validity_rate=report.get_validity_rate(),
                coverage_percent=report.get_coverage_percentage(),
                defects_found=report.get_defect_count()
            )
            
            self.logger.debug(f"Generated feedback for token {token_text}: "
                            f"validity={feedback_data.validity_rate:.3f}, "
                            f"coverage={feedback_data.coverage_percent:.1f}%, "
                            f"defects={feedback_data.defects_found}")
            
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate feedback for token {token_text}: {e}")
            
            # Return degraded feedback as fallback
            return FeedbackData.create_now(
                validity_rate=0.5,  # Assume moderate impact
                coverage_percent=50.0,
                defects_found=5
            )
    
    def batch_perturbation_analysis(
        self, 
        original_document: str,
        perturbation_configs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple perturbations in batch for efficiency.
        
        Args:
            original_document: Original document content
            perturbation_configs: List of perturbation configurations
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of impact analysis results
        """
        self.logger.info(f"Running batch perturbation analysis for {len(perturbation_configs)} perturbations")
        
        # Run original document once
        original_report = self.run_fuzzing_session(original_document, "batch_original")
        
        results = []
        
        for i, config in enumerate(perturbation_configs):
            if progress_callback:
                progress = (i + 1) / len(perturbation_configs)
                progress_callback(f"Analyzing perturbation {i+1}/{len(perturbation_configs)}", progress)
            
            try:
                perturbed_document = config.get('perturbed_document', '')
                perturbation_metadata = config.get('metadata', {})
                
                # Run fuzzing on perturbed document
                session_id = f"batch_perturbed_{i+1:03d}"
                perturbed_report = self.run_fuzzing_session(perturbed_document, session_id)
                
                # Calculate impact
                impact_result = self._calculate_batch_impact(
                    original_report, 
                    perturbed_report, 
                    perturbation_metadata
                )
                
                results.append(impact_result)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze perturbation {i+1}: {e}")
                results.append({
                    'error': str(e),
                    'perturbation_index': i,
                    'metadata': config.get('metadata', {})
                })
        
        self.logger.info(f"Batch perturbation analysis completed: {len(results)} results")
        return results
    
    def run_batch_with_resource_management(
        self,
        documents: List[str],
        session_prefix: str = "managed_batch",
        max_concurrent_sessions: Optional[int] = None,
        memory_limit_mb: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[FeedbackReport]:
        """Run batch fuzzing with advanced resource management.
        
        Args:
            documents: List of document contents
            session_prefix: Prefix for session IDs
            max_concurrent_sessions: Maximum concurrent sessions (uses config if None)
            memory_limit_mb: Memory limit in MB (no limit if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of FeedbackReport objects
        """
        if not documents:
            return []
        
        max_concurrent = max_concurrent_sessions or self.config.max_workers
        self.logger.info(f"Running managed batch fuzzing: {len(documents)} documents, "
                        f"max concurrent: {max_concurrent}")
        
        reports = []
        active_futures = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit initial batch
            for i in range(min(max_concurrent, len(documents))):
                session_id = f"{session_prefix}_{i+1:03d}"
                future = executor.submit(self.run_fuzzing_session, documents[i], session_id)
                active_futures[future] = i
            
            next_doc_index = max_concurrent
            
            # Process completions and submit new tasks
            while active_futures:
                # Check memory usage if limit is set
                if memory_limit_mb and self._get_memory_usage() > memory_limit_mb:
                    self.logger.warning(f"Memory usage ({self._get_memory_usage():.1f}MB) "
                                      f"exceeds limit ({memory_limit_mb}MB), waiting...")
                    time.sleep(1)
                    continue
                
                # Wait for at least one task to complete
                completed_futures = []
                for future in list(active_futures.keys()):
                    if future.done():
                        completed_futures.append(future)
                
                if not completed_futures:
                    time.sleep(0.1)  # Brief wait
                    continue
                
                # Process completed tasks
                for future in completed_futures:
                    doc_index = active_futures.pop(future)
                    completed += 1
                    
                    # Update progress
                    progress = completed / len(documents)
                    if progress_callback:
                        progress_callback(f"Completed {completed}/{len(documents)} documents", progress)
                    
                    try:
                        report = future.result()
                        # Ensure reports are in correct order
                        while len(reports) <= doc_index:
                            reports.append(None)
                        reports[doc_index] = report
                        
                    except Exception as e:
                        self.logger.error(f"Batch processing failed for document {doc_index+1}: {e}")
                        session_id = f"{session_prefix}_{doc_index+1:03d}"
                        error_report = self._create_error_report(session_id, documents[doc_index], str(e))
                        while len(reports) <= doc_index:
                            reports.append(None)
                        reports[doc_index] = error_report
                
                # Submit new tasks if available
                while (next_doc_index < len(documents) and 
                       len(active_futures) < max_concurrent):
                    session_id = f"{session_prefix}_{next_doc_index+1:03d}"
                    future = executor.submit(
                        self.run_fuzzing_session, 
                        documents[next_doc_index], 
                        session_id
                    )
                    active_futures[future] = next_doc_index
                    next_doc_index += 1
        
        # Filter out None values (shouldn't happen but safety check)
        reports = [r for r in reports if r is not None]
        
        self.logger.info(f"Managed batch fuzzing completed: {len(reports)} reports generated")
        return reports
    
    def run_batch_with_scheduling(
        self,
        document_batches: List[List[str]],
        batch_names: Optional[List[str]] = None,
        inter_batch_delay: float = 0.0,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, List[FeedbackReport]]:
        """Run multiple batches with scheduling and resource allocation.
        
        Args:
            document_batches: List of document batches
            batch_names: Optional names for batches
            inter_batch_delay: Delay between batches in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping batch names to report lists
        """
        if not document_batches:
            return {}
        
        if batch_names is None:
            batch_names = [f"batch_{i+1}" for i in range(len(document_batches))]
        
        if len(batch_names) != len(document_batches):
            raise ValueError("Number of batch names must match number of document batches")
        
        self.logger.info(f"Running scheduled batch processing: {len(document_batches)} batches")
        
        results = {}
        total_documents = sum(len(batch) for batch in document_batches)
        processed_documents = 0
        
        for batch_idx, (batch_name, documents) in enumerate(zip(batch_names, document_batches)):
            self.logger.info(f"Processing batch '{batch_name}' ({len(documents)} documents)")
            
            # Update progress
            if progress_callback:
                batch_progress = batch_idx / len(document_batches)
                progress_callback(f"Starting batch {batch_name}", batch_progress)
            
            # Run batch
            batch_reports = self.run_batch_fuzzing(
                documents, 
                session_prefix=batch_name,
                progress_callback=lambda msg, prog: progress_callback(
                    f"Batch {batch_name}: {msg}", 
                    (processed_documents + prog * len(documents)) / total_documents
                ) if progress_callback else None
            )
            
            results[batch_name] = batch_reports
            processed_documents += len(documents)
            
            # Inter-batch delay
            if inter_batch_delay > 0 and batch_idx < len(document_batches) - 1:
                self.logger.info(f"Waiting {inter_batch_delay}s before next batch")
                time.sleep(inter_batch_delay)
        
        self.logger.info(f"Scheduled batch processing completed: {len(results)} batches")
        return results
    
    def generate_aggregated_report(
        self, 
        reports: List[FeedbackReport],
        report_name: str = "aggregated_report"
    ) -> Dict[str, Any]:
        """Generate aggregated report from multiple fuzzing sessions.
        
        Args:
            reports: List of feedback reports to aggregate
            report_name: Name for the aggregated report
            
        Returns:
            Dictionary with aggregated metrics and analysis
        """
        if not reports:
            return {'error': 'No reports provided'}
        
        self.logger.info(f"Generating aggregated report from {len(reports)} sessions")
        
        # Calculate aggregate metrics
        total_test_cases = sum(r.total_test_cases for r in reports)
        avg_validity_rate = sum(r.get_validity_rate() for r in reports) / len(reports)
        avg_coverage = sum(r.get_coverage_percentage() for r in reports) / len(reports)
        total_defects = sum(r.get_defect_count() for r in reports)
        total_execution_time = sum(r.performance_metrics.total_execution_time for r in reports)
        
        # Calculate distributions
        validity_rates = [r.get_validity_rate() for r in reports]
        coverage_rates = [r.get_coverage_percentage() for r in reports]
        defect_counts = [float(r.get_defect_count()) for r in reports]

        
        aggregated_report = {
            'report_name': report_name,
            'timestamp': datetime.now().isoformat(),
            'session_count': len(reports),
            
            # Aggregate metrics
            'total_test_cases': total_test_cases,
            'average_validity_rate': avg_validity_rate,
            'average_coverage_percentage': avg_coverage,
            'total_defects_found': total_defects,
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / len(reports),
            
            # Distributions
            'validity_rate_distribution': {
                'min': min(validity_rates),
                'max': max(validity_rates),
                'mean': avg_validity_rate,
                'std': self._calculate_std_dev(validity_rates)
            },
            'coverage_distribution': {
                'min': min(coverage_rates),
                'max': max(coverage_rates),
                'mean': avg_coverage,
                'std': self._calculate_std_dev(coverage_rates)
            },
            'defect_distribution': {
                'min': min(defect_counts),
                'max': max(defect_counts),
                'mean': total_defects / len(reports),
                'std': self._calculate_std_dev(defect_counts)


            },
            
            # Success metrics
            'successful_sessions': len([r for r in reports if r.is_successful_session()]),
            'failed_sessions': len([r for r in reports if not r.is_successful_session()]),
            'success_rate': len([r for r in reports if r.is_successful_session()]) / len(reports),
            
            # Performance metrics
            'throughput_tests_per_second': total_test_cases / total_execution_time if total_execution_time > 0 else 0,
            'average_memory_usage_mb': sum(
                r.performance_metrics.memory_usage_mb for r in reports 
                if r.performance_metrics.memory_usage_mb > 0
            ) / max(1, len([r for r in reports if r.performance_metrics.memory_usage_mb > 0])),
            
            # Session details
            'session_ids': [r.session_id for r in reports],
            'document_hashes': list(set(r.document_hash for r in reports))
        }
        
        return aggregated_report
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def simulate_feedback_for_token(self, document_content: str, token_text: str) -> FeedbackData:
        """Simulate fuzzer feedback for a specific token perturbation.
        
        This method provides compatibility with the existing SCS system.
        
        Args:
            document_content: Document content with token perturbation
            token_text: The perturbed token text
            
        Returns:
            FeedbackData compatible with SCS system
        """
        self.logger.debug(f"Simulating feedback for token: {token_text}")
        
        try:
            # Run a lightweight fuzzing session
            report = self.run_fuzzing_session(document_content, f"token_{hash(token_text)}")
            
            # Convert to FeedbackData format
            feedback_data = FeedbackData.create_now(
                validity_rate=report.get_validity_rate(),
                coverage_percent=report.get_coverage_percentage(),
                defects_found=report.get_defect_count()
            )
            
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"Failed to simulate feedback for token {token_text}: {e}")
            
            # Return degraded feedback as fallback
            return FeedbackData.create_now(
                validity_rate=0.5,  # Assume moderate impact
                coverage_percent=50.0,
                defects_found=5
            )
    
    def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get state for an active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionState if session is active, None otherwise
        """
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.active_sessions.keys())
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel an active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cancelled, False if not found
        """
        if session_id in self.active_sessions:
            session_state = self.active_sessions[session_id]
            session_state.update_phase("cancelled", 0.0)
            session_state.add_error("Session cancelled by user")
            del self.active_sessions[session_id]
            self.session_history.append(session_id)
            self.logger.info(f"Session {session_id} cancelled")
            return True
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_sessions': self.total_sessions,
            'total_execution_time': self.total_execution_time,
            'average_session_time': self.total_execution_time / max(1, self.total_sessions),
            'active_sessions': len(self.active_sessions),
            'target_system_calls': getattr(self.target_system, 'get_call_count', lambda: 0)(),

            'test_generation_time': getattr(self.test_generator, 'last_generation_time', 0),
            'test_execution_time': getattr(self.test_executor, 'last_execution_time', 0),
            'memory_usage_mb': self._get_memory_usage(),
            'session_history_count': len(self.session_history)
        }
    
    def reset_state(self) -> None:
        """Reset simulator state for fresh session."""
        self.logger.debug("Resetting simulator state")
        
        # Cancel all active sessions
        for session_id in list(self.active_sessions.keys()):
            self.cancel_session(session_id)
        
        # Reset target system state
        self.target_system.reset_state()
        
        # Clear any cached data
        if hasattr(self.test_generator, 'clear_cache'):
            self.test_generator.clear_cache()
        
        # Reset performance counters
        self.total_sessions = 0
        self.total_execution_time = 0.0
        self.session_history.clear()
        
        self.logger.info("Simulator state reset completed")
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'LLMFuzzerSimulator':
        """Create simulator from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            LLMFuzzerSimulator instance
        """
        config = FuzzerConfig.from_dict(config_dict)
        return cls(config)
    
    def _create_empty_report(
        self, 
        session_id: str, 
        document_content: str, 
        session_state: Optional[SessionState] = None
    ) -> FeedbackReport:
        """Create an empty feedback report with session information."""
        report = FeedbackReport(
            session_id=session_id,
            document_hash=self._hash_content(document_content),
            total_test_cases=0,
            validity_rate=0.0,
            coverage_metrics=CoverageMetrics(),
            defects_found=[],
            performance_metrics=PerformanceMetrics()
        )
        
        # Add session metadata if available
        if session_state:
            report.metadata = {
                'session_errors': session_state.errors,
                'session_warnings': session_state.warnings,
                'session_metadata': session_state.metadata,
                'session_phase': session_state.current_phase,
            }
        else:
            report.metadata = report.metadata or {}

        return report

    
    def _create_error_report(
        self, 
        session_id: str, 
        document_content: str, 
        error_message: str,
        session_state: Optional[SessionState] = None
    ) -> FeedbackReport:
        """Create an error feedback report with detailed error information."""
        report = self._create_empty_report(session_id, document_content, session_state)
        
        # Add error information to metadata
        metadata = report.metadata or {}
        metadata['error'] = error_message
        metadata['error_timestamp'] = datetime.now().isoformat()
        if session_state:
            metadata['failed_phase'] = session_state.current_phase
        report.metadata = metadata

        
        return report
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for document content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _build_fallback_api_spec(self, document_content: str) -> APISpec:
        title = document_content.strip().splitlines()[0] if document_content.strip() else "Document"
        description = (document_content.strip()[:200] + "...") if len(document_content.strip()) > 200 else document_content.strip()
        safe_title = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")
        safe_title = safe_title[:40] if safe_title else "Document"
        parameters = [
            ParameterSpec(
                name="payload",
                type="string",
                description="Document-derived payload for fuzzing",
                required=True,
            )
        ]
        return APISpec(
            name=f"{safe_title}_payload",
            description=description or "Fallback API specification",
            parameters=parameters,
            method="POST",
        )

    def _load_jacoco_coverage(self, report_dir: Path) -> Optional[CoverageMetrics]:
        csv_path = report_dir / "coverage.csv"
        if not csv_path.exists():
            self.logger.warning(f"JaCoCo coverage CSV not found: {csv_path}")
            return None

        totals = {
            "line_covered": 0,
            "line_missed": 0,
            "branch_covered": 0,
            "branch_missed": 0,
            "method_covered": 0,
            "method_missed": 0,
        }

        try:
            with csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    totals["line_covered"] += int(row.get("LINE_COVERED", 0))
                    totals["line_missed"] += int(row.get("LINE_MISSED", 0))
                    totals["branch_covered"] += int(row.get("BRANCH_COVERED", 0))
                    totals["branch_missed"] += int(row.get("BRANCH_MISSED", 0))
                    totals["method_covered"] += int(row.get("METHOD_COVERED", 0))
                    totals["method_missed"] += int(row.get("METHOD_MISSED", 0))
        except Exception as exc:
            self.logger.warning(f"Failed to parse JaCoCo CSV: {exc}")
            return None

        line_total = totals["line_covered"] + totals["line_missed"]
        branch_total = totals["branch_covered"] + totals["branch_missed"]
        method_total = totals["method_covered"] + totals["method_missed"]

        line_coverage = (totals["line_covered"] / line_total * 100.0) if line_total else 0.0
        branch_coverage = (totals["branch_covered"] / branch_total * 100.0) if branch_total else 0.0
        method_coverage = (totals["method_covered"] / method_total * 100.0) if method_total else 0.0

        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=method_coverage,
            api_endpoint_coverage=0.0,
            total_lines=line_total,
            total_branches=branch_total,
        )
    
    def _add_statistical_analysis(self, comparison: ComparisonReport) -> None:
        """Add statistical analysis to comparison report."""
        try:
            # Simple statistical significance test
            # Check if the impact is consistent across perturbed documents
            if len(comparison.perturbed_reports) >= 3:
                validity_rates = [r.get_validity_rate() for r in comparison.perturbed_reports]
                coverage_rates = [r.get_coverage_percentage() for r in comparison.perturbed_reports]
                
                # Calculate variance
                validity_variance = self._calculate_variance(validity_rates)
                coverage_variance = self._calculate_variance(coverage_rates)
                
                # Consider significant if variance is low and impact is substantial
                comparison.statistical_significance = (
                    validity_variance < 0.01 and  # Low variance
                    coverage_variance < 100 and   # Low variance
                    abs(comparison.overall_impact) > 0.1  # Substantial impact
                )
            else:
                comparison.statistical_significance = False
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate statistical significance: {e}")
            comparison.statistical_significance = False
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_impact_score(self, impact_analysis: Dict[str, Any]) -> float:
        """Calculate overall impact score from impact analysis."""
        # Weighted combination of different impact metrics
        validity_weight = 0.4
        coverage_weight = 0.35
        defect_weight = 0.25
        
        # Normalize changes to [0, 1] scale
        validity_impact = abs(impact_analysis.get('validity_change', 0))
        coverage_impact = abs(impact_analysis.get('coverage_change', 0)) / 100.0  # Convert percentage
        defect_impact = min(abs(impact_analysis.get('defect_change', 0)) / 10.0, 1.0)  # Cap at 10 defects
        
        overall_score = (
            validity_weight * validity_impact +
            coverage_weight * coverage_impact +
            defect_weight * defect_impact
        )
        
        return min(overall_score, 1.0)  # Cap at 1.0
    
    def _calculate_batch_impact(
        self, 
        original_report: FeedbackReport, 
        perturbed_report: FeedbackReport,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate impact for batch analysis."""
        return {
            'validity_change': perturbed_report.get_validity_rate() - original_report.get_validity_rate(),
            'coverage_change': perturbed_report.get_coverage_percentage() - original_report.get_coverage_percentage(),
            'defect_change': perturbed_report.get_defect_count() - original_report.get_defect_count(),
            'performance_change': (
                perturbed_report.performance_metrics.total_execution_time - 
                original_report.performance_metrics.total_execution_time
            ),
            'impact_score': self._calculate_impact_score({
                'validity_change': perturbed_report.get_validity_rate() - original_report.get_validity_rate(),
                'coverage_change': perturbed_report.get_coverage_percentage() - original_report.get_coverage_percentage(),
                'defect_change': perturbed_report.get_defect_count() - original_report.get_defect_count()
            }),
            'metadata': metadata,
            'session_ids': {
                'original': original_report.session_id,
                'perturbed': perturbed_report.session_id
            }
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def set_random_seed(self, seed: int) -> None:
        """Set the random seed for deterministic execution.
        
        Args:
            seed: Random seed to set
        """
        self.seed_manager.set_master_seed(seed)
        self.config.random_seed = seed
        self.logger.info(f"Random seed set to: {seed}")
    
    def get_random_seed(self) -> int:
        """Get the current random seed.
        
        Returns:
            Current master seed
        """
        return self.seed_manager.get_master_seed()
    
    def export_experiment_state(self) -> Dict[str, Any]:
        """Export current experiment state for reproduction.
        
        Returns:
            Dictionary with complete experiment state
        """
        return {
            'config': self.config.__dict__,
            'seed_state': self.seed_manager.export_state(),
            'session_count': self.total_sessions,
            'total_execution_time': self.total_execution_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def import_experiment_state(self, state_dict: Dict[str, Any]) -> None:
        """Import experiment state for reproduction.
        
        Args:
            state_dict: State dictionary from export_experiment_state()
        """
        # Import seed state
        if 'seed_state' in state_dict:
            self.seed_manager.import_state(state_dict['seed_state'])
        
        # Update configuration
        if 'config' in state_dict:
            config_dict = state_dict['config']
            self.config = FuzzerConfig.from_dict(config_dict)
        
        self.logger.info("Experiment state imported successfully")
    
    def reproduce_experiment(self, experiment_id: str, document_path: str) -> FeedbackReport:
        """Reproduce a previous experiment by ID.
        
        Args:
            experiment_id: ID of experiment to reproduce
            document_path: Path to document file
            
        Returns:
            FeedbackReport from reproduced experiment
        """
        self.logger.info(f"Reproducing experiment: {experiment_id}")
        
        # Load experiment metadata
        metadata = self.metadata_manager.load_metadata(experiment_id)
        
        # Validate reproducibility
        validation = metadata.validate_reproducibility(document_path)
        if not all(validation.values()):
            self.logger.warning("Reproducibility validation failed:")
            for check, result in validation.items():
                if not result:
                    self.logger.warning(f"  - {check}: FAILED")
        
        # Import experiment state
        self.config = metadata.fuzzer_config
        self.seed_manager.import_state(metadata.random_seed_state)
        
        # Run fuzzing session
        with open(document_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        
        report = self.run_fuzzing_session(
            document_content=document_content,
            session_id=f"reproduced_{experiment_id}",
            document_path=document_path,
            save_metadata=False  # Don't save metadata for reproduced experiments
        )
        
        # Compare results with original
        original_validity = metadata.validity_rate
        original_coverage = metadata.coverage_percentage
        original_defects = metadata.defects_found
        
        self.logger.info("Reproduction comparison:")
        self.logger.info(f"  Validity: {report.get_validity_rate():.3f} vs {original_validity:.3f}")
        self.logger.info(f"  Coverage: {report.get_coverage_percentage():.1f}% vs {original_coverage:.1f}%")
        self.logger.info(f"  Defects: {report.get_defect_count()} vs {original_defects}")
        
        return report
    
    def create_reproduction_package(self, experiment_id: str, output_dir: str) -> str:
        """Create a reproduction package for an experiment.
        
        Args:
            experiment_id: Experiment ID to package
            output_dir: Directory to create package in
            
        Returns:
            Path to created reproduction package
        """
        return self.metadata_manager.create_reproduction_package(experiment_id, output_dir)
    
    def list_experiments(self) -> List[str]:
        """List all available experiment IDs.
        
        Returns:
            List of experiment IDs
        """
        return self.metadata_manager.list_experiments()
    
    def get_experiment_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all experiments.
        
        Returns:
            List of experiment summary dictionaries
        """
        return self.metadata_manager.get_experiment_summaries()
    
    def validate_deterministic_behavior(self, document_content: str, runs: int = 3) -> Dict[str, Any]:
        """Validate that the fuzzer produces deterministic results.
        
        Args:
            document_content: Document content to test
            runs: Number of runs to perform
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating deterministic behavior with {runs} runs")
        
        # Store original seed
        original_seed = self.seed_manager.get_master_seed()
        
        results = []
        
        try:
            for run in range(runs):
                # Reset to same seed for each run
                self.seed_manager.set_master_seed(original_seed)
                
                # Run fuzzing session
                report = self.run_fuzzing_session(
                    document_content=document_content,
                    session_id=f"deterministic_test_{run+1}",
                    save_metadata=False
                )
                
                results.append({
                    'run': run + 1,
                    'validity_rate': report.get_validity_rate(),
                    'coverage_percentage': report.get_coverage_percentage(),
                    'defect_count': report.get_defect_count(),
                    'test_count': report.total_test_cases
                })
            
            # Check if all results are identical
            first_result = results[0]
            all_identical = all(
                (
                    abs(r['validity_rate'] - first_result['validity_rate']) < 1e-10 and
                    abs(r['coverage_percentage'] - first_result['coverage_percentage']) < 1e-10 and
                    r['defect_count'] == first_result['defect_count'] and
                    r['test_count'] == first_result['test_count']
                )
                for r in results
            )
            
            validation_result = {
                'deterministic': all_identical,
                'runs': runs,
                'results': results,
                'seed_used': original_seed,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            if all_identical:
                self.logger.info("✅ Deterministic behavior validated successfully")
            else:
                self.logger.warning("⚠️ Non-deterministic behavior detected")
                
            return validation_result
            
        finally:
            # Restore original seed
            self.seed_manager.set_master_seed(original_seed)

def main():
    """CLI entry point for fuzzer simulator."""
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="LLM-Assisted Fuzzer Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fuzzer on a document
  fuzzer-simulator --input doc.md --output results.json
  
  # Run with custom configuration
  fuzzer-simulator --input doc.md --config fuzzer_config.yaml
  
  # Run batch fuzzing
  fuzzer-simulator --batch --input-dir documents/ --output-dir results/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input document file or directory (for batch mode)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file (YAML/JSON)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run in batch mode (input should be directory)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    from src.utils.logger import get_logger
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = get_logger("FuzzerSimulator", level=log_level)
    
    try:
        # Load configuration
        from src.fuzzer.config_integration import load_integrated_fuzzer_config
        
        cli_overrides = {}
        if args.seed is not None:
            cli_overrides['random_seed'] = args.seed
        
        config = load_integrated_fuzzer_config(
            config_file_path=args.config,
            cli_overrides=cli_overrides
        )
        
        # Create fuzzer
        fuzzer = LLMFuzzerSimulator(config)
        
        if args.batch:
            # Batch mode
            input_dir = Path(args.input)
            if not input_dir.is_dir():
                logger.error(f"Input directory not found: {input_dir}")
                return 1
            
            # Find all supported files
            supported_extensions = ['.md', '.txt', '.py', '.java', '.rst', '.adoc']
            documents = []
            file_paths = []
            
            for ext in supported_extensions:
                for file_path in input_dir.glob(f"*{ext}"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        documents.append(content)
                        file_paths.append(str(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")
            
            if not documents:
                logger.error(f"No supported documents found in {input_dir}")
                return 1
            
            logger.info(f"Running batch fuzzing on {len(documents)} documents")
            
            # Run batch fuzzing
            reports = fuzzer.run_batch_fuzzing(documents, session_prefix="batch")
            
            # Generate aggregated report
            aggregated_report = fuzzer.generate_aggregated_report(reports, "batch_fuzzing")
            
            # Add file paths to report
            aggregated_report['file_paths'] = file_paths
            
            result = aggregated_report
            
        else:
            # Single document mode
            input_file = Path(args.input)
            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                return 1
            
            # Read document
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    document_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read input file: {e}")
                return 1
            
            logger.info(f"Running fuzzer on: {input_file}")
            
            # Run fuzzing session
            report = fuzzer.run_fuzzing_session(
                document_content=document_content,
                session_id="cli_session",
                document_path=str(input_file)
            )
            
            result = report.to_dict()
        
        # Save results
        if args.output:
            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved to: {output_file}")
        else:
            # Print to stdout
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Fuzzer execution interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Fuzzer execution failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
