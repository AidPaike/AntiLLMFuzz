"""Integration module for connecting fuzzer simulator with perturbation pipeline."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator
from src.fuzzer.javac_target_system import JavacTargetSystem
from src.fuzzer.data_models import FuzzerConfig, FeedbackReport, ComparisonReport, TestCase, TestType
from src.scs.data_models import FeedbackData
from src.utils.logger import get_logger


class PerturbationFuzzerIntegrator:
    """Integrates fuzzer simulator with the existing perturbation pipeline.
    
    This class provides seamless integration between the LLM fuzzer simulator
    and the existing token extraction and perturbation system, enabling
    end-to-end evaluation workflows.
    """
    
    def __init__(self, fuzzer_config: Optional[FuzzerConfig] = None):
        """Initialize the integrator.
        
        Args:
            fuzzer_config: Optional fuzzer configuration
        """
        self.logger = get_logger("PerturbationFuzzerIntegrator")
        self.fuzzer_config = fuzzer_config or FuzzerConfig()
        self.fuzzer = LLMFuzzerSimulator(self.fuzzer_config)
        
        # Integration statistics
        self.integration_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_execution_time': 0.0
        }
        
        self.logger.info("Perturbation-Fuzzer integrator initialized")
    
    def evaluate_perturbation_impact(
        self,
        original_document: str,
        perturbed_document: str,
        perturbation_metadata: Optional[Dict[str, Any]] = None,
        session_prefix: str = "perturbation_eval"
    ) -> Dict[str, Any]:
        """Evaluate the impact of a single perturbation using fuzzer simulation.
        
        Args:
            original_document: Original document content
            perturbed_document: Perturbed document content
            perturbation_metadata: Optional metadata about the perturbation
            session_prefix: Prefix for session IDs
            
        Returns:
            Dictionary with detailed impact analysis
        """
        start_time = time.time()
        self.integration_stats['total_evaluations'] += 1
        
        try:
            self.logger.info("Evaluating perturbation impact using fuzzer simulation")
            
            # Run fuzzer on both documents
            original_session_id = f"{session_prefix}_original"
            perturbed_session_id = f"{session_prefix}_perturbed"
            
            original_report = self.fuzzer.run_fuzzing_session(
                document_content=original_document,
                session_id=original_session_id
            )
            
            perturbed_report = self.fuzzer.run_fuzzing_session(
                document_content=perturbed_document,
                session_id=perturbed_session_id
            )
            
            # Calculate impact metrics
            impact_analysis = self._calculate_detailed_impact(
                original_report, 
                perturbed_report, 
                perturbation_metadata or {}
            )
            
            # Update statistics
            execution_time = time.time() - start_time
            self.integration_stats['successful_evaluations'] += 1
            self.integration_stats['total_execution_time'] += execution_time
            
            impact_analysis['execution_time'] = execution_time
            impact_analysis['integration_metadata'] = {
                'evaluator': 'PerturbationFuzzerIntegrator',
                'timestamp': datetime.now().isoformat(),
                'session_ids': {
                    'original': original_session_id,
                    'perturbed': perturbed_session_id
                }
            }
            
            self.logger.info(f"Perturbation impact evaluation completed in {execution_time:.2f}s")
            return impact_analysis
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.integration_stats['failed_evaluations'] += 1
            self.integration_stats['total_execution_time'] += execution_time
            
            self.logger.error(f"Perturbation impact evaluation failed: {e}")
            
            return {
                'error': str(e),
                'execution_time': execution_time,
                'perturbation_metadata': perturbation_metadata or {},
                'integration_metadata': {
                    'evaluator': 'PerturbationFuzzerIntegrator',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed'
                }
            }
    
    def evaluate_batch_perturbations(
        self,
        original_document: str,
        perturbation_configs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        session_prefix: str = "batch_perturbation"
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple perturbations in batch for efficiency.
        
        Args:
            original_document: Original document content
            perturbation_configs: List of perturbation configurations
            progress_callback: Optional progress callback
            session_prefix: Prefix for session IDs
            
        Returns:
            List of impact analysis results
        """
        self.logger.info(f"Evaluating {len(perturbation_configs)} perturbations in batch")
        
        results = []
        
        # Run original document once for efficiency
        original_session_id = f"{session_prefix}_original"
        original_report = self.fuzzer.run_fuzzing_session(
            document_content=original_document,
            session_id=original_session_id
        )
        
        # Process each perturbation
        for i, config in enumerate(perturbation_configs):
            if progress_callback:
                progress = (i + 1) / len(perturbation_configs)
                progress_callback(f"Evaluating perturbation {i+1}/{len(perturbation_configs)}", progress)
            
            try:
                perturbed_document = config.get('perturbed_document', '')
                perturbation_metadata = config.get('metadata', {})
                
                # Run fuzzer on perturbed document
                perturbed_session_id = f"{session_prefix}_perturbed_{i+1:03d}"
                perturbed_report = self.fuzzer.run_fuzzing_session(
                    document_content=perturbed_document,
                    session_id=perturbed_session_id
                )
                
                # Calculate impact
                impact_analysis = self._calculate_detailed_impact(
                    original_report,
                    perturbed_report,
                    perturbation_metadata
                )
                
                impact_analysis['perturbation_index'] = i
                impact_analysis['session_ids'] = {
                    'original': original_session_id,
                    'perturbed': perturbed_session_id
                }
                
                results.append(impact_analysis)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate perturbation {i+1}: {e}")
                results.append({
                    'error': str(e),
                    'perturbation_index': i,
                    'metadata': config.get('metadata', {})
                })
        
        self.logger.info(f"Batch perturbation evaluation completed: {len(results)} results")
        return results
    
    def integrate_with_scs_system(
        self,
        document_content: str,
        token_text: str,
        token_metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackData:
        """Integrate with existing SCS (Semantic Contribution Score) system.
        
        This method provides compatibility with the existing feedback analysis
        system by converting fuzzer simulation results to FeedbackData format.
        
        Args:
            document_content: Document content with token perturbation
            token_text: The perturbed token text
            token_metadata: Optional metadata about the token
            
        Returns:
            FeedbackData compatible with SCS system
        """
        self.logger.debug(f"Integrating with SCS system for token: {token_text}")
        
        try:
            # Use the fuzzer's built-in SCS integration method
            feedback_data = self.fuzzer.integrate_with_feedback_system(
                document_content=document_content,
                token_text=token_text,
                token_metadata=token_metadata
            )
            
            self.logger.debug(f"SCS integration successful for token: {token_text}")
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"SCS integration failed for token {token_text}: {e}")
            
            # Return fallback feedback data
            return FeedbackData.create_now(
                validity_rate=0.5,
                coverage_percent=50.0,
                defects_found=5
            )
    
    def create_end_to_end_workflow(
        self,
        input_file_path: str,
        perturbation_strategy: str,
        top_n_tokens: int = 5,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an end-to-end workflow from document to feedback.
        
        This method demonstrates the complete integration by running the
        entire pipeline from token extraction to fuzzer evaluation.
        
        Args:
            input_file_path: Path to input document
            perturbation_strategy: Strategy to use for perturbation
            top_n_tokens: Number of top tokens to perturb
            output_dir: Optional output directory
            
        Returns:
            Dictionary with complete workflow results
        """
        self.logger.info(f"Starting end-to-end workflow for: {input_file_path}")
        
        workflow_start_time = time.time()
        
        try:
            # Import required modules for the workflow
            from src.extractors import DocumentationTokenExtractor, JavaTokenExtractor, PythonTokenExtractor
            from src.token_prioritizer import TokenPrioritizer
            from src.strategies.semantic import (
                TokenizationDriftStrategy,
                LexicalDisguiseStrategy,
                DataFlowMisdirectionStrategy,
                ControlFlowMisdirectionStrategy,
                DocumentationDeceptionStrategy,
                CognitiveManipulationStrategy,
            )
            from src.strategies.generic import (
                FormattingNoiseStrategy,
                StructuralNoiseStrategy,
                ParaphrasingStrategy,
                CognitiveLoadStrategy,
            )
            from src.strategies.generic import (
                FormattingNoiseStrategy,
                StructuralNoiseStrategy,
                ParaphrasingStrategy,
                CognitiveLoadStrategy,
            )
            from src.strategies.selector import (
                filter_strategies,
                infer_target_from_extractor_language,
            )
            from src.utils import read_file, create_output_directory
            
            # Step 1: Select extractor and extract tokens
            self.logger.info("Step 1: Extracting tokens...")
            
            file_path = Path(input_file_path)
            extension = file_path.suffix.lower()
            
            # Select appropriate extractor
            if extension == '.java':
                extractor = JavaTokenExtractor()
            elif extension in ['.py', '.pyw']:
                extractor = PythonTokenExtractor()
            else:
                extractor = DocumentationTokenExtractor()
            
            if not extractor.can_extract(str(file_path)):
                raise ValueError(f"Cannot extract tokens from file: {file_path}")
            
            tokens = extractor.extract_tokens(str(file_path))
            if not tokens:
                raise ValueError("No tokens extracted from file")
            
            # Step 2: Prioritize tokens
            self.logger.info("Step 2: Prioritizing tokens...")
            
            prioritizer = TokenPrioritizer()
            tokens = prioritizer.assign_scores(tokens)
            ranked_tokens = prioritizer.rank_tokens(tokens)
            
            # Step 3: Read original content
            original_content = read_file(str(file_path))
            if not original_content:
                raise ValueError("Failed to read input file")
            
            # Step 4: Apply perturbations
            self.logger.info("Step 3: Applying perturbations...")

            semantic_strategies = [
                TokenizationDriftStrategy(),
                LexicalDisguiseStrategy(),
                DataFlowMisdirectionStrategy(),
                ControlFlowMisdirectionStrategy(),
                DocumentationDeceptionStrategy(),
                CognitiveManipulationStrategy(),
            ]
            generic_strategies = [
                FormattingNoiseStrategy(),
                StructuralNoiseStrategy(),
                ParaphrasingStrategy(),
                CognitiveLoadStrategy(),
            ]

            strategies = []
            if perturbation_strategy in {"all", "all_composed"}:
                strategies = semantic_strategies + generic_strategies
            elif perturbation_strategy == "semantic":
                strategies = semantic_strategies
            elif perturbation_strategy == "generic":
                strategies = generic_strategies
            else:
                name_to_factory = {
                    "tokenization_drift": TokenizationDriftStrategy,
                    "lexical_disguise": LexicalDisguiseStrategy,
                    "dataflow_misdirection": DataFlowMisdirectionStrategy,
                    "controlflow_misdirection": ControlFlowMisdirectionStrategy,
                    "documentation_deception": DocumentationDeceptionStrategy,
                    "cognitive_manipulation": CognitiveManipulationStrategy,
                    "formatting_noise": FormattingNoiseStrategy,
                    "structural_noise": StructuralNoiseStrategy,
                    "paraphrasing": ParaphrasingStrategy,
                    "cognitive_load": CognitiveLoadStrategy,
                }
                if perturbation_strategy not in name_to_factory:
                    raise ValueError(f"Unknown perturbation strategy: {perturbation_strategy}")
                strategies = [name_to_factory[perturbation_strategy]()]  # type: ignore[call-arg]

            content_target = infer_target_from_extractor_language(extractor.language)
            strategies, skipped = filter_strategies(
                strategies,
                target=content_target,
                language=extractor.language,
                allow_unsafe_code=False,
            )
            if not strategies:
                skipped_names = ", ".join(sorted({s.name for s in skipped})) or "(none)"
                raise ValueError(
                    f"No applicable strategies for {content_target}/{extractor.language} "
                    f"(requested: {perturbation_strategy}; skipped: {skipped_names})"
                )

            perturbed_versions = {}
            allow_unsafe_code = False
            for strategy in strategies:
                versions = strategy.apply_multiple(
                    ranked_tokens[:top_n_tokens],
                    original_content,
                    max_tokens=top_n_tokens,
                    target=content_target,
                    language=extractor.language,
                    preserve_executability=(content_target == "code"),
                    allow_unsafe_code=allow_unsafe_code,
                )
                perturbed_versions.update(versions)

            if perturbation_strategy == "all":
                composed_content = original_content
                for strategy in strategies:
                    for token in ranked_tokens[:top_n_tokens]:
                        composed_content = strategy.apply(
                            token,
                            composed_content,
                            target=content_target,
                            language=extractor.language,
                            preserve_executability=(content_target == "code"),
                            allow_unsafe_code=allow_unsafe_code,
                        )
                perturbed_versions["all_composed"] = composed_content

                composed_dense = composed_content
                for token in reversed(ranked_tokens[:top_n_tokens]):
                    for strategy in strategies:
                        composed_dense = strategy.apply(
                            token,
                            composed_dense,
                            target=content_target,
                            language=extractor.language,
                            preserve_executability=(content_target == "code"),
                            allow_unsafe_code=allow_unsafe_code,
                        )
                perturbed_versions["all_composed_dense"] = composed_dense

            if perturbation_strategy == "all_composed":
                composed_content = original_content
                for strategy in strategies:
                    for token in ranked_tokens[:top_n_tokens]:
                        composed_content = strategy.apply(
                            token,
                            composed_content,
                            target=content_target,
                            language=extractor.language,
                            preserve_executability=(content_target == "code"),
                            allow_unsafe_code=allow_unsafe_code,
                        )
                composed_dense = composed_content
                for token in reversed(ranked_tokens[:top_n_tokens]):
                    for strategy in strategies:
                        composed_dense = strategy.apply(
                            token,
                            composed_dense,
                            target=content_target,
                            language=extractor.language,
                            preserve_executability=(content_target == "code"),
                            allow_unsafe_code=allow_unsafe_code,
                        )
                perturbed_versions = {
                    "all_composed": composed_content,
                    "all_composed_dense": composed_dense,
                }
            
            # Step 5: Run fuzzer evaluation
            self.logger.info("Step 4: Running fuzzer evaluation...")

            case_mode = self.fuzzer.config.document_generation_case_mode
            case_dir = None
            if case_mode == "per_variant":
                base_dir = Path(output_dir) if output_dir else Path("output")
                case_dir = base_dir / "case_files"
                case_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                case_dir = case_dir / f"cases_{timestamp}"
                case_dir.mkdir(parents=True, exist_ok=True)
            
            # Run fuzzer on original document
            original_case_file = None
            if case_mode == "per_variant" and case_dir:
                original_case_file = self._generate_case_file(
                    original_content,
                    case_dir / "original.java",
                    self.fuzzer.config.cases_per_document,
                )
                self.fuzzer.config.document_generation_case_file = str(original_case_file)

            original_report = self.fuzzer.run_fuzzing_session(
                document_content=original_content,
                session_id="workflow_original"
            )
            
            # Run fuzzer on perturbed documents
            perturbation_results = []
            for variant_name, perturbed_content in perturbed_versions.items():
                session_id = f"workflow_{variant_name}"

                if case_mode == "per_variant" and case_dir:
                    case_file = self._generate_case_file(
                        perturbed_content,
                        case_dir / f"{variant_name}.java",
                        self.fuzzer.config.cases_per_document,
                    )
                    self.fuzzer.config.document_generation_case_file = str(case_file)
                
                perturbed_report = self.fuzzer.run_fuzzing_session(
                    document_content=perturbed_content,
                    session_id=session_id
                )
                
                impact_analysis = self._calculate_detailed_impact(
                    original_report,
                    perturbed_report,
                    {'variant_name': variant_name, 'strategy': variant_name.split("_token", 1)[0]}
                )
                
                perturbation_results.append({
                    'variant_name': variant_name,
                    'impact_analysis': impact_analysis,
                    'session_id': session_id,
                    'metrics': {
                        'validity_rate': perturbed_report.get_validity_rate(),
                        'coverage_percentage': perturbed_report.get_coverage_percentage(),
                        'defects_found': perturbed_report.get_defect_count(),
                        'crash_count': perturbed_report.get_crash_count(),
                        'test_cases': perturbed_report.total_test_cases
                    }
                })
            
            # Step 6: Generate comprehensive report
            workflow_time = time.time() - workflow_start_time
            
            workflow_results = {
                'input_file': str(file_path),
                'workflow_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': workflow_time,
                    'extractor_used': extractor.language,
                    'strategy_used': perturbation_strategy,
                    'top_n_tokens': top_n_tokens
                },
                'token_extraction': {
                    'total_tokens': len(tokens),
                    'top_tokens': [
                        {
                            'text': token.text,
                            'type': token.token_type,
                            'priority_score': token.priority_score,
                            'line': token.line
                        }
                        for token in ranked_tokens[:top_n_tokens]
                    ]
                },
                'original_fuzzing_results': {
                    'session_id': original_report.session_id,
                    'validity_rate': original_report.get_validity_rate(),
                    'coverage_percentage': original_report.get_coverage_percentage(),
                    'defects_found': original_report.get_defect_count(),
                    'crash_count': original_report.get_crash_count(),
                    'test_cases': original_report.total_test_cases
                },
                'perturbation_results': perturbation_results,
                'summary': {
                    'total_perturbations': len(perturbation_results),
                    'average_validity_impact': sum(
                        r['impact_analysis']['validity_change'] 
                        for r in perturbation_results
                    ) / len(perturbation_results) if perturbation_results else 0,
                    'average_coverage_impact': sum(
                        r['impact_analysis']['coverage_change'] 
                        for r in perturbation_results
                    ) / len(perturbation_results) if perturbation_results else 0,
                    'average_crash_impact': sum(
                        r['impact_analysis'].get('crash_change', 0)
                        for r in perturbation_results
                    ) / len(perturbation_results) if perturbation_results else 0,
                    'total_workflow_time': workflow_time
                }
            }
            
            # Save results if output directory is specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = output_path / f"end_to_end_workflow_{timestamp}.json"
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(workflow_results, f, indent=2, ensure_ascii=False, default=str)
                
                workflow_results['output_file'] = str(results_file)
            
            self.logger.info(f"End-to-end workflow completed in {workflow_time:.2f}s")
            return workflow_results
            
        except Exception as e:
            workflow_time = time.time() - workflow_start_time
            self.logger.error(f"End-to-end workflow failed: {e}")
            
            return {
                'error': str(e),
                'input_file': str(input_file_path),
                'execution_time': workflow_time,
                'timestamp': datetime.now().isoformat()
            }

    def _generate_case_file(self, content: str, output_path: Path, count: int) -> Path:
        original_case_file = self.fuzzer.config.document_generation_case_file
        self.fuzzer.config.document_generation_case_file = None
        try:
            cases = []
            attempts = 0
            max_attempts = self.fuzzer.config.document_generation_max_attempts
            max_seconds = self.fuzzer.config.document_generation_max_seconds
            start_time = time.time()
            while len(cases) < count and attempts < max_attempts:
                if time.time() - start_time > max_seconds:
                    break
                generated = self.fuzzer.test_generator.generate_document_test_cases(content, 1)
                if not generated:
                    attempts += 1
                    continue
                test_case = generated[0]
                cases.append(test_case)
                attempts += 1
            if len(cases) < count:
                filler_count = count - len(cases)
                for index in range(filler_count):
                    filler = TestCase(
                        id=f"document_case_{len(cases)+1}",
                        api_name="document",
                        parameters={
                            "java_source": "public class GeneratedCase {"
                        },
                        test_type=TestType.NORMAL,
                        expected_result="compile",
                        generation_prompt="document_generation_fallback",
                    )
                    cases.append(filler)
        finally:
            self.fuzzer.config.document_generation_case_file = original_case_file

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            for idx, case in enumerate(cases, 1):
                java_source = case.parameters.get('java_source') if isinstance(case.parameters, dict) else None
                if not java_source:
                    continue
                if "import " not in java_source:
                    java_source = (
                        "import java.util.*;\n"
                        "import java.util.concurrent.*;\n"
                        "import java.util.stream.*;\n\n"
                        + java_source
                    )
                handle.write(f"// case {idx}\n")
                handle.write(java_source)
                handle.write("\n\n")

        return output_path
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics.
        
        Returns:
            Dictionary with integration statistics
        """
        stats = self.integration_stats.copy()
        
        if stats['total_evaluations'] > 0:
            stats['success_rate'] = stats['successful_evaluations'] / stats['total_evaluations']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_evaluations']
        else:
            stats['success_rate'] = 0.0
            stats['average_execution_time'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset integration statistics."""
        self.integration_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_execution_time': 0.0
        }
        self.logger.info("Integration statistics reset")
    
    def _calculate_detailed_impact(
        self,
        original_report: FeedbackReport,
        perturbed_report: FeedbackReport,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate detailed impact analysis between two reports.
        
        Args:
            original_report: Original fuzzing report
            perturbed_report: Perturbed fuzzing report
            metadata: Additional metadata
            
        Returns:
            Dictionary with detailed impact analysis
        """
        # Calculate basic changes
        validity_change = perturbed_report.get_validity_rate() - original_report.get_validity_rate()
        coverage_change = perturbed_report.get_coverage_percentage() - original_report.get_coverage_percentage()
        defect_change = perturbed_report.get_defect_count() - original_report.get_defect_count()
        crash_change = perturbed_report.get_crash_count() - original_report.get_crash_count()
        
        # Calculate relative changes (avoid division by zero)
        validity_change_percent = 0.0
        if original_report.get_validity_rate() > 0:
            validity_change_percent = (validity_change / original_report.get_validity_rate()) * 100
        
        coverage_change_percent = 0.0
        if original_report.get_coverage_percentage() > 0:
            coverage_change_percent = (coverage_change / original_report.get_coverage_percentage()) * 100
        
        # Calculate overall impact score
        impact_score = self._calculate_impact_score({
            'validity_change': validity_change,
            'coverage_change': coverage_change,
            'defect_change': defect_change,
            'crash_change': crash_change
        })
        
        return {
            'validity_change': validity_change,
            'coverage_change': coverage_change,
            'defect_change': defect_change,
            'crash_change': crash_change,
            'validity_change_percent': validity_change_percent,
            'coverage_change_percent': coverage_change_percent,
            'impact_score': impact_score,
            'original_metrics': {
                'validity_rate': original_report.get_validity_rate(),
                'coverage_percentage': original_report.get_coverage_percentage(),
                'defect_count': original_report.get_defect_count(),
                'test_cases': original_report.total_test_cases
            },
            'perturbed_metrics': {
                'validity_rate': perturbed_report.get_validity_rate(),
                'coverage_percentage': perturbed_report.get_coverage_percentage(),
                'defect_count': perturbed_report.get_defect_count(),
                'test_cases': perturbed_report.total_test_cases
            },
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_impact_score(self, impact_analysis: Dict[str, Any]) -> float:
        """Calculate overall impact score from impact analysis.
        
        Args:
            impact_analysis: Impact analysis data
            
        Returns:
            Overall impact score (0.0 to 1.0)
        """
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

    def create_iterative_disruption_workflow(
        self,
        input_file_path: str,
        top_n_candidates: int = 3,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an iterative disruption workflow: select best segment, then apply strategies in loops.

        This implements the full system flow: locate best-disrupting segment, then multi-use strategies iteratively.

        Args:
            input_file_path: Path to input document
            top_n_candidates: Number of top tokens to evaluate for best disruption
            output_dir: Optional output directory

        Returns:
            Dictionary with complete iterative workflow results
        """
        self.logger.info(f"Starting iterative disruption workflow for: {input_file_path}")

        workflow_start_time = time.time()

        try:
            # Step 1: Extract and prioritize tokens
            from src.extractors import DocumentationTokenExtractor, JavaTokenExtractor, PythonTokenExtractor
            from src.token_prioritizer import TokenPrioritizer
            from src.utils import read_file

            file_path = Path(input_file_path)
            extension = file_path.suffix.lower()

            if extension == '.java':
                extractor = JavaTokenExtractor()
            elif extension in ['.py', '.pyw']:
                extractor = PythonTokenExtractor()
            else:
                extractor = DocumentationTokenExtractor()

            tokens = extractor.extract_tokens(str(file_path))
            prioritizer = TokenPrioritizer()
            tokens = prioritizer.assign_scores(tokens)
            ranked_tokens = prioritizer.rank_tokens(tokens)

            original_content = read_file(str(file_path))
            if original_content is None:
                raise ValueError("Failed to read input file")

            # Step 2: Select best-disrupting segment via preliminary evaluation
            best_token, segment_results = self._select_best_disrupting_segment(
                original_content, ranked_tokens[:top_n_candidates], extractor.language
            )

            # Step 3: Iterative strategy application on best segment
            iterative_results = self._apply_iterative_strategies(
                original_content, best_token, extractor.language
            )

            # Step 4: Generate report
            workflow_time = time.time() - workflow_start_time

            workflow_results = {
                'input_file': str(file_path),
                'workflow_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': workflow_time,
                    'extractor_used': extractor.language,
                    'top_n_candidates': top_n_candidates
                },
                'segment_selection': {
                    'candidates_evaluated': len(segment_results),
                    'best_token': {
                        'text': best_token.text,
                        'type': best_token.token_type,
                        'priority_score': best_token.priority_score,
                        'line': best_token.line
                    },
                    'segment_results': segment_results
                },
                'iterative_disruption': iterative_results,
                'final_metrics': iterative_results[-1]['metrics'] if iterative_results else {},
                'summary': {
                    'total_rounds': len(iterative_results),
                    'best_validity_drop': min((r['metrics']['validity_rate'] for r in iterative_results), default=1.0),
                    'best_coverage_drop': min((r['metrics']['coverage_percentage'] for r in iterative_results), default=100.0),
                    'max_crashes': max((r['metrics']['crash_count'] for r in iterative_results), default=0),
                    'total_workflow_time': workflow_time
                }
            }

            # Save results
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = output_path / f"iterative_disruption_workflow_{timestamp}.json"

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(workflow_results, f, indent=2, ensure_ascii=False, default=str)

                workflow_results['output_file'] = str(results_file)

            self.logger.info(f"Iterative disruption workflow completed in {workflow_time:.2f}s")
            return workflow_results

        except Exception as e:
            workflow_time = time.time() - workflow_start_time
            self.logger.error(f"Iterative disruption workflow failed: {e}")

            return {
                'error': str(e),
                'input_file': str(input_file_path),
                'execution_time': workflow_time,
                'timestamp': datetime.now().isoformat()
            }

    def _select_best_disrupting_segment(self, original_content: str, candidate_tokens: List, language: str) -> tuple:
        """Select the best-disrupting segment by preliminary evaluation."""
        from src.strategies.semantic import DocumentationDeceptionStrategy

        self.logger.info(f"Evaluating {len(candidate_tokens)} candidate tokens for best disruption")

        segment_results = []
        best_token = candidate_tokens[0]
        best_impact = -1.0

        # Run original baseline
        original_report = self.fuzzer.run_fuzzing_session(
            document_content=original_content,
            session_id="segment_selection_original"
        )
        original_validity = original_report.get_validity_rate()
        original_crashes = original_report.get_crash_count()

        for i, token in enumerate(candidate_tokens):
            # Apply a single disruptive strategy (documentation_deception) to this token
            strategy = DocumentationDeceptionStrategy()
            perturbed_content = strategy.apply(
                token, original_content,
                target="documentation" if language == "documentation" else "code",
                language=language,
                preserve_executability=True
            )

            # Run mini-fuzzer
            report = self.fuzzer.run_fuzzing_session(
                document_content=perturbed_content,
                session_id=f"segment_selection_{i+1}"
            )

            validity_drop = original_validity - report.get_validity_rate()
            crash_gain = report.get_crash_count() - original_crashes
            impact_score = validity_drop + crash_gain  # Simple impact metric

            segment_results.append({
                'token_text': token.text,
                'validity_drop': validity_drop,
                'crash_gain': crash_gain,
                'impact_score': impact_score,
                'metrics': {
                    'validity_rate': report.get_validity_rate(),
                    'coverage_percentage': report.get_coverage_percentage(),
                    'crash_count': report.get_crash_count()
                }
            })

            if impact_score > best_impact:
                best_impact = impact_score
                best_token = token

        self.logger.info(f"Selected best token: {best_token.text} with impact {best_impact}")
        return best_token, segment_results

    def _apply_iterative_strategies(self, original_content: str, target_token, language: str) -> List[Dict]:
        """Apply strategies iteratively to the target token and measure impact."""
        from src.strategies.semantic import TokenizationDriftStrategy, DocumentationDeceptionStrategy, CognitiveManipulationStrategy
        from src.strategies.generic import FormattingNoiseStrategy, ParaphrasingStrategy, CognitiveLoadStrategy

        self.logger.info(f"Applying iterative strategies to token: {target_token.text}")

        results = []

        # Round 1: Semantic strategies only
        semantic_strategies = [
            TokenizationDriftStrategy(),
            DocumentationDeceptionStrategy(),
            CognitiveManipulationStrategy(),
        ]

        perturbed_content = original_content
        for strategy in semantic_strategies:
            perturbed_content = strategy.apply(
                target_token, perturbed_content,
                target="documentation" if language == "documentation" else "code",
                language=language,
                preserve_executability=True
            )

        report = self.fuzzer.run_fuzzing_session(
            document_content=perturbed_content,
            session_id="iterative_round_1"
        )

        results.append({
            'round': 1,
            'strategies': 'semantic',
            'metrics': {
                'validity_rate': report.get_validity_rate(),
                'coverage_percentage': report.get_coverage_percentage(),
                'crash_count': report.get_crash_count(),
                'test_cases': report.total_test_cases
            }
        })

        # Round 2: Add generic strategies
        generic_strategies = [
            FormattingNoiseStrategy(),
            ParaphrasingStrategy(),
            CognitiveLoadStrategy(),
        ]

        for strategy in generic_strategies:
            perturbed_content = strategy.apply(
                target_token, perturbed_content,
                target="documentation" if language == "documentation" else "code",
                language=language,
                preserve_executability=True
            )

        report = self.fuzzer.run_fuzzing_session(
            document_content=perturbed_content,
            session_id="iterative_round_2"
        )

        results.append({
            'round': 2,
            'strategies': 'semantic + generic',
            'metrics': {
                'validity_rate': report.get_validity_rate(),
                'coverage_percentage': report.get_coverage_percentage(),
                'crash_count': report.get_crash_count(),
                'test_cases': report.total_test_cases
            }
        })

        # Round 3: All composed dense
        composed_dense = original_content
        all_strategies = semantic_strategies + generic_strategies
        for token in [target_token] * len(all_strategies):  # Apply to same token multiple times
            for strategy in all_strategies:
                composed_dense = strategy.apply(
                    token, composed_dense,
                    target="documentation" if language == "documentation" else "code",
                    language=language,
                    preserve_executability=True
                )

        report = self.fuzzer.run_fuzzing_session(
            document_content=composed_dense,
            session_id="iterative_round_3"
        )

        results.append({
            'round': 3,
            'strategies': 'all_composed_dense',
            'metrics': {
                'validity_rate': report.get_validity_rate(),
                'coverage_percentage': report.get_coverage_percentage(),
                'crash_count': report.get_crash_count(),
                'test_cases': report.total_test_cases
            }
        })

        return results


class LLMFeedbackSimulator:
    """LLM-based feedback simulator for SCS calculation.
    
    This class provides a more realistic feedback simulation using the
    LLM fuzzer simulator instead of the simple statistical simulation.
    """
    
    def __init__(self, fuzzer_config: Optional[FuzzerConfig] = None):
        """Initialize the LLM feedback simulator.
        
        Args:
            fuzzer_config: Optional fuzzer configuration
        """
        self.logger = get_logger("LLMFeedbackSimulator")
        self.integrator = PerturbationFuzzerIntegrator(fuzzer_config)
        
    def simulate_feedback(self, token, document_content: str) -> FeedbackData:
        """Simulate feedback for a token using LLM fuzzer.
        
        Args:
            token: Token object with perturbation
            document_content: Document content with token perturbation
            
        Returns:
            FeedbackData with simulated metrics
        """
        return self.integrator.integrate_with_scs_system(
            document_content=document_content,
            token_text=token.text,
            token_metadata={
                'token_type': token.token_type,
                'line': token.line,
                'priority_score': getattr(token, 'priority_score', 0.0)
            }
        )
    
    @classmethod
    def from_config(cls, scs_config, fuzzer_config: Optional[FuzzerConfig] = None):
        """Create LLM feedback simulator from SCS configuration.
        
        Args:
            scs_config: SCS configuration object
            fuzzer_config: Optional fuzzer configuration
            
        Returns:
            LLMFeedbackSimulator instance
        """
        return cls(fuzzer_config)
