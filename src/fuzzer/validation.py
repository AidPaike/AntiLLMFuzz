"""System validation and performance testing for fuzzer simulator integration."""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator
from src.fuzzer.data_models import FuzzerConfig
from src.fuzzer.integration import PerturbationFuzzerIntegrator
from src.fuzzer.config_integration import (
    load_integrated_fuzzer_config,
    validate_integration_requirements,
    setup_integration_environment
)
from src.utils.logger import get_logger


class SystemValidator:
    """Comprehensive system validation and performance testing."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the system validator.
        
        Args:
            output_dir: Optional output directory for validation reports
        """
        self.logger = get_logger("SystemValidator")
        self.output_dir = Path(output_dir) if output_dir else Path("validation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation results
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'integration_validation': {},
            'performance_validation': {},
            'functionality_validation': {},
            'stress_testing': {},
            'perturbation_impact_validation': {}
        }
        
        self.logger.info(f"System validator initialized, output dir: {self.output_dir}")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation.
        
        Returns:
            Dictionary with complete validation results
        """
        self.logger.info("Starting comprehensive system validation")
        start_time = time.time()
        
        try:
            # 1. Integration validation
            self.logger.info("Step 1: Integration validation...")
            self.validation_results['integration_validation'] = self._validate_integration()
            
            # 2. Performance validation
            self.logger.info("Step 2: Performance validation...")
            self.validation_results['performance_validation'] = self._validate_performance()
            
            # 3. Functionality validation
            self.logger.info("Step 3: Functionality validation...")
            self.validation_results['functionality_validation'] = self._validate_functionality()
            
            # 4. Stress testing
            self.logger.info("Step 4: Stress testing...")
            self.validation_results['stress_testing'] = self._run_stress_tests()
            
            # 5. Perturbation impact validation
            self.logger.info("Step 5: Perturbation impact validation...")
            self.validation_results['perturbation_impact_validation'] = self._validate_perturbation_impact()
            
            # Calculate overall validation score
            total_time = time.time() - start_time
            self.validation_results['total_validation_time'] = total_time
            self.validation_results['overall_score'] = self._calculate_overall_score()
            
            # Save validation report
            self._save_validation_report()
            
            self.logger.info(f"Comprehensive validation completed in {total_time:.2f}s")
            self.logger.info(f"Overall validation score: {self.validation_results['overall_score']:.1f}/100")
            
            return self.validation_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            self.validation_results['error'] = str(e)
            self.validation_results['total_validation_time'] = time.time() - start_time
            return self.validation_results
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate system integration components."""
        integration_results = {
            'requirements_check': {},
            'environment_setup': False,
            'configuration_loading': False,
            'module_imports': {},
            'cli_integration': False
        }
        
        try:
            # Check integration requirements
            integration_results['requirements_check'] = validate_integration_requirements()
            
            # Test environment setup
            integration_results['environment_setup'] = setup_integration_environment()
            
            # Test configuration loading
            try:
                config = load_integrated_fuzzer_config()
                integration_results['configuration_loading'] = True
                integration_results['config_validation'] = config.validate().is_valid
            except Exception as e:
                self.logger.error(f"Configuration loading failed: {e}")
                integration_results['configuration_loading'] = False
            
            # Test module imports
            modules_to_test = [
                ('fuzzer_simulator', 'src.fuzzer.llm_fuzzer_simulator', 'LLMFuzzerSimulator'),
                ('integration', 'src.fuzzer.integration', 'PerturbationFuzzerIntegrator'),
                ('scs_data', 'src.scs.data_models', 'FeedbackData'),
                ('config_integration', 'src.fuzzer.config_integration', 'load_integrated_fuzzer_config')
            ]
            
            for module_name, module_path, class_name in modules_to_test:
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    getattr(module, class_name)
                    integration_results['module_imports'][module_name] = True
                except Exception as e:
                    self.logger.error(f"Failed to import {module_name}: {e}")
                    integration_results['module_imports'][module_name] = False
            
            # Test CLI integration (basic import test)
            try:
                from src.cli import main
                integration_results['cli_integration'] = True
            except Exception as e:
                self.logger.error(f"CLI integration test failed: {e}")
                integration_results['cli_integration'] = False
            
            # Calculate integration score
            total_checks = (
                len(integration_results['requirements_check']) +
                1 +  # environment_setup
                1 +  # configuration_loading
                len(integration_results['module_imports']) +
                1    # cli_integration
            )
            
            passed_checks = (
                sum(integration_results['requirements_check'].values()) +
                int(integration_results['environment_setup']) +
                int(integration_results['configuration_loading']) +
                sum(integration_results['module_imports'].values()) +
                int(integration_results['cli_integration'])
            )
            
            integration_results['score'] = (passed_checks / total_checks) * 100
            
            self.logger.info(f"Integration validation score: {integration_results['score']:.1f}/100")
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Integration validation failed: {e}")
            integration_results['error'] = str(e)
            integration_results['score'] = 0.0
            return integration_results
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate system performance against requirements."""
        performance_results = {
            'document_processing_speed': {},
            'test_generation_speed': {},
            'memory_usage': {},
            'concurrent_execution': {},
            'scalability': {}
        }
        
        try:
            # Create test fuzzer
            config = load_integrated_fuzzer_config()
            fuzzer = LLMFuzzerSimulator(config)
            
            # Test document processing speed
            performance_results['document_processing_speed'] = self._test_document_processing_speed(fuzzer)
            
            # Test test generation speed
            performance_results['test_generation_speed'] = self._test_generation_speed(fuzzer)
            
            # Test memory usage
            performance_results['memory_usage'] = self._test_memory_usage(fuzzer)
            
            # Test concurrent execution
            performance_results['concurrent_execution'] = self._test_concurrent_execution(fuzzer)
            
            # Test scalability
            performance_results['scalability'] = self._test_scalability(fuzzer)
            
            # Calculate performance score
            performance_results['score'] = self._calculate_performance_score(performance_results)
            
            self.logger.info(f"Performance validation score: {performance_results['score']:.1f}/100")
            return performance_results
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            performance_results['error'] = str(e)
            performance_results['score'] = 0.0
            return performance_results
    
    def _validate_functionality(self) -> Dict[str, Any]:
        """Validate core functionality."""
        functionality_results = {
            'basic_fuzzing_session': False,
            'perturbation_integration': False,
            'scs_integration': False,
            'batch_processing': False,
            'error_handling': False,
            'reproducibility': False
        }
        
        try:
            config = load_integrated_fuzzer_config()
            config.random_seed = 42  # For reproducibility testing
            
            # Test basic fuzzing session
            functionality_results['basic_fuzzing_session'] = self._test_basic_fuzzing_session(config)
            
            # Test perturbation integration
            functionality_results['perturbation_integration'] = self._test_perturbation_integration(config)
            
            # Test SCS integration
            functionality_results['scs_integration'] = self._test_scs_integration(config)
            
            # Test batch processing
            functionality_results['batch_processing'] = self._test_batch_processing(config)
            
            # Test error handling
            functionality_results['error_handling'] = self._test_error_handling(config)
            
            # Test reproducibility
            functionality_results['reproducibility'] = self._test_reproducibility(config)
            
            # Calculate functionality score
            passed_tests = sum(functionality_results.values())
            total_tests = len([k for k in functionality_results.keys() if k != 'score'])
            functionality_results['score'] = (passed_tests / total_tests) * 100
            
            self.logger.info(f"Functionality validation score: {functionality_results['score']:.1f}/100")
            return functionality_results
            
        except Exception as e:
            self.logger.error(f"Functionality validation failed: {e}")
            functionality_results['error'] = str(e)
            functionality_results['score'] = 0.0
            return functionality_results
    
    def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests to validate system reliability."""
        stress_results = {
            'high_load_test': {},
            'memory_stress_test': {},
            'concurrent_sessions_test': {},
            'large_document_test': {},
            'extended_runtime_test': {}
        }
        
        try:
            config = load_integrated_fuzzer_config()
            
            # High load test
            stress_results['high_load_test'] = self._run_high_load_test(config)
            
            # Memory stress test
            stress_results['memory_stress_test'] = self._run_memory_stress_test(config)
            
            # Concurrent sessions test
            stress_results['concurrent_sessions_test'] = self._run_concurrent_sessions_test(config)
            
            # Large document test
            stress_results['large_document_test'] = self._run_large_document_test(config)
            
            # Extended runtime test
            stress_results['extended_runtime_test'] = self._run_extended_runtime_test(config)
            
            # Calculate stress test score
            stress_results['score'] = self._calculate_stress_score(stress_results)
            
            self.logger.info(f"Stress testing score: {stress_results['score']:.1f}/100")
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Stress testing failed: {e}")
            stress_results['error'] = str(e)
            stress_results['score'] = 0.0
            return stress_results
    
    def _validate_perturbation_impact(self) -> Dict[str, Any]:
        """Validate perturbation impact detection."""
        impact_results = {
            'impact_detection': False,
            'statistical_significance': False,
            'comparison_accuracy': False,
            'batch_analysis': False
        }
        
        try:
            config = load_integrated_fuzzer_config()
            integrator = PerturbationFuzzerIntegrator(config)
            
            # Create test documents
            original_doc = """
            def authenticate(username, password):
                if validate_credentials(username, password):
                    return create_session(username)
                return None
            """
            
            perturbed_doc = """
            def authenticate(username, password):
                if validate​_credentials(username, password):  # Zero-width space added
                    return create_session(username)
                return None
            """
            
            # Test impact detection
            try:
                impact_analysis = integrator.evaluate_perturbation_impact(
                    original_document=original_doc,
                    perturbed_document=perturbed_doc
                )
                
                if 'error' not in impact_analysis:
                    impact_results['impact_detection'] = True
                    
                    # Check if impact is measurable
                    has_measurable_impact = (
                        abs(impact_analysis.get('validity_change', 0)) > 0.01 or
                        abs(impact_analysis.get('coverage_change', 0)) > 1.0 or
                        abs(impact_analysis.get('defect_change', 0)) > 0
                    )
                    
                    if has_measurable_impact:
                        impact_results['statistical_significance'] = True
                
            except Exception as e:
                self.logger.error(f"Impact detection test failed: {e}")
            
            # Test comparison accuracy
            try:
                # Run multiple comparisons to test consistency
                results = []
                for i in range(3):
                    result = integrator.evaluate_perturbation_impact(
                        original_document=original_doc,
                        perturbed_document=perturbed_doc,
                        session_prefix=f"comparison_test_{i}"
                    )
                    if 'error' not in result:
                        results.append(result)
                
                if len(results) >= 2:
                    # Check consistency across runs
                    validity_changes = [r['validity_change'] for r in results]
                    if statistics.stdev(validity_changes) < 0.1:  # Low variance
                        impact_results['comparison_accuracy'] = True
                
            except Exception as e:
                self.logger.error(f"Comparison accuracy test failed: {e}")
            
            # Test batch analysis
            try:
                perturbation_configs = [
                    {
                        'perturbed_document': perturbed_doc,
                        'metadata': {'perturbation_type': 'zero_width_space'}
                    }
                ]
                
                batch_results = integrator.evaluate_batch_perturbations(
                    original_document=original_doc,
                    perturbation_configs=perturbation_configs
                )
                
                if batch_results and 'error' not in batch_results[0]:
                    impact_results['batch_analysis'] = True
                
            except Exception as e:
                self.logger.error(f"Batch analysis test failed: {e}")
            
            # Calculate impact validation score
            passed_tests = sum(impact_results.values())
            total_tests = len([k for k in impact_results.keys() if k != 'score'])
            impact_results['score'] = (passed_tests / total_tests) * 100
            
            self.logger.info(f"Perturbation impact validation score: {impact_results['score']:.1f}/100")
            return impact_results
            
        except Exception as e:
            self.logger.error(f"Perturbation impact validation failed: {e}")
            impact_results['error'] = str(e)
            impact_results['score'] = 0.0
            return impact_results
    
    def _test_document_processing_speed(self, fuzzer: LLMFuzzerSimulator) -> Dict[str, Any]:
        """Test document processing speed."""
        test_doc = "def test_function(param): return validate(param)"
        
        try:
            start_time = time.time()
            
            # Process document multiple times
            for _ in range(10):
                fuzzer.document_processor.parse_document(test_doc)
            
            processing_time = (time.time() - start_time) / 10
            lines_per_second = len(test_doc.split('\n')) / processing_time
            
            # Target: 1000+ lines/second
            meets_requirement = lines_per_second >= 1000
            
            return {
                'processing_time_per_doc': processing_time,
                'lines_per_second': lines_per_second,
                'meets_requirement': meets_requirement,
                'target': 1000
            }
            
        except Exception as e:
            return {'error': str(e), 'meets_requirement': False}
    
    def _test_generation_speed(self, fuzzer: LLMFuzzerSimulator) -> Dict[str, Any]:
        """Test test case generation speed."""
        try:
            # Create mock API spec
            from src.fuzzer.data_models import APISpec, ParameterSpec
            
            api_spec = APISpec(
                name="test_api",
                description="Test API",
                parameters=[
                    ParameterSpec(
                        name="param1",
                        type="string",
                        description="Test parameter",
                        required=True,
                        constraints=[],
                        examples=["test"]
                    )
                ],
                return_type="boolean",
                examples=["test_api('test')"],
                constraints=[]
            )
            
            start_time = time.time()
            
            # Generate test cases
            test_cases = fuzzer.test_generator.generate_test_cases([api_spec], 20)
            
            generation_time = time.time() - start_time
            cases_per_minute = (len(test_cases) / generation_time) * 60
            
            # Target: 50+ test cases/minute
            meets_requirement = cases_per_minute >= 50
            
            return {
                'generation_time': generation_time,
                'test_cases_generated': len(test_cases),
                'cases_per_minute': cases_per_minute,
                'meets_requirement': meets_requirement,
                'target': 50
            }
            
        except Exception as e:
            return {'error': str(e), 'meets_requirement': False}
    
    def _test_memory_usage(self, fuzzer: LLMFuzzerSimulator) -> Dict[str, Any]:
        """Test memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run a fuzzing session
            test_doc = "def test_function(): return True"
            fuzzer.run_fuzzing_session(test_doc, "memory_test")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Target: <2GB for typical sessions
            meets_requirement = final_memory < 2048
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'meets_requirement': meets_requirement,
                'target_mb': 2048
            }
            
        except ImportError:
            return {'error': 'psutil not available', 'meets_requirement': False}
        except Exception as e:
            return {'error': str(e), 'meets_requirement': False}
    
    def _test_concurrent_execution(self, fuzzer: LLMFuzzerSimulator) -> Dict[str, Any]:
        """Test concurrent execution capability."""
        try:
            test_docs = [
                "def test1(): return True",
                "def test2(): return False",
                "def test3(): return None"
            ]
            
            start_time = time.time()
            
            # Run batch fuzzing with parallel execution
            reports = fuzzer.run_batch_fuzzing(test_docs, parallel=True)
            
            execution_time = time.time() - start_time
            
            # Check if all sessions completed successfully
            successful_sessions = len([r for r in reports if r.is_successful_session()])
            success_rate = successful_sessions / len(test_docs)
            
            meets_requirement = success_rate >= 0.8  # 80% success rate
            
            return {
                'execution_time': execution_time,
                'total_sessions': len(test_docs),
                'successful_sessions': successful_sessions,
                'success_rate': success_rate,
                'meets_requirement': meets_requirement,
                'target_success_rate': 0.8
            }
            
        except Exception as e:
            return {'error': str(e), 'meets_requirement': False}
    
    def _test_scalability(self, fuzzer: LLMFuzzerSimulator) -> Dict[str, Any]:
        """Test system scalability."""
        try:
            # Test with increasing document sizes
            scalability_results = []
            
            for size_multiplier in [1, 5, 10]:
                base_doc = "def test_function(): return True\n"
                large_doc = base_doc * size_multiplier
                
                start_time = time.time()
                report = fuzzer.run_fuzzing_session(large_doc, f"scalability_test_{size_multiplier}")
                execution_time = time.time() - start_time
                
                scalability_results.append({
                    'size_multiplier': size_multiplier,
                    'document_size': len(large_doc),
                    'execution_time': execution_time,
                    'successful': report.is_successful_session()
                })
            
            # Check if execution time scales reasonably
            times = [r['execution_time'] for r in scalability_results if r['successful']]
            if len(times) >= 2:
                # Time should not increase exponentially
                time_ratio = times[-1] / times[0]
                size_ratio = scalability_results[-1]['size_multiplier'] / scalability_results[0]['size_multiplier']
                
                # Time increase should be less than 2x the size increase
                meets_requirement = time_ratio < (size_ratio * 2)
            else:
                meets_requirement = False
            
            return {
                'scalability_results': scalability_results,
                'meets_requirement': meets_requirement
            }
            
        except Exception as e:
            return {'error': str(e), 'meets_requirement': False}
    
    def _test_basic_fuzzing_session(self, config: FuzzerConfig) -> bool:
        """Test basic fuzzing session functionality."""
        try:
            fuzzer = LLMFuzzerSimulator(config)
            test_doc = "def authenticate(user, pass): return validate(user, pass)"
            
            report = fuzzer.run_fuzzing_session(test_doc, "basic_test")
            
            # Check if session completed successfully
            return (
                report.session_id == "basic_test" and
                report.total_test_cases > 0 and
                report.get_validity_rate() >= 0.0 and
                report.get_coverage_percentage() >= 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Basic fuzzing session test failed: {e}")
            return False
    
    def _test_perturbation_integration(self, config: FuzzerConfig) -> bool:
        """Test perturbation integration functionality."""
        try:
            integrator = PerturbationFuzzerIntegrator(config)
            
            original_doc = "def test(): return True"
            perturbed_doc = "def test​(): return True"  # Zero-width space
            
            result = integrator.evaluate_perturbation_impact(original_doc, perturbed_doc)
            
            return 'error' not in result and 'validity_change' in result
            
        except Exception as e:
            self.logger.error(f"Perturbation integration test failed: {e}")
            return False
    
    def _test_scs_integration(self, config: FuzzerConfig) -> bool:
        """Test SCS integration functionality."""
        try:
            integrator = PerturbationFuzzerIntegrator(config)
            
            test_doc = "def validate(input): return True"
            feedback_data = integrator.integrate_with_scs_system(test_doc, "validate")
            
            return (
                hasattr(feedback_data, 'validity_rate') and
                hasattr(feedback_data, 'coverage_percent') and
                hasattr(feedback_data, 'defects_found')
            )
            
        except Exception as e:
            self.logger.error(f"SCS integration test failed: {e}")
            return False
    
    def _test_batch_processing(self, config: FuzzerConfig) -> bool:
        """Test batch processing functionality."""
        try:
            fuzzer = LLMFuzzerSimulator(config)
            
            test_docs = [
                "def test1(): return True",
                "def test2(): return False"
            ]
            
            reports = fuzzer.run_batch_fuzzing(test_docs)
            
            return len(reports) == len(test_docs) and all(r.total_test_cases > 0 for r in reports)
            
        except Exception as e:
            self.logger.error(f"Batch processing test failed: {e}")
            return False
    
    def _test_error_handling(self, config: FuzzerConfig) -> bool:
        """Test error handling functionality."""
        try:
            fuzzer = LLMFuzzerSimulator(config)
            
            # Test with invalid document
            report = fuzzer.run_fuzzing_session("", "error_test")
            
            # Should handle gracefully and return a report
            return report is not None and hasattr(report, 'session_id')
            
        except Exception as e:
            self.logger.error(f"Error handling test failed: {e}")
            return False
    
    def _test_reproducibility(self, config: FuzzerConfig) -> bool:
        """Test reproducibility functionality."""
        try:
            config.random_seed = 42
            
            fuzzer1 = LLMFuzzerSimulator(config)
            fuzzer2 = LLMFuzzerSimulator(config)
            
            test_doc = "def test(): return True"
            
            report1 = fuzzer1.run_fuzzing_session(test_doc, "repro_test_1")
            report2 = fuzzer2.run_fuzzing_session(test_doc, "repro_test_2")
            
            # Results should be similar (allowing for small variations)
            return (
                abs(report1.get_validity_rate() - report2.get_validity_rate()) < 0.1 and
                abs(report1.get_coverage_percentage() - report2.get_coverage_percentage()) < 5.0
            )
            
        except Exception as e:
            self.logger.error(f"Reproducibility test failed: {e}")
            return False
    
    def _run_high_load_test(self, config: FuzzerConfig) -> Dict[str, Any]:
        """Run high load stress test."""
        try:
            fuzzer = LLMFuzzerSimulator(config)
            
            # Generate many documents
            test_docs = [f"def test_{i}(): return {i % 2 == 0}" for i in range(20)]
            
            start_time = time.time()
            reports = fuzzer.run_batch_fuzzing(test_docs, parallel=True)
            execution_time = time.time() - start_time
            
            successful_sessions = len([r for r in reports if r.is_successful_session()])
            success_rate = successful_sessions / len(test_docs)
            
            return {
                'total_documents': len(test_docs),
                'execution_time': execution_time,
                'successful_sessions': successful_sessions,
                'success_rate': success_rate,
                'passed': success_rate >= 0.7  # 70% success rate under load
            }
            
        except Exception as e:
            return {'error': str(e), 'passed': False}
    
    def _run_memory_stress_test(self, config: FuzzerConfig) -> Dict[str, Any]:
        """Run memory stress test."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            fuzzer = LLMFuzzerSimulator(config)
            
            # Run many sessions to test memory management
            for i in range(10):
                test_doc = f"def stress_test_{i}(): return True"
                fuzzer.run_fuzzing_session(test_doc, f"memory_stress_{i}")
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'passed': memory_increase < 500  # Less than 500MB increase
            }
            
        except ImportError:
            return {'error': 'psutil not available', 'passed': False}
        except Exception as e:
            return {'error': str(e), 'passed': False}
    
    def _run_concurrent_sessions_test(self, config: FuzzerConfig) -> Dict[str, Any]:
        """Run concurrent sessions stress test."""
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            def run_session(session_id):
                fuzzer = LLMFuzzerSimulator(config)
                test_doc = f"def concurrent_test_{session_id}(): return True"
                return fuzzer.run_fuzzing_session(test_doc, f"concurrent_{session_id}")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(run_session, i) for i in range(10)]
                reports = [f.result() for f in futures]
            
            execution_time = time.time() - start_time
            successful_sessions = len([r for r in reports if r.is_successful_session()])
            success_rate = successful_sessions / len(reports)
            
            return {
                'total_sessions': len(reports),
                'execution_time': execution_time,
                'successful_sessions': successful_sessions,
                'success_rate': success_rate,
                'passed': success_rate >= 0.8  # 80% success rate
            }
            
        except Exception as e:
            return {'error': str(e), 'passed': False}
    
    def _run_large_document_test(self, config: FuzzerConfig) -> Dict[str, Any]:
        """Run large document stress test."""
        try:
            fuzzer = LLMFuzzerSimulator(config)
            
            # Create a large document
            large_doc = "\n".join([f"def function_{i}(param): return validate_{i}(param)" for i in range(100)])
            
            start_time = time.time()
            report = fuzzer.run_fuzzing_session(large_doc, "large_doc_test")
            execution_time = time.time() - start_time
            
            return {
                'document_size': len(large_doc),
                'execution_time': execution_time,
                'successful': report.is_successful_session(),
                'test_cases_generated': report.total_test_cases,
                'passed': report.is_successful_session() and execution_time < 120  # Less than 2 minutes
            }
            
        except Exception as e:
            return {'error': str(e), 'passed': False}
    
    def _run_extended_runtime_test(self, config: FuzzerConfig) -> Dict[str, Any]:
        """Run extended runtime stress test."""
        try:
            fuzzer = LLMFuzzerSimulator(config)
            
            start_time = time.time()
            successful_sessions = 0
            total_sessions = 0
            
            # Run sessions for a limited time to avoid long test runs
            while time.time() - start_time < 30:  # 30 seconds
                test_doc = f"def extended_test_{total_sessions}(): return True"
                report = fuzzer.run_fuzzing_session(test_doc, f"extended_{total_sessions}")
                
                if report.is_successful_session():
                    successful_sessions += 1
                
                total_sessions += 1
            
            execution_time = time.time() - start_time
            success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
            
            return {
                'execution_time': execution_time,
                'total_sessions': total_sessions,
                'successful_sessions': successful_sessions,
                'success_rate': success_rate,
                'passed': success_rate >= 0.8  # 80% success rate over time
            }
            
        except Exception as e:
            return {'error': str(e), 'passed': False}
    
    def _calculate_performance_score(self, performance_results: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        scores = []
        
        # Document processing speed
        if 'document_processing_speed' in performance_results:
            result = performance_results['document_processing_speed']
            if result.get('meets_requirement', False):
                scores.append(100)
            else:
                # Partial credit based on actual performance
                actual = result.get('lines_per_second', 0)
                target = result.get('target', 1000)
                scores.append(min(100, (actual / target) * 100))
        
        # Test generation speed
        if 'test_generation_speed' in performance_results:
            result = performance_results['test_generation_speed']
            if result.get('meets_requirement', False):
                scores.append(100)
            else:
                actual = result.get('cases_per_minute', 0)
                target = result.get('target', 50)
                scores.append(min(100, (actual / target) * 100))
        
        # Memory usage
        if 'memory_usage' in performance_results:
            result = performance_results['memory_usage']
            if result.get('meets_requirement', False):
                scores.append(100)
            else:
                actual = result.get('final_memory_mb', 0)
                target = result.get('target_mb', 2048)
                if actual > 0:
                    scores.append(max(0, 100 - ((actual - target) / target) * 100))
        
        # Concurrent execution
        if 'concurrent_execution' in performance_results:
            result = performance_results['concurrent_execution']
            if result.get('meets_requirement', False):
                scores.append(100)
            else:
                success_rate = result.get('success_rate', 0)
                scores.append(success_rate * 100)
        
        # Scalability
        if 'scalability' in performance_results:
            result = performance_results['scalability']
            if result.get('meets_requirement', False):
                scores.append(100)
            else:
                scores.append(50)  # Partial credit
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_stress_score(self, stress_results: Dict[str, Any]) -> float:
        """Calculate overall stress test score."""
        scores = []
        
        for test_name, result in stress_results.items():
            if test_name == 'score':
                continue
            
            if isinstance(result, dict) and 'passed' in result:
                scores.append(100 if result['passed'] else 0)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall validation score."""
        scores = []
        weights = {
            'integration_validation': 0.25,
            'performance_validation': 0.25,
            'functionality_validation': 0.25,
            'stress_testing': 0.15,
            'perturbation_impact_validation': 0.10
        }
        
        for category, weight in weights.items():
            if category in self.validation_results:
                result = self.validation_results[category]
                if isinstance(result, dict) and 'score' in result:
                    scores.append(result['score'] * weight)
        
        return sum(scores)
    
    def _save_validation_report(self) -> None:
        """Save validation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"system_validation_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Validation report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")


def run_system_validation(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive system validation.
    
    Args:
        output_dir: Optional output directory for validation reports
        
    Returns:
        Dictionary with validation results
    """
    validator = SystemValidator(output_dir)
    return validator.run_comprehensive_validation()


if __name__ == "__main__":
    # Run validation when script is executed directly
    results = run_system_validation()
    
    print("\n" + "=" * 80)
    print("SYSTEM VALIDATION RESULTS")
    print("=" * 80)
    print(f"Overall Score: {results.get('overall_score', 0):.1f}/100")
    print(f"Total Time: {results.get('total_validation_time', 0):.2f}s")
    
    for category, result in results.items():
        if isinstance(result, dict) and 'score' in result:
            print(f"{category.replace('_', ' ').title()}: {result['score']:.1f}/100")
    
    if results.get('overall_score', 0) >= 80:
        print("\n✅ System validation PASSED")
    else:
        print("\n❌ System validation FAILED")
    
    print("=" * 80)