"""Command-line interface for anti_llm4fuzz tool."""

import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from src.extractors import DocumentationTokenExtractor, JavaTokenExtractor, PythonTokenExtractor
from src.token_prioritizer import TokenPrioritizer
from src.strategies.semantic import (
    TokenizationDriftStrategy,
    LexicalDisguiseStrategy,
    DataFlowMisdirectionStrategy,
    ControlFlowMisdirectionStrategy,
    DocumentationDeceptionStrategy,
    CognitiveManipulationStrategy
)
from src.strategies.generic import (
    FormattingNoiseStrategy,
    StructuralNoiseStrategy,
    ParaphrasingStrategy,
    CognitiveLoadStrategy
)
from src.strategies.selector import (
    filter_strategies,
    infer_target_from_extractor_language,
)
from src.utils import (
    get_logger,
    create_output_directory,
    generate_output_filename,
    read_file,
    write_file
)
from src.scs import SCSConfig, FeedbackSimulator, SCSCalculator, HotspotAnalyzer
from src.fuzzer import LLMFuzzerSimulator, FuzzerConfig
from src.fuzzer.integration import PerturbationFuzzerIntegrator, LLMFeedbackSimulator
from src.fuzzer.config_integration import (
    load_integrated_fuzzer_config, 
    create_cli_overrides_from_args,
    setup_integration_environment
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="anti_llm4fuzz",
        description="LLM Fuzzer Semantic Disruptor - Disrupt LLM-based fuzzers with semantic perturbations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and perturb documentation with tokenization drift
  anti_llm4fuzz --input data/doc.md --top-n 5 --strategy tokenization_drift
  
  # Apply all strategies to Java file
  anti_llm4fuzz --input src/Main.java --top-n 3 --strategy all
  
  # Apply only semantic strategies
  anti_llm4fuzz --input data/doc.md --strategy semantic
  
  # Process with cognitive manipulation and debug logging
  anti_llm4fuzz --input script.py --strategy cognitive_manipulation --log-level DEBUG
  
  # Quick test with default settings
  anti_llm4fuzz -i myfile.md
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file path (documentation, Java, or Python)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Base output directory (default: output)'
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=5,
        help='Number of top tokens to perturb (default: 5)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=[
            'tokenization_drift', 'lexical_disguise', 'dataflow_misdirection',
            'controlflow_misdirection', 'documentation_deception', 'cognitive_manipulation',
            'formatting_noise', 'structural_noise', 'paraphrasing', 'cognitive_load',
            'semantic', 'generic', 'all', 'all_composed'
        ],
        default='tokenization_drift',
        help='Perturbation strategy to use (default: tokenization_drift)'
    )

    parser.add_argument(
        '--allow-unsafe-code-strategies',
        action='store_true',
        help='Allow risky/unsafe strategies on source code inputs (may break compilation/execution)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    # SCS (Semantic Contribution Score) options
    parser.add_argument(
        '--enable-scs',
        action='store_true',
        help='Enable SCS (Semantic Contribution Score) calculation'
    )
    
    parser.add_argument(
        '--scs-validity-weight',
        type=float,
        default=0.40,
        help='Weight for validity metric in SCS calculation (default: 0.40)'
    )
    
    parser.add_argument(
        '--scs-coverage-weight',
        type=float,
        default=0.35,
        help='Weight for coverage metric in SCS calculation (default: 0.35)'
    )
    
    parser.add_argument(
        '--scs-defect-weight',
        type=float,
        default=0.25,
        help='Weight for defect detection metric in SCS calculation (default: 0.25)'
    )
    
    parser.add_argument(
        '--scs-baseline-validity',
        type=float,
        default=0.85,
        help='Baseline validity rate for SCS comparison (default: 0.85)'
    )
    
    parser.add_argument(
        '--scs-baseline-coverage',
        type=float,
        default=65.0,
        help='Baseline coverage percentage for SCS comparison (default: 65.0)'
    )
    
    parser.add_argument(
        '--scs-baseline-defects',
        type=int,
        default=10,
        help='Baseline defect count for SCS comparison (default: 10)'
    )
    
    parser.add_argument(
        '--use-llm-fuzzer',
        action='store_true',
        help='Use LLM-based fuzzer simulator for more realistic feedback (default: simple simulator)'
    )
    
    parser.add_argument(
        '--enable-fuzzer-integration',
        action='store_true',
        help='Enable full fuzzer integration with perturbation impact analysis'
    )

    parser.add_argument(
        '--iterative-disruption',
        action='store_true',
        help='Run iterative disruption workflow: select best segment, apply strategies in loops'
    )
    
    # Fuzzer simulator specific options
    parser.add_argument(
        '--fuzzer-mode',
        action='store_true',
        help='Run in fuzzer simulator mode instead of perturbation mode'
    )
    
    parser.add_argument(
        '--fuzzer-config',
        type=str,
        help='Path to fuzzer configuration file (YAML/JSON)'
    )
    
    parser.add_argument(
        '--fuzzer-cases-per-api',
        type=int,
        default=20,
        help='Number of test cases to generate per API (default: 20)'
    )
    
    parser.add_argument(
        '--fuzzer-parallel',
        action='store_true',
        help='Enable parallel test execution in fuzzer mode'
    )
    
    parser.add_argument(
        '--fuzzer-seed',
        type=int,
        help='Random seed for deterministic fuzzer behavior'
    )
    
    parser.add_argument(
        '--fuzzer-compare',
        action='store_true',
        help='Compare fuzzing results between original and perturbed documents'
    )
    
    parser.add_argument(
        '--fuzzer-batch',
        action='store_true',
        help='Run batch fuzzing on multiple documents'
    )
    
    parser.add_argument(
        '--fuzzer-output-format',
        type=str,
        choices=['json', 'csv', 'yaml'],
        default='json',
        help='Output format for fuzzer reports (default: json)'
    )

    parser.add_argument(
        '--fuzzer-target-mode',
        type=str,
        choices=['simulated', 'javac'],
        default='simulated',
        help='Target system mode (simulated or javac)'
    )

    parser.add_argument(
        '--fuzzer-javac-home',
        type=str,
        help='JDK home containing bin/javac and bin/java'
    )

    parser.add_argument(
        '--fuzzer-javac-source',
        type=str,
        help='OpenJDK source root for coverage mapping'
    )

    parser.add_argument(
        '--fuzzer-jacoco-cli',
        type=str,
        help='Path to jacoco-cli.jar'
    )

    parser.add_argument(
        '--fuzzer-jacoco-agent',
        type=str,
        help='Path to jacocoagent.jar'
    )

    parser.add_argument(
        '--fuzzer-coverage-scope',
        type=str,
        choices=['javac', 'all'],
        default='javac',
        help='Coverage scope (javac or all modules)'
    )

    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='anti_llm4fuzz v1.0.0'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser.parse_args()


def select_extractor(file_path: str):
    """Select appropriate extractor based on file extension.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Appropriate extractor instance or None
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    # Try each extractor
    try:
        if extension == '.java':
            extractor = JavaTokenExtractor()
            if extractor.can_extract(file_path):
                return extractor
    except RuntimeError:
        pass  # javalang not installed
    
    try:
        if extension in ['.py', '.pyw']:
            extractor = PythonTokenExtractor()
            if extractor.can_extract(file_path):
                return extractor
    except RuntimeError:
        pass
    
    # Try documentation extractor as fallback
    try:
        extractor = DocumentationTokenExtractor()
        if extractor.can_extract(file_path):
            return extractor
    except RuntimeError:
        pass
    
    return None


def run_fuzzer_mode(args, logger):
    """Run the tool in fuzzer simulator mode."""
    import json
    import yaml
    from pathlib import Path
    
    if not args.quiet:
        logger.info("Running in LLM Fuzzer Simulator mode")
        logger.info("=" * 80)
    
    # Validate input file
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Load fuzzer configuration
    fuzzer_config = FuzzerConfig()
    
    if args.fuzzer_config:
        config_path = Path(args.fuzzer_config)
        if not config_path.exists():
            logger.error(f"Fuzzer config file not found: {config_path}")
            return 1
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            # Extract fuzzer config if nested
            if 'llm_fuzzer' in config_dict:
                config_dict = config_dict['llm_fuzzer']
            
            fuzzer_config = FuzzerConfig.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load fuzzer config: {e}")
            return 1
    
    # Override config with CLI arguments
    if args.fuzzer_cases_per_api:
        fuzzer_config.cases_per_api = args.fuzzer_cases_per_api
    
    if args.fuzzer_parallel:
        fuzzer_config.parallel_execution = True
    
    if args.fuzzer_seed is not None:
        fuzzer_config.random_seed = args.fuzzer_seed
    
    if args.fuzzer_output_format:
        fuzzer_config.report_format = args.fuzzer_output_format
    
    # Validate configuration
    config_validation = fuzzer_config.validate()
    if not config_validation.is_valid:
        logger.error(f"Invalid fuzzer configuration: {config_validation.errors}")
        return 1
    
    if config_validation.warnings and not args.quiet:
        for warning in config_validation.warnings:
            logger.warning(f"Config warning: {warning}")
    
    # Initialize fuzzer simulator
    try:
        fuzzer = LLMFuzzerSimulator(fuzzer_config)
        if not args.quiet:
            logger.info(f"✓ Fuzzer simulator initialized with seed: {fuzzer.get_random_seed()}")
    except Exception as e:
        logger.error(f"Failed to initialize fuzzer simulator: {e}")
        return 1
    
    # Read document content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            document_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return 1
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_output_directory(args.output_dir, "fuzzer_results")
    
    if not args.quiet:
        logger.info(f"Output directory: {output_dir}")
        logger.info("")
    
    try:
        if args.fuzzer_compare:
            # Comparison mode: run on original and perturbed documents
            return run_fuzzer_comparison_mode(
                args, logger, fuzzer, document_content, input_file, output_dir, timestamp
            )
        elif args.fuzzer_batch:
            # Batch mode: process multiple documents
            return run_fuzzer_batch_mode(
                args, logger, fuzzer, input_file, output_dir, timestamp
            )
        else:
            # Single document mode
            return run_fuzzer_single_mode(
                args, logger, fuzzer, document_content, input_file, output_dir, timestamp
            )
    
    except KeyboardInterrupt:
        logger.info("Fuzzer execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fuzzer execution failed: {e}")
        return 1


def run_fuzzer_single_mode(args, logger, fuzzer, document_content, input_file, output_dir, timestamp):
    """Run fuzzer on a single document."""
    if not args.quiet:
        logger.info("Running fuzzer on single document...")
    
    # Progress callback
    def progress_callback(message, progress):
        if not args.quiet:
            logger.info(f"[{progress*100:.1f}%] {message}")
    
    # Run fuzzing session
    session_id = f"single_{timestamp}"
    report = fuzzer.run_fuzzing_session(
        document_content=document_content,
        session_id=session_id,
        document_path=str(input_file),
        progress_callback=progress_callback if not args.quiet else None
    )
    
    # Save report
    report_filename = f"fuzzer_report_{timestamp}.{args.fuzzer_output_format}"
    report_path = output_dir / report_filename
    
    try:
        report_data = report.to_dict()
        
        if args.fuzzer_output_format == 'json':
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        elif args.fuzzer_output_format == 'yaml':
            with open(report_path, 'w', encoding='utf-8') as f:
                yaml.dump(report_data, f, default_flow_style=False, allow_unicode=True)
        elif args.fuzzer_output_format == 'csv':
            # For CSV, create a simplified flat structure
            import csv
            with open(report_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Session ID', report.session_id])
                writer.writerow(['Total Test Cases', report.total_test_cases])
                writer.writerow(['Validity Rate', f"{report.get_validity_rate():.3f}"])
                writer.writerow(['Coverage Percentage', f"{report.get_coverage_percentage():.1f}"])
                writer.writerow(['Defects Found', report.get_defect_count()])
                writer.writerow(['Execution Time (s)', f"{report.performance_metrics.total_execution_time:.2f}"])
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return 1
    
    # Display results
    if not args.quiet:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Fuzzer Results:")
        logger.info("=" * 80)
        logger.info(f"Session ID: {report.session_id}")
        logger.info(f"Test Cases Generated: {report.total_test_cases}")
        logger.info(f"Validity Rate: {report.get_validity_rate():.2%}")
        logger.info(f"Coverage: {report.get_coverage_percentage():.1f}%")
        logger.info(f"Defects Found: {report.get_defect_count()}")
        logger.info(f"Execution Time: {report.performance_metrics.total_execution_time:.2f}s")
        logger.info(f"Report saved: {report_path}")
        logger.info("=" * 80)
    else:
        # In quiet mode, just print the report path
        print(report_path)
    
    return 0


def run_fuzzer_comparison_mode(args, logger, fuzzer, original_content, input_file, output_dir, timestamp):
    """Run fuzzer comparison between original and perturbed documents."""
    if not args.quiet:
        logger.info("Running fuzzer comparison mode...")
        logger.info("This will generate perturbations and compare fuzzing results")
    
    # First, generate perturbations using the existing system
    if not args.quiet:
        logger.info("Step 1: Generating perturbations...")
    
    # Use existing perturbation logic (simplified)
    extractor = select_extractor(str(input_file))
    if extractor is None:
        logger.error("No suitable extractor found for comparison mode")
        return 1
    
    # Extract and prioritize tokens
    tokens = extractor.extract_tokens(str(input_file))
    if not tokens:
        logger.error("No tokens extracted for perturbation")
        return 1
    
    from src.token_prioritizer import TokenPrioritizer
    prioritizer = TokenPrioritizer()
    tokens = prioritizer.assign_scores(tokens)
    ranked_tokens = prioritizer.rank_tokens(tokens)
    
    # Generate a few perturbations for comparison (pick safe default for code inputs)
    content_target = infer_target_from_extractor_language(extractor.language)
    if content_target == "documentation":
        from src.strategies.semantic import TokenizationDriftStrategy
        strategy = TokenizationDriftStrategy()
    else:
        from src.strategies.generic import FormattingNoiseStrategy
        strategy = FormattingNoiseStrategy()
     
    perturbed_versions = strategy.apply_multiple(
        ranked_tokens[:min(3, len(ranked_tokens))],  # Use top 3 tokens
        original_content,
        max_tokens=3,
        target=content_target,
        language=extractor.language,
        preserve_executability=(content_target == "code"),
        allow_unsafe_code=args.allow_unsafe_code_strategies,
     )
    
    perturbed_documents = list(perturbed_versions.values())
    
    if not args.quiet:
        logger.info(f"Generated {len(perturbed_documents)} perturbed versions")
        logger.info("Step 2: Running fuzzer comparison...")
    
    # Progress callback
    def progress_callback(message, progress):
        if not args.quiet:
            logger.info(f"[{progress*100:.1f}%] {message}")
    
    # Run comparison
    comparison_report = fuzzer.compare_fuzzing_results(
        original_document=original_content,
        perturbed_documents=perturbed_documents,
        progress_callback=progress_callback if not args.quiet else None
    )
    
    # Save comparison report
    comparison_filename = f"fuzzer_comparison_{timestamp}.{args.fuzzer_output_format}"
    comparison_path = output_dir / comparison_filename
    
    try:
        comparison_data = comparison_report.to_dict()
        
        if args.fuzzer_output_format == 'json':
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        elif args.fuzzer_output_format == 'yaml':
            with open(comparison_path, 'w', encoding='utf-8') as f:
                yaml.dump(comparison_data, f, default_flow_style=False, allow_unicode=True)
        
    except Exception as e:
        logger.error(f"Failed to save comparison report: {e}")
        return 1
    
    # Display results
    if not args.quiet:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Fuzzer Comparison Results:")
        logger.info("=" * 80)
        logger.info(f"Original Validity: {comparison_report.original_report.get_validity_rate():.2%}")
        logger.info(f"Original Coverage: {comparison_report.original_report.get_coverage_percentage():.1f}%")
        logger.info(f"Original Defects: {comparison_report.original_report.get_defect_count()}")
        logger.info("")
        logger.info(f"Validity Impact: {comparison_report.validity_impact:.2%}")
        logger.info(f"Coverage Impact: {comparison_report.coverage_impact:.2%}")
        logger.info(f"Defect Impact: {comparison_report.defect_impact:.2%}")
        logger.info(f"Overall Impact: {comparison_report.overall_impact:.2%}")
        logger.info(f"Statistical Significance: {comparison_report.statistical_significance}")
        logger.info(f"Comparison report saved: {comparison_path}")
        logger.info("=" * 80)
    
    return 0


def run_fuzzer_batch_mode(args, logger, fuzzer, input_file, output_dir, timestamp):
    """Run fuzzer in batch mode on multiple documents."""
    if not args.quiet:
        logger.info("Running fuzzer in batch mode...")
    
    # For batch mode, treat input as a directory or pattern
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file - treat as list of file paths
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                file_paths = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to read batch file list: {e}")
            return 1
    elif input_path.is_dir():
        # Directory - find all supported files
        file_paths = []
        for ext in ['.md', '.txt', '.rst', '.adoc', '.java', '.py']:
            file_paths.extend(str(p) for p in input_path.glob(f"*{ext}"))
    else:
        logger.error(f"Invalid input for batch mode: {input_path}")
        return 1
    
    if not file_paths:
        logger.error("No files found for batch processing")
        return 1
    
    if not args.quiet:
        logger.info(f"Found {len(file_paths)} files for batch processing")
    
    # Read all documents
    documents = []
    valid_paths = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append(content)
            valid_paths.append(file_path)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
    
    if not documents:
        logger.error("No valid documents found for batch processing")
        return 1
    
    # Progress callback
    def progress_callback(message, progress):
        if not args.quiet:
            logger.info(f"[{progress*100:.1f}%] {message}")
    
    # Run batch fuzzing
    reports = fuzzer.run_batch_fuzzing(
        documents=documents,
        session_prefix=f"batch_{timestamp}",
        parallel=args.fuzzer_parallel,
        progress_callback=progress_callback if not args.quiet else None
    )
    
    # Generate aggregated report
    aggregated_report = fuzzer.generate_aggregated_report(
        reports=reports,
        report_name=f"batch_aggregated_{timestamp}"
    )
    
    # Save individual reports and aggregated report
    reports_dir = output_dir / "individual_reports"
    reports_dir.mkdir(exist_ok=True)
    
    for i, (report, file_path) in enumerate(zip(reports, valid_paths)):
        report_filename = f"report_{i+1:03d}_{Path(file_path).stem}.{args.fuzzer_output_format}"
        report_path = reports_dir / report_filename
        
        try:
            report_data = report.to_dict()
            report_data['source_file'] = file_path
            
            if args.fuzzer_output_format == 'json':
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Failed to save individual report {i+1}: {e}")
    
    # Save aggregated report
    aggregated_filename = f"batch_aggregated_{timestamp}.{args.fuzzer_output_format}"
    aggregated_path = output_dir / aggregated_filename
    
    try:
        if args.fuzzer_output_format == 'json':
            with open(aggregated_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_report, f, indent=2, ensure_ascii=False, default=str)
        elif args.fuzzer_output_format == 'yaml':
            with open(aggregated_path, 'w', encoding='utf-8') as f:
                yaml.dump(aggregated_report, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error(f"Failed to save aggregated report: {e}")
        return 1
    
    # Display results
    if not args.quiet:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Batch Fuzzer Results:")
        logger.info("=" * 80)
        logger.info(f"Documents Processed: {aggregated_report['session_count']}")
        logger.info(f"Total Test Cases: {aggregated_report['total_test_cases']}")
        logger.info(f"Average Validity Rate: {aggregated_report['average_validity_rate']:.2%}")
        logger.info(f"Average Coverage: {aggregated_report['average_coverage_percentage']:.1f}%")
        logger.info(f"Total Defects Found: {aggregated_report['total_defects_found']}")
        logger.info(f"Success Rate: {aggregated_report['success_rate']:.2%}")
        logger.info(f"Total Execution Time: {aggregated_report['total_execution_time']:.2f}s")
        logger.info(f"Throughput: {aggregated_report['throughput_tests_per_second']:.1f} tests/sec")
        logger.info("")
        logger.info(f"Individual reports: {reports_dir}")
        logger.info(f"Aggregated report: {aggregated_path}")
        logger.info("=" * 80)
    
    return 0


def main():
    """Run the CLI tool."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Adjust log level if quiet mode
    if args.quiet:
        log_level = 'ERROR'
    else:
        log_level = args.log_level
    
    # Initialize logger
    logger = get_logger(level=log_level)
    
    if not args.quiet:
        logger.info("=" * 80)
        logger.info("anti_llm4fuzz - LLM Fuzzer Semantic Disruptor")
        logger.info("=" * 80)
    
    # Handle fuzzer mode
    if args.fuzzer_mode:
        return run_fuzzer_mode(args, logger)

    # Handle iterative disruption mode
    if args.iterative_disruption:
        if not args.quiet:
            logger.info("Running iterative disruption workflow...")

        try:
            # Setup integration environment
            if not setup_integration_environment():
                raise RuntimeError("Integration environment setup failed")

            # Load integrated fuzzer configuration
            cli_overrides = create_cli_overrides_from_args(args)
            fuzzer_config = load_integrated_fuzzer_config(cli_overrides=cli_overrides)

            integrator = PerturbationFuzzerIntegrator(fuzzer_config)

            # Run iterative disruption workflow
            results = integrator.create_iterative_disruption_workflow(
                input_file_path=str(args.input),
                top_n_candidates=args.top_n,
                output_dir=str(args.output_dir)
            )

            if 'error' in results:
                logger.error(f"Iterative disruption failed: {results['error']}")
                return 1
            else:
                logger.info(f"Iterative disruption completed in {results['workflow_metadata']['execution_time']:.2f}s")
                logger.info(f"Best segment: {results['segment_selection']['best_token']['text']}")
                logger.info(f"Final validity: {results['final_metrics']['validity_rate']:.3f}")
                logger.info(f"Final coverage: {results['final_metrics']['coverage_percentage']:.1f}%")
                logger.info(f"Final crashes: {results['final_metrics']['crash_count']}")

            return 0

        except Exception as e:
            logger.error(f"Iterative disruption failed: {e}")
            return 1

    # Validate input file
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    if not args.quiet:
        logger.info(f"Input: {input_file}")
        logger.info(f"Top-N: {args.top_n}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info("")
    
    # Step 1: Select and initialize extractor
    if not args.quiet:
        logger.info("Step 1: Selecting token extractor...")
    
    extractor = select_extractor(str(input_file))
    
    if extractor is None:
        logger.error(f"No suitable extractor found for: {input_file}")
        logger.error("Supported: .md, .txt, .rst, .adoc, .java, .py")
        logger.error("Note: Java requires 'javalang' package")
        return 1
    
    if not args.quiet:
        logger.info(f"  ✓ Using {extractor.language} extractor")
    
    # Step 2: Extract tokens
    if not args.quiet:
        logger.info("")
        logger.info("Step 2: Extracting tokens...")
    
    try:
        tokens = extractor.extract_tokens(str(input_file))
        if not tokens:
            logger.warning("No tokens extracted")
            return 1
        if not args.quiet:
            logger.info(f"  ✓ Extracted {len(tokens)} tokens")
    except Exception as e:
        logger.error(f"Failed to extract tokens: {e}")
        return 1
    
    # Step 3: Prioritize tokens
    if not args.quiet:
        logger.info("")
        logger.info("Step 3: Prioritizing tokens...")
    
    prioritizer = TokenPrioritizer()
    tokens = prioritizer.assign_scores(tokens)
    ranked_tokens = prioritizer.rank_tokens(tokens)
    
    if not args.quiet:
        logger.info(f"  ✓ Ranked {len(ranked_tokens)} tokens")
    
    # Step 3.5: Calculate SCS scores (if enabled)
    scs_config = None
    if args.enable_scs:
        if not args.quiet:
            logger.info("")
            logger.info("Step 3.5: Calculating SCS scores...")
        
        try:
            # Create SCS configuration
            scs_config = SCSConfig(
                validity_weight=args.scs_validity_weight,
                coverage_weight=args.scs_coverage_weight,
                defect_weight=args.scs_defect_weight,
                baseline_validity=args.scs_baseline_validity,
                baseline_coverage=args.scs_baseline_coverage,
                baseline_defects=args.scs_baseline_defects
            )
            
            # Validate configuration
            scs_config.validate()
            
            # Initialize simulator and calculator
            if args.use_llm_fuzzer:
                # Use LLM-based simulator for more realistic feedback
                try:
                    # Setup integration environment
                    if not setup_integration_environment():
                        logger.warning("Integration environment setup failed, falling back to simple simulator")
                        simulator = FeedbackSimulator.from_config(scs_config)
                    else:
                        # Load integrated fuzzer configuration
                        cli_overrides = create_cli_overrides_from_args(args)
                        fuzzer_config = load_integrated_fuzzer_config(cli_overrides=cli_overrides)
                        fuzzer_config.target_mode = args.fuzzer_target_mode
                        fuzzer_config.javac_home = args.fuzzer_javac_home
                        fuzzer_config.javac_source_root = args.fuzzer_javac_source
                        fuzzer_config.jacoco_cli_path = args.fuzzer_jacoco_cli
                        fuzzer_config.jacoco_agent_path = args.fuzzer_jacoco_agent
                        fuzzer_config.coverage_scope = args.fuzzer_coverage_scope

                        
                        simulator = LLMFeedbackSimulator(fuzzer_config)
                        logger.info("Using LLM-based fuzzer simulator for SCS calculation")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM fuzzer simulator: {e}")
                    logger.info("Falling back to simple feedback simulator")
                    simulator = FeedbackSimulator.from_config(scs_config)
            else:
                # Use simple feedback simulator (legacy)
                simulator = FeedbackSimulator.from_config(scs_config)
                logger.info("Using simple feedback simulator for SCS calculation")
            calculator = SCSCalculator.from_config(scs_config)
            baseline_feedback = scs_config.get_baseline_feedback()
            original_content = read_file(args.input)
            if original_content is None:
                logger.error("Failed to read input content for SCS calculation")
                return 1
            
            # Simulate feedback and calculate SCS for each token
            for token in ranked_tokens:
                # Simulate perturbed feedback
                if args.use_llm_fuzzer:
                    # For LLM simulator, we need the document content
                    perturbed_feedback = simulator.simulate_feedback(token, original_content)
                else:
                    # Legacy simulator doesn't need document content
                    perturbed_feedback = simulator.simulate_feedback(token, original_content)

                
                # Calculate SCS score
                scs_score = calculator.calculate_scs(baseline_feedback, perturbed_feedback)
                
                # Update token
                token.scs_score = scs_score
            
            # Re-rank by SCS score
            ranked_tokens = sorted(ranked_tokens, key=lambda t: (t.scs_score, t.priority_score), reverse=True)
            
            if not args.quiet:
                logger.info(f"  ✓ Calculated SCS scores for {len(ranked_tokens)} tokens")
                
        except ValueError as e:
            logger.error(f"SCS configuration error: {e}")
            return 1
    
    if not args.quiet:
        logger.info("")
        if args.enable_scs:
            logger.info(f"Top {min(args.top_n, 5)} tokens (by SCS):")
            for i, token in enumerate(ranked_tokens[:min(args.top_n, 5)], 1):
                text = token.text[:40] + "..." if len(token.text) > 40 else token.text
                text = text.replace('\n', ' ')
                logger.info(f"  {i}. [SCS: {token.scs_score:.1f}] [Priority: {token.priority_score:.1f}] {text}")
        else:
            logger.info(f"Top {min(args.top_n, 5)} tokens:")
            for i, token in enumerate(ranked_tokens[:min(args.top_n, 5)], 1):
                text = token.text[:40] + "..." if len(token.text) > 40 else token.text
                text = text.replace('\n', ' ')
                logger.info(f"  {i}. [{token.priority_score:.1f}] {text}")
    
    # Step 4: Initialize strategies
    if not args.quiet:
        logger.info("")
        logger.info("Step 4: Initializing strategies...")
    
    # Define all available strategies
    semantic_strategies = [
        TokenizationDriftStrategy(),
        LexicalDisguiseStrategy(),
        DataFlowMisdirectionStrategy(),
        ControlFlowMisdirectionStrategy(),
        DocumentationDeceptionStrategy(),
        CognitiveManipulationStrategy()
    ]
    
    generic_strategies = [
        FormattingNoiseStrategy(),
        StructuralNoiseStrategy(),
        ParaphrasingStrategy(),
        CognitiveLoadStrategy()
    ]
    
    # Select strategies based on argument
    strategies = []
    if args.strategy == 'all':
        strategies = semantic_strategies + generic_strategies
    elif args.strategy == 'all_composed':
        strategies = semantic_strategies + generic_strategies
    elif args.strategy == 'semantic':
        strategies = semantic_strategies
    elif args.strategy == 'generic':
        strategies = generic_strategies
    elif args.strategy == 'tokenization_drift':
        strategies = [TokenizationDriftStrategy()]
    elif args.strategy == 'lexical_disguise':
        strategies = [LexicalDisguiseStrategy()]
    elif args.strategy == 'dataflow_misdirection':
        strategies = [DataFlowMisdirectionStrategy()]
    elif args.strategy == 'controlflow_misdirection':
        strategies = [ControlFlowMisdirectionStrategy()]
    elif args.strategy == 'documentation_deception':
        strategies = [DocumentationDeceptionStrategy()]
    elif args.strategy == 'cognitive_manipulation':
        strategies = [CognitiveManipulationStrategy()]
    elif args.strategy == 'formatting_noise':
        strategies = [FormattingNoiseStrategy()]
    elif args.strategy == 'structural_noise':
        strategies = [StructuralNoiseStrategy()]
    elif args.strategy == 'paraphrasing':
        strategies = [ParaphrasingStrategy()]
    elif args.strategy == 'cognitive_load':
        strategies = [CognitiveLoadStrategy()]

    # Filter strategies based on target (documentation vs executable code)
    content_target = infer_target_from_extractor_language(extractor.language)
    strategies, skipped = filter_strategies(
        list(strategies),
        target=content_target,
        language=extractor.language,
        allow_unsafe_code=args.allow_unsafe_code_strategies,
    )

    if skipped and not args.quiet:
        skipped_names = ", ".join(sorted({s.name for s in skipped}))
        logger.warning(f"Skipped strategies for {content_target}/{extractor.language}: {skipped_names}")

    if not strategies:
        logger.error(
            f"No applicable strategies for {content_target}/{extractor.language} "
            f"(requested: {args.strategy})."
        )
        if content_target == "code" and not args.allow_unsafe_code_strategies:
            logger.error("Tip: use --allow-unsafe-code-strategies to override, or pick a code-safe strategy.")
        return 1
    
    if not args.quiet:
        for strategy in strategies:
            logger.info(f"  ✓ {strategy.name} ({strategy.category})")
    
    # Step 5: Read content
    original_content = read_file(str(input_file))
    if original_content is None:
        logger.error("Failed to read input file")
        return 1
    
    # Step 6: Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_output_directory(args.output_dir, "perturbations")
    
    if not args.quiet:
        logger.info("")
        logger.info(f"Step 5: Output directory: {output_dir}")
    
    # Step 7: Apply perturbations
    if not args.quiet:
        logger.info("")
        logger.info(f"Step 6: Applying perturbations...")
    
    base_name = input_file.stem
    all_output_files = []
    
    for strategy in strategies:
        perturbed_versions = strategy.apply_multiple(
            ranked_tokens, 
            original_content, 
            max_tokens=args.top_n,
            target=content_target,
            language=extractor.language,
            preserve_executability=(content_target == "code"),
            allow_unsafe_code=args.allow_unsafe_code_strategies,
        )
        
        for variant_name, perturbed_content in perturbed_versions.items():
            output_filename = generate_output_filename(
                base_name, 
                variant_name, 
                timestamp,
                extension=input_file.suffix
            )
            output_path = output_dir / output_filename
            
            if write_file(str(output_path), perturbed_content):
                all_output_files.append(output_filename)
        
        if not args.quiet:
            logger.info(f"  ✓ {strategy.name}: {len(perturbed_versions)} files")
    
    # Step 7.5: Fuzzer Integration (if enabled)
    fuzzer_results = None
    if args.enable_fuzzer_integration:
        if not args.quiet:
            logger.info("")
            logger.info("Step 7.5: Running fuzzer integration analysis...")
        
        try:
            # Setup integration environment
            if not setup_integration_environment():
                raise RuntimeError("Integration environment setup failed")
            
            # Load integrated fuzzer configuration
            cli_overrides = create_cli_overrides_from_args(args)
            fuzzer_config = load_integrated_fuzzer_config(cli_overrides=cli_overrides)
            
            integrator = PerturbationFuzzerIntegrator(fuzzer_config)
            
            # Run end-to-end workflow
            fuzzer_results = integrator.create_end_to_end_workflow(
                input_file_path=str(input_file),
                perturbation_strategy=args.strategy,
                top_n_tokens=args.top_n,
                output_dir=str(output_dir)
            )
            
            if not args.quiet:
                if 'error' in fuzzer_results:
                    logger.error(f"Fuzzer integration failed: {fuzzer_results['error']}")
                else:
                    logger.info(f"  ✓ Fuzzer integration completed in {fuzzer_results['workflow_metadata']['execution_time']:.2f}s")
                    logger.info(f"  ✓ Average validity impact: {fuzzer_results['summary']['average_validity_impact']:.3f}")
                    logger.info(f"  ✓ Average coverage impact: {fuzzer_results['summary']['average_coverage_impact']:.1f}%")
        
        except Exception as e:
            logger.error(f"Fuzzer integration failed: {e}")
            fuzzer_results = {'error': str(e)}

    # Step 8: Generate metadata
    stats = extractor.get_statistics(tokens)
    
    metadata = {
        "input_file": str(input_file),
        "timestamp": timestamp,
        "extractor": extractor.language,
        "configuration": {
            "top_n_tokens": args.top_n,
            "strategies": [s.name for s in strategies],
            "fuzzer_integration_enabled": args.enable_fuzzer_integration,
            "use_llm_fuzzer": args.use_llm_fuzzer
        },
        "statistics": {
            "total_tokens": stats['total_tokens'],
            "tokens_by_type": stats['by_type'],
            "security_related": sum(1 for t in tokens if prioritizer.is_security_related(t)),
            "output_files": len(all_output_files)
        },
        "output_files": all_output_files
    }
    
    # Add fuzzer results to metadata if available
    if fuzzer_results:
        metadata["fuzzer_integration"] = fuzzer_results
    
    # Add SCS data if enabled
    if args.enable_scs and scs_config:
        metadata["scs_enabled"] = True
        metadata["scs_config"] = {
            "validity_weight": scs_config.validity_weight,
            "coverage_weight": scs_config.coverage_weight,
            "defect_weight": scs_config.defect_weight,
            "baseline_validity": scs_config.baseline_validity,
            "baseline_coverage": scs_config.baseline_coverage,
            "baseline_defects": scs_config.baseline_defects
        }
        
        # Add SCS statistics
        analyzer = HotspotAnalyzer(ranked_tokens)
        scs_stats = analyzer.get_statistics()
        metadata["scs_statistics"] = {
            "mean": round(scs_stats['mean'], 2),
            "median": round(scs_stats['median'], 2),
            "max": round(scs_stats['max'], 2),
            "min": round(scs_stats['min'], 2),
            "std_dev": round(scs_stats['std_dev'], 2)
        }
        
        # Add top tokens with SCS scores
        top_tokens = analyzer.get_top_n(min(args.top_n, 10))
        metadata["top_tokens_by_scs"] = [
            {
                "text": t.text[:50],
                "line": t.line,
                "type": t.token_type,
                "priority_score": round(t.priority_score, 2),
                "scs_score": round(t.scs_score, 2)
            }
            for t in top_tokens
        ]
    else:
        metadata["scs_enabled"] = False
    
    metadata_path = output_dir / f"metadata_{timestamp}.json"
    write_file(str(metadata_path), json.dumps(metadata, indent=2, ensure_ascii=False))
    
    # Summary
    if not args.quiet:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info("=" * 80)
        logger.info(f"Tokens extracted: {stats['total_tokens']}")
        logger.info(f"Security-related: {metadata['statistics']['security_related']}")
        logger.info(f"Output files: {len(all_output_files)}")
        logger.info(f"Output directory: {output_dir}")
        
        # Add SCS summary if enabled
        if args.enable_scs and scs_config:
            logger.info("")
            logger.info("SCS Statistics:")
            logger.info("-" * 80)
            logger.info(f"Mean SCS:   {metadata['scs_statistics']['mean']:.2f}")
            logger.info(f"Median SCS: {metadata['scs_statistics']['median']:.2f}")
            logger.info(f"Max SCS:    {metadata['scs_statistics']['max']:.2f}")
            logger.info(f"Min SCS:    {metadata['scs_statistics']['min']:.2f}")
            
            # Count tokens in different SCS ranges
            high_scs = len([t for t in ranked_tokens if t.scs_score > 70])
            medium_scs = len([t for t in ranked_tokens if 40 <= t.scs_score <= 70])
            low_scs = len([t for t in ranked_tokens if t.scs_score < 40])
            
            logger.info("")
            logger.info("SCS Distribution:")
            logger.info(f"  High (>70):     {high_scs}")
            logger.info(f"  Medium (40-70): {medium_scs}")
            logger.info(f"  Low (<40):      {low_scs}")
        
        # Add fuzzer integration summary if enabled
        if args.enable_fuzzer_integration and isinstance(fuzzer_results, dict) and 'error' not in fuzzer_results:
            logger.info("")
            logger.info("Fuzzer Integration Results:")
            logger.info("-" * 80)
            fuzzer_results_dict = fuzzer_results if isinstance(fuzzer_results, dict) else {}
            summary = fuzzer_results_dict.get('summary', {})
            if not isinstance(summary, dict):
                summary = {}
            workflow_meta = fuzzer_results_dict.get('workflow_metadata', {})
            if not isinstance(workflow_meta, dict):
                workflow_meta = {}
            logger.info(f"Total Perturbations: {summary.get('total_perturbations', 0)}")
            logger.info(f"Avg Validity Impact: {summary.get('average_validity_impact', 0.0):.3f}")
            logger.info(f"Avg Coverage Impact: {summary.get('average_coverage_impact', 0.0):.1f}%")
            logger.info(f"Workflow Time: {workflow_meta.get('execution_time', 0.0):.2f}s")


            
            if isinstance(fuzzer_results_dict, dict) and 'output_file' in fuzzer_results_dict:
                logger.info(f"Detailed Results: {fuzzer_results_dict['output_file']}")
        
        logger.info("")
        logger.info("✓ Completed successfully!")
        logger.info("=" * 80)
    else:
        # In quiet mode, just print the output directory
        print(output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
