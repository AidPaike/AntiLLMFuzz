"""Configuration integration for fuzzer simulator with existing system."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.fuzzer.data_models import FuzzerConfig
from src.utils.logger import get_logger


def load_integrated_fuzzer_config(
    config_file_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> FuzzerConfig:
    """Load fuzzer configuration with integration support.
    
    This function loads the fuzzer configuration from multiple sources
    with proper precedence:
    1. CLI arguments (highest priority)
    2. Configuration file
    3. Environment variables
    4. Defaults (lowest priority)
    
    Args:
        config_file_path: Optional path to configuration file
        cli_overrides: Optional CLI argument overrides
        
    Returns:
        FuzzerConfig instance with integrated settings
    """
    logger = get_logger("ConfigIntegration")
    
    # Start with default configuration
    config_dict = {}
    
    # Load from main configuration file if available
    try:
        from src.utils.config_loader import get_config_loader
        config_path = config_file_path or "config/config.yaml"
        main_config = get_config_loader(config_path).config

        
        if 'llm_fuzzer' in main_config:
            config_dict.update(_flatten_fuzzer_config(main_config['llm_fuzzer']))
            logger.debug("Loaded fuzzer config from main configuration file")
    except Exception as e:
        logger.warning(f"Failed to load main configuration: {e}")
    
    # Apply environment variable overrides
    env_overrides = _get_env_overrides()
    if env_overrides:
        config_dict.update(env_overrides)
        logger.debug(f"Applied {len(env_overrides)} environment variable overrides")
    
    # Apply CLI overrides
    if cli_overrides:
        config_dict.update(cli_overrides)
        logger.debug(f"Applied {len(cli_overrides)} CLI overrides")
    
    # Create and validate configuration
    try:
        fuzzer_config = FuzzerConfig.from_dict(config_dict)
        
        # Validate configuration
        validation = fuzzer_config.validate()
        if not validation.is_valid:
            logger.error(f"Invalid fuzzer configuration: {validation.errors}")
            raise ValueError(f"Configuration validation failed: {validation.errors}")
        
        if validation.warnings:
            for warning in validation.warnings:
                logger.warning(f"Config warning: {warning}")
        
        logger.info("Fuzzer configuration loaded and validated successfully")
        return fuzzer_config
        
    except Exception as e:
        logger.error(f"Failed to create fuzzer configuration: {e}")
        # Return default configuration as fallback
        return FuzzerConfig()


def _get_env_overrides() -> Dict[str, Any]:
    """Get configuration overrides from environment variables.
    
    Returns:
        Dictionary with environment variable overrides
    """
    overrides = {}
    
    # LLM configuration
    if os.getenv('FUZZER_LLM_MODEL'):
        overrides['llm_model'] = os.getenv('FUZZER_LLM_MODEL')

    if os.getenv('FUZZER_TARGET_MODE'):
        overrides['target_mode'] = os.getenv('FUZZER_TARGET_MODE')

    if os.getenv('FUZZER_JAVAC_HOME'):
        overrides['javac_home'] = os.getenv('FUZZER_JAVAC_HOME')

    if os.getenv('FUZZER_JAVAC_SOURCE'):
        overrides['javac_source_root'] = os.getenv('FUZZER_JAVAC_SOURCE')

    if os.getenv('FUZZER_JACOCO_CLI'):
        overrides['jacoco_cli_path'] = os.getenv('FUZZER_JACOCO_CLI')

    if os.getenv('FUZZER_JACOCO_AGENT'):
        overrides['jacoco_agent_path'] = os.getenv('FUZZER_JACOCO_AGENT')

    if os.getenv('FUZZER_COVERAGE_SCOPE'):
        overrides['coverage_scope'] = os.getenv('FUZZER_COVERAGE_SCOPE')

    
    temp_value = os.getenv('FUZZER_LLM_TEMPERATURE')
    if temp_value:
        try:
            overrides['llm_temperature'] = float(temp_value)
        except ValueError:
            pass

    max_tokens_value = os.getenv('FUZZER_LLM_MAX_TOKENS')
    if max_tokens_value:
        try:
            overrides['llm_max_tokens'] = int(max_tokens_value)
        except ValueError:
            pass

    
    # Test generation configuration
    cases_value = os.getenv('FUZZER_CASES_PER_API')
    if cases_value:
        try:
            overrides['cases_per_api'] = int(cases_value)
        except ValueError:
            pass

    
    # Execution configuration
    parallel_value = os.getenv('FUZZER_PARALLEL_EXECUTION')
    if parallel_value:
        overrides['parallel_execution'] = parallel_value.lower() in ('true', '1', 'yes')
    
    max_workers_value = os.getenv('FUZZER_MAX_WORKERS')
    if max_workers_value:
        try:
            overrides['max_workers'] = int(max_workers_value)
        except ValueError:
            pass
    
    # Reproducibility
    seed_value = os.getenv('FUZZER_RANDOM_SEED')
    if seed_value:
        try:
            overrides['random_seed'] = int(seed_value)
        except ValueError:
            pass

    
    # API configuration
    api_key_value = os.getenv('HUIYAN_API_KEY')
    if api_key_value:
        overrides['llm_api_key'] = api_key_value
        overrides['summary_api_key'] = api_key_value

    
    return overrides


def create_cli_overrides_from_args(args) -> Dict[str, Any]:
    """Create configuration overrides from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Dictionary with CLI overrides
    """
    overrides = {}
    
    # Map CLI arguments to configuration keys
    if hasattr(args, 'fuzzer_cases_per_api') and args.fuzzer_cases_per_api:
        overrides['cases_per_api'] = args.fuzzer_cases_per_api
    
    if hasattr(args, 'fuzzer_parallel') and args.fuzzer_parallel:
        overrides['parallel_execution'] = True
    
    if hasattr(args, 'fuzzer_seed') and args.fuzzer_seed is not None:
        overrides['random_seed'] = args.fuzzer_seed
    
    if hasattr(args, 'fuzzer_output_format') and args.fuzzer_output_format:
        overrides['report_format'] = args.fuzzer_output_format
    
    return overrides


def _flatten_fuzzer_config(fuzzer_config: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    llm_config = fuzzer_config.get('llm', {})
    flat.update({
        'llm_model': llm_config.get('model'),
        'llm_temperature': llm_config.get('temperature'),
        'llm_max_tokens': llm_config.get('max_tokens'),
        'llm_timeout': llm_config.get('timeout'),
        'llm_api_key': llm_config.get('api_key'),
    })

    summary_config = fuzzer_config.get('summary', {})
    flat.update({
        'summary_enabled': summary_config.get('enabled'),
        'summary_model': summary_config.get('model'),
        'summary_temperature': summary_config.get('temperature'),
        'summary_max_tokens': summary_config.get('max_tokens'),
        'summary_timeout': summary_config.get('timeout'),
        'summary_endpoint': summary_config.get('endpoint'),
        'summary_api_key': summary_config.get('api_key'),
    })

    test_gen = fuzzer_config.get('test_generation', {})
    flat.update({
        'cases_per_api': test_gen.get('cases_per_api'),
        'security_test_ratio': test_gen.get('security_test_ratio'),
        'edge_case_ratio': test_gen.get('edge_case_ratio'),
        'normal_case_ratio': test_gen.get('normal_case_ratio'),
    })

    doc_gen = fuzzer_config.get('document_generation', {})
    flat.update({
        'document_generation_enabled': doc_gen.get('enabled'),
        'document_generation_language': doc_gen.get('language'),
        'cases_per_document': doc_gen.get('cases_per_document'),
        'document_prompt_trigger': doc_gen.get('trigger'),
        'document_prompt_hint': doc_gen.get('input_hint'),
        'document_generation_case_file': doc_gen.get('case_file'),
        'document_generation_case_mode': doc_gen.get('case_mode'),
        'document_generation_max_attempts': doc_gen.get('max_attempts'),
        'document_generation_max_seconds': doc_gen.get('max_seconds'),
    })

    target_system = fuzzer_config.get('target_system', {})
    flat.update({
        'base_error_rate': target_system.get('base_error_rate'),
        'response_time_base': target_system.get('response_time_base'),
        'vulnerability_injection': target_system.get('vulnerability_injection'),
        'coverage_tracking': target_system.get('coverage_tracking'),
    })

    javac_config = fuzzer_config.get('javac', {})
    flat.update({
        'target_mode': javac_config.get('target_mode'),
        'javac_home': javac_config.get('javac_home'),
        'javac_source_root': javac_config.get('javac_source_root'),
        'jacoco_cli_path': javac_config.get('jacoco_cli_path'),
        'jacoco_agent_path': javac_config.get('jacoco_agent_path'),
        'coverage_output_dir': javac_config.get('coverage_output_dir'),
        'coverage_scope': javac_config.get('coverage_scope'),
    })

    exec_config = fuzzer_config.get('execution', {})
    flat.update({
        'timeout_per_test': exec_config.get('timeout_per_test'),
        'parallel_execution': exec_config.get('parallel_execution'),
        'max_workers': exec_config.get('max_workers'),
        'retry_failed_tests': exec_config.get('retry_failed_tests'),
    })

    repro = fuzzer_config.get('reproducibility', {})
    flat.update({
        'random_seed': repro.get('random_seed'),
    })

    output = fuzzer_config.get('output', {})
    flat.update({
        'detailed_logs': output.get('detailed_logs'),
        'save_test_cases': output.get('save_test_cases'),
        'export_coverage': output.get('export_coverage'),
        'report_format': output.get('report_format'),
    })

    return {k: v for k, v in flat.items() if v is not None}


def get_default_fuzzer_config_path() -> Optional[str]:
    """Get the default path to the fuzzer configuration file.
    
    Returns:
        Path to configuration file or None if not found
    """
    # Check common configuration file locations
    possible_paths = [
        Path("config/config.yaml"),
        Path("config/fuzzer_config.yaml"),
        Path(".kiro/config/fuzzer.yaml"),
        Path("fuzzer_config.yaml")
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def validate_integration_requirements() -> Dict[str, bool]:
    """Validate that all integration requirements are met.
    
    Returns:
        Dictionary with validation results
    """
    logger = get_logger("ConfigIntegration")
    
    validation_results = {
        'config_file_accessible': False,
        'llm_api_key_available': False,
        'fuzzer_modules_importable': False,
        'scs_modules_importable': False
    }
    
    # Check configuration file accessibility
    try:
        from src.utils.config_loader import get_config_loader
        get_config_loader()
        validation_results['config_file_accessible'] = True

    except Exception as e:
        logger.warning(f"Configuration file not accessible: {e}")
    
    # Check LLM API key availability
    try:
        from src.utils.config_loader import get_config_loader
        api_config = get_config_loader().config.get('api', {})
        endpoint = (api_config.get('endpoint') or "").lower()
        if "localhost:11434" in endpoint or "127.0.0.1:11434" in endpoint:
            validation_results['llm_api_key_available'] = True
        elif os.getenv('HUIYAN_API_KEY'):
            validation_results['llm_api_key_available'] = True
        else:
            logger.warning("HUIYAN_API_KEY environment variable not set")
    except Exception as e:
        logger.warning(f"Failed to validate LLM endpoint: {e}")
    
    # Check fuzzer modules
    try:
        from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator
        from src.fuzzer.data_models import FuzzerConfig
        validation_results['fuzzer_modules_importable'] = True
    except ImportError as e:
        logger.error(f"Fuzzer modules not importable: {e}")
    
    # Check SCS modules
    try:
        from src.scs.data_models import FeedbackData
        from src.scs.scs_calculator import SCSCalculator
        validation_results['scs_modules_importable'] = True
    except ImportError as e:
        logger.error(f"SCS modules not importable: {e}")
    
    # Log validation summary
    passed = sum(validation_results.values())
    total = len(validation_results)
    
    if passed == total:
        logger.info("✅ All integration requirements validated successfully")
    else:
        logger.warning(f"⚠️ Integration validation: {passed}/{total} checks passed")
        for check, result in validation_results.items():
            if not result:
                logger.warning(f"  - {check}: FAILED")
    
    return validation_results


def setup_integration_environment() -> bool:
    """Set up the integration environment with necessary configurations.
    
    Returns:
        True if setup successful, False otherwise
    """
    logger = get_logger("ConfigIntegration")
    
    try:
        # Validate requirements
        validation = validate_integration_requirements()
        
        # Check if critical requirements are met
        critical_requirements = [
            'fuzzer_modules_importable',
            'scs_modules_importable'
        ]
        
        for req in critical_requirements:
            if not validation[req]:
                logger.error(f"Critical requirement not met: {req}")
                return False
        
        # Warn about optional requirements
        if not validation['llm_api_key_available']:
            logger.warning("LLM API key not available - ensure local endpoint supports no auth")
        
        if not validation['config_file_accessible']:
            logger.warning("Configuration file not accessible - using defaults")
        
        logger.info("Integration environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup integration environment: {e}")
        return False
