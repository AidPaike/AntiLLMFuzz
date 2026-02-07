"""Configuration manager for LLM fuzzer simulator."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.fuzzer.data_models import FuzzerConfig, ValidationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FuzzerConfigManager:
    """Manages configuration for LLM fuzzer simulator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self._config_cache: Optional[FuzzerConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def load_config(self, reload: bool = False) -> FuzzerConfig:
        """Load fuzzer configuration from file.
        
        Args:
            reload: Force reload from file even if cached
            
        Returns:
            FuzzerConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
            ValueError: If configuration is invalid
        """
        if self._config_cache is not None and not reload:
            return self._config_cache
        
        logger.info(f"Loading fuzzer configuration from {self.config_path}")
        
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using default configuration.")
            self._config_cache = FuzzerConfig().get_default_config()
            return self._config_cache
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}

            # Extract fuzzer configuration
            raw_config: Dict[str, Any] = self._raw_config or {}
            fuzzer_config_dict = raw_config.get('llm_fuzzer', {})


            
            if not fuzzer_config_dict:
                logger.warning("No llm_fuzzer section found in config. Using default configuration.")
                self._config_cache = FuzzerConfig().get_default_config()
                return self._config_cache
            
            # Parse configuration with environment variable substitution
            self._config_cache = self._parse_fuzzer_config(fuzzer_config_dict)
            
            # Validate configuration
            validation = self._config_cache.validate()
            if not validation.is_valid:
                error_msg = f"Invalid configuration: {'; '.join(validation.errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if validation.warnings:
                for warning in validation.warnings:
                    logger.warning(f"Configuration warning: {warning}")
            
            logger.info("Fuzzer configuration loaded successfully")
            return self._config_cache
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _parse_fuzzer_config(self, config_dict: Dict[str, Any]) -> FuzzerConfig:
        """Parse fuzzer configuration from dictionary with environment variable substitution.
        
        Args:
            config_dict: Raw configuration dictionary
            
        Returns:
            FuzzerConfig instance
        """
        # Extract nested configurations
        llm_config = config_dict.get('llm') or {}
        summary_config = config_dict.get('summary') or {}
        test_gen_config = config_dict.get('test_generation') or {}
        doc_gen_config = config_dict.get('document_generation') or {}
        target_config = config_dict.get('target_system') or {}
        javac_config = config_dict.get('javac') or {}
        exec_config = config_dict.get('execution') or {}
        repro_config = config_dict.get('reproducibility') or {}
        output_config = config_dict.get('output') or {}

        # Handle environment variable substitution for API key
        api_key = self._substitute_env_vars(llm_config.get('api_key', ''))
        summary_key = self._substitute_env_vars(summary_config.get('api_key', ''))


        return FuzzerConfig(
            # LLM Configuration
            llm_model=llm_config.get('model', 'gpt-4'),

            llm_temperature=llm_config.get('temperature', 0.7),
            llm_max_tokens=llm_config.get('max_tokens', 2000),
            llm_timeout=llm_config.get('timeout', 30),
            llm_api_key=api_key,

            summary_enabled=summary_config.get('enabled', False),
            summary_model=summary_config.get('model', 'gpt-4o'),
            summary_temperature=summary_config.get('temperature', 0.2),
            summary_max_tokens=summary_config.get('max_tokens', 800),
            summary_timeout=summary_config.get('timeout', 60),
            summary_api_key=summary_key,
            summary_endpoint=summary_config.get('endpoint'),
            
            # Test Generation
            cases_per_api=test_gen_config.get('cases_per_api', 20),
            security_test_ratio=test_gen_config.get('security_test_ratio', 0.3),
            edge_case_ratio=test_gen_config.get('edge_case_ratio', 0.2),
            normal_case_ratio=test_gen_config.get('normal_case_ratio', 0.5),

            document_generation_enabled=doc_gen_config.get('enabled', False),
            document_generation_language=doc_gen_config.get('language', 'java'),
            cases_per_document=doc_gen_config.get('cases_per_document', 20),
            document_prompt_trigger=doc_gen_config.get(
                'trigger',
                '/* Please create a very short program which uses new Java features in a complex way */'
            ),
            document_prompt_hint=doc_gen_config.get('input_hint', 'import java.lang.Object;'),
            document_generation_case_file=doc_gen_config.get('case_file'),
            document_generation_case_mode=doc_gen_config.get('case_mode', 'fixed'),
            document_generation_max_attempts=doc_gen_config.get('max_attempts', 60),
            document_generation_max_seconds=doc_gen_config.get('max_seconds', 300),
            
            # Target System Simulation
            base_error_rate=target_config.get('base_error_rate', 0.05),
            response_time_base=target_config.get('response_time_base', 0.1),
            vulnerability_injection=target_config.get('vulnerability_injection', True),
            coverage_tracking=target_config.get('coverage_tracking', True),

            # Real target execution (javac)
            target_mode=javac_config.get('target_mode', 'simulated'),
            javac_home=javac_config.get('javac_home'),
            javac_source_root=javac_config.get('javac_source_root'),
            jacoco_cli_path=javac_config.get('jacoco_cli_path'),
            jacoco_agent_path=javac_config.get('jacoco_agent_path'),
            coverage_output_dir=javac_config.get('coverage_output_dir', 'coverage'),
            coverage_scope=javac_config.get('coverage_scope', 'javac'),

            # Execution Parameters
            timeout_per_test=exec_config.get('timeout_per_test', 10.0),
            parallel_execution=exec_config.get('parallel_execution', True),
            max_workers=exec_config.get('max_workers', 4),
            retry_failed_tests=exec_config.get('retry_failed_tests', True),

            
            # Reproducibility
            random_seed=repro_config.get('random_seed'),
            
            # Output Configuration
            detailed_logs=output_config.get('detailed_logs', True),
            save_test_cases=output_config.get('save_test_cases', True),
            export_coverage=output_config.get('export_coverage', True),
            report_format=output_config.get('report_format', 'json')
        )
    
    def _substitute_env_vars(self, value: str) -> Optional[str]:
        """Substitute environment variables in configuration values.
        
        Args:
            value: Configuration value that may contain environment variables
            
        Returns:
            Value with environment variables substituted, or None if not found
        """
        if not value:
            return None
        
        if value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]  # Remove ${ and }
            env_value = os.getenv(env_var)
            if env_value is None:
                logger.warning(f"Environment variable {env_var} not found")
            return env_value
        
        return value
    
    def save_config(self, config: FuzzerConfig, config_path: Optional[str] = None) -> None:
        """Save fuzzer configuration to file.
        
        Args:
            config: FuzzerConfig to save
            config_path: Optional path to save to. If None, uses current config path.
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        # Validate configuration before saving
        validation = config.validate()
        if not validation.is_valid:
            error_msg = f"Cannot save invalid configuration: {'; '.join(validation.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert to dictionary format
        config_dict = {
            'llm_fuzzer': {
                'llm': {
                    'model': config.llm_model,
                    'temperature': config.llm_temperature,
                    'max_tokens': config.llm_max_tokens,
                    'timeout': config.llm_timeout,
                    'api_key': config.llm_api_key or "${HUIYAN_API_KEY}"
                },
                'summary': {
                    'enabled': config.summary_enabled,
                    'model': config.summary_model,
                    'temperature': config.summary_temperature,
                    'max_tokens': config.summary_max_tokens,
                    'timeout': config.summary_timeout,
                    'endpoint': config.summary_endpoint,
                    'api_key': config.summary_api_key or "${HUIYAN_API_KEY}"
                },
                'test_generation': {
                    'cases_per_api': config.cases_per_api,
                    'security_test_ratio': config.security_test_ratio,
                    'edge_case_ratio': config.edge_case_ratio,
                    'normal_case_ratio': config.normal_case_ratio
                },
                'document_generation': {
                    'enabled': config.document_generation_enabled,
                    'language': config.document_generation_language,
                    'cases_per_document': config.cases_per_document,
                    'trigger': config.document_prompt_trigger,
                    'input_hint': config.document_prompt_hint,
                    'case_file': config.document_generation_case_file,
                    'case_mode': config.document_generation_case_mode,
                    'max_attempts': config.document_generation_max_attempts,
                    'max_seconds': config.document_generation_max_seconds,
                },
                'target_system': {
                    'base_error_rate': config.base_error_rate,
                    'response_time_base': config.response_time_base,
                    'vulnerability_injection': config.vulnerability_injection,
                    'coverage_tracking': config.coverage_tracking
                },
                'execution': {
                    'timeout_per_test': config.timeout_per_test,
                    'parallel_execution': config.parallel_execution,
                    'max_workers': config.max_workers,
                    'retry_failed_tests': config.retry_failed_tests
                },
                'reproducibility': {
                    'random_seed': config.random_seed
                },
                'output': {
                    'detailed_logs': config.detailed_logs,
                    'save_test_cases': config.save_test_cases,
                    'export_coverage': config.export_coverage,
                    'report_format': config.report_format
                }
            }
        }
        
        # If we have existing raw config, merge with it to preserve other sections
        if self._raw_config:
            merged_config = self._raw_config.copy()
            merged_config.update(config_dict)
            config_dict = merged_config
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Fuzzer configuration saved to {save_path}")
        
        # Update cache
        self._config_cache = config
    
    def get_config(self) -> FuzzerConfig:
        """Get current configuration, loading if necessary.
        
        Returns:
            FuzzerConfig instance
        """
        if self._config_cache is None:
            return self.load_config()
        return self._config_cache
    
    def update_config(self, **kwargs) -> FuzzerConfig:
        """Update configuration with new values.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            Updated FuzzerConfig instance
        """
        current_config = self.get_config()
        
        # Create new config with updated values
        config_dict = {
            'llm_model': kwargs.get('llm_model', current_config.llm_model),
            'llm_temperature': kwargs.get('llm_temperature', current_config.llm_temperature),
            'llm_max_tokens': kwargs.get('llm_max_tokens', current_config.llm_max_tokens),
            'llm_timeout': kwargs.get('llm_timeout', current_config.llm_timeout),
            'llm_api_key': kwargs.get('llm_api_key', current_config.llm_api_key),
            'cases_per_api': kwargs.get('cases_per_api', current_config.cases_per_api),
            'security_test_ratio': kwargs.get('security_test_ratio', current_config.security_test_ratio),
            'edge_case_ratio': kwargs.get('edge_case_ratio', current_config.edge_case_ratio),
            'normal_case_ratio': kwargs.get('normal_case_ratio', current_config.normal_case_ratio),
            'base_error_rate': kwargs.get('base_error_rate', current_config.base_error_rate),
            'response_time_base': kwargs.get('response_time_base', current_config.response_time_base),
            'vulnerability_injection': kwargs.get('vulnerability_injection', current_config.vulnerability_injection),
            'coverage_tracking': kwargs.get('coverage_tracking', current_config.coverage_tracking),
            'timeout_per_test': kwargs.get('timeout_per_test', current_config.timeout_per_test),
            'parallel_execution': kwargs.get('parallel_execution', current_config.parallel_execution),
            'max_workers': kwargs.get('max_workers', current_config.max_workers),
            'retry_failed_tests': kwargs.get('retry_failed_tests', current_config.retry_failed_tests),
            'random_seed': kwargs.get('random_seed', current_config.random_seed),
            'detailed_logs': kwargs.get('detailed_logs', current_config.detailed_logs),
            'save_test_cases': kwargs.get('save_test_cases', current_config.save_test_cases),
            'export_coverage': kwargs.get('export_coverage', current_config.export_coverage),
            'report_format': kwargs.get('report_format', current_config.report_format)
        }
        
        updated_config = FuzzerConfig(**config_dict)
        self._config_cache = updated_config
        
        return updated_config
    
    def validate_current_config(self) -> ValidationResult:
        """Validate current configuration.
        
        Returns:
            ValidationResult with validation status
        """
        config = self.get_config()
        return config.validate()


# Global configuration manager instance
_config_manager: Optional[FuzzerConfigManager] = None


def get_fuzzer_config_manager(config_path: Optional[str] = None) -> FuzzerConfigManager:
    """Get global fuzzer configuration manager instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        FuzzerConfigManager instance
    """
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = FuzzerConfigManager(config_path)
    return _config_manager


def get_fuzzer_config(config_path: Optional[str] = None) -> FuzzerConfig:
    """Get fuzzer configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        FuzzerConfig instance
    """
    manager = get_fuzzer_config_manager(config_path)
    return manager.get_config()
