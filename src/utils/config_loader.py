"""Configuration loader for hyperparameters from YAML file."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from src.scs.data_models import SCSConfig
from src.fuzzer.data_models import FuzzerConfig


class ConfigLoader:
    """Load and manage configuration from YAML file."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize config loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_prioritization_config(self) -> Dict[str, Any]:
        """Get token prioritization configuration.
        
        Returns:
            Dictionary with prioritization settings
        """
        return self.config.get('prioritization', {})
    
    def get_scs_config(self) -> SCSConfig:
        """Get SCS configuration as SCSConfig object.
        
        Returns:
            SCSConfig instance with loaded parameters
        """
        scs_config = self.config.get('scs', {})
        weights = scs_config.get('weights', {})
        baseline = scs_config.get('baseline', {})
        simulation = scs_config.get('simulation', {})
        
        return SCSConfig(
            validity_weight=weights.get('validity', 0.40),
            coverage_weight=weights.get('coverage', 0.35),
            defect_weight=weights.get('defect', 0.25),
            baseline_validity=baseline.get('validity', 0.85),
            baseline_coverage=baseline.get('coverage', 65.0),
            baseline_defects=baseline.get('defects', 10),
            variance=simulation.get('variance', 0.1)
        )
    
    def get_hotspot_config(self) -> Dict[str, Any]:
        """Get hotspot analysis configuration.
        
        Returns:
            Dictionary with hotspot settings
        """
        return self.config.get('hotspot', {})
    
    def get_perturbation_config(self) -> Dict[str, Any]:
        """Get perturbation configuration.
        
        Returns:
            Dictionary with perturbation settings
        """
        return self.config.get('perturbation', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration.
        
        Returns:
            Dictionary with output settings
        """
        return self.config.get('output', {})

    def get_tsd_config(self) -> Dict[str, Any]:
        """Get Tokenization-Level Semantic Drift (TSD) configuration."""
        return self.config.get('tsd', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Dictionary with logging settings
        """
        return self.config.get('logging', {})
    
    def get_base_scores(self) -> Dict[str, float]:
        """Get base scores for token types.
        
        Returns:
            Dictionary mapping token types to base scores
        """
        prioritization = self.get_prioritization_config()
        return prioritization.get('base_scores', {
            'phrase': 3.0,
            'noun': 2.0,
            'verb': 1.5,
            'default': 1.0
        })
    
    def get_bonus_scores(self) -> Dict[str, float]:
        """Get bonus scores for different contexts.
        
        Returns:
            Dictionary mapping context types to bonus scores
        """
        prioritization = self.get_prioritization_config()
        return prioritization.get('bonus_scores', {
            'security_related': 5.0,
            'validation_related': 4.0,
            'boundary_check': 3.0
        })
    
    def get_security_keywords(self) -> list:
        """Get security-related keywords.
        
        Returns:
            List of security keywords
        """
        prioritization = self.get_prioritization_config()
        return prioritization.get('security_keywords', [])
    
    def get_validation_keywords(self) -> list:
        """Get validation-related keywords.
        
        Returns:
            List of validation keywords
        """
        prioritization = self.get_prioritization_config()
        return prioritization.get('validation_keywords', [])
    
    def get_boundary_keywords(self) -> list:
        """Get boundary check keywords.
        
        Returns:
            List of boundary keywords
        """
        prioritization = self.get_prioritization_config()
        return prioritization.get('boundary_keywords', [])
    
    def get_impact_factors(self) -> Dict[str, Dict[str, float]]:
        """Get impact factor ranges for different token types.
        
        Returns:
            Dictionary with impact factor ranges
        """
        scs_config = self.config.get('scs', {})
        return scs_config.get('impact_factors', {})
    
    def get_visualization_thresholds(self) -> Dict[str, float]:
        """Get visualization thresholds for hotspot display.
        
        Returns:
            Dictionary with threshold values
        """
        hotspot = self.get_hotspot_config()
        return hotspot.get('visualization', {
            'low_threshold': 25.0,
            'medium_threshold': 50.0,
            'high_threshold': 75.0
        })
    
    def get_top_k(self) -> int:
        """Get number of top hotspots to select.
        
        Returns:
            Top K value
        """
        hotspot = self.get_hotspot_config()
        return hotspot.get('top_k', 10)
    
    def get_default_top_n(self) -> int:
        """Get default number of top tokens to perturb.
        
        Returns:
            Default top N value
        """
        perturbation = self.get_perturbation_config()
        return perturbation.get('default_top_n', 5)
    
    def get_application_config(self) -> Dict[str, Any]:
        """Get application configuration.
        
        Returns:
            Dictionary with application settings
        """
        return self.config.get('application', {})
    
    def get_app_version(self) -> str:
        """Get application version.
        
        Returns:
            Application version string
        """
        app_config = self.get_application_config()
        return app_config.get('version', 'LLM Fuzzer Semantic Disruptor v1.0.0')
    
    def get_default_input_file(self) -> str:
        """Get default input file path.
        
        Returns:
            Default input file path
        """
        app_config = self.get_application_config()
        return app_config.get('default_input_file', 'data/00java_std.md')
    
    def get_default_output_dir(self) -> str:
        """Get default output directory.
        
        Returns:
            Default output directory path
        """
        app_config = self.get_application_config()
        return app_config.get('default_output_dir', 'output')
    
    def get_default_strategy(self) -> str:
        """Get default perturbation strategy.
        
        Returns:
            Default strategy name
        """
        app_config = self.get_application_config()
        return app_config.get('default_strategy', 'tokenization_drift')
    
    def get_default_log_level(self) -> str:
        """Get default log level.
        
        Returns:
            Default log level
        """
        app_config = self.get_application_config()
        return app_config.get('default_log_level', 'INFO')
    
    def get_max_token_display_length(self) -> int:
        """Get maximum token display length for logging.
        
        Returns:
            Maximum token display length
        """
        app_config = self.get_application_config()
        return app_config.get('max_token_display_length', 50)
    
    def get_priority_management_config(self) -> Dict[str, Any]:
        """Get dynamic priority management configuration.
        
        Returns:
            Dictionary with priority management settings
        """
        return self.config.get('priority_management', {})
    
    def get_feedback_analysis_config(self) -> Dict[str, Any]:
        """Get feedback analysis configuration.
        
        Returns:
            Dictionary with feedback analysis settings
        """
        return self.config.get('feedback_analysis', {})
    
    def get_fuzzer_config(self) -> FuzzerConfig:
        """Get LLM fuzzer configuration as FuzzerConfig object.
        
        Returns:
            FuzzerConfig instance with loaded parameters
        """
        fuzzer_config = self.config.get('llm_fuzzer', {})
        
        # Extract nested configurations
        llm_config = fuzzer_config.get('llm', {})
        test_gen_config = fuzzer_config.get('test_generation', {})
        target_config = fuzzer_config.get('target_system', {})
        exec_config = fuzzer_config.get('execution', {})
        repro_config = fuzzer_config.get('reproducibility', {})
        output_config = fuzzer_config.get('output', {})
        
        # Handle environment variable substitution for API key
        api_key = llm_config.get('api_key', '')
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]  # Remove ${ and }
            api_key = os.getenv(env_var)
        
        return FuzzerConfig(
            # LLM Configuration
            llm_model=llm_config.get('model', 'gpt-4'),
            llm_temperature=llm_config.get('temperature', 0.7),
            llm_max_tokens=llm_config.get('max_tokens', 2000),
            llm_timeout=llm_config.get('timeout', 30),
            llm_api_key=api_key,
            
            # Test Generation
            cases_per_api=test_gen_config.get('cases_per_api', 20),
            security_test_ratio=test_gen_config.get('security_test_ratio', 0.3),
            edge_case_ratio=test_gen_config.get('edge_case_ratio', 0.2),
            normal_case_ratio=test_gen_config.get('normal_case_ratio', 0.5),
            
            # Target System Simulation
            base_error_rate=target_config.get('base_error_rate', 0.05),
            response_time_base=target_config.get('response_time_base', 0.1),
            vulnerability_injection=target_config.get('vulnerability_injection', True),
            coverage_tracking=target_config.get('coverage_tracking', True),
            
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
    
    def validate_fuzzer_config(self) -> Dict[str, Any]:
        """Validate fuzzer configuration and return validation results.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        fuzzer_config = self.config.get('llm_fuzzer', {})
        
        if not fuzzer_config:
            errors.append("No llm_fuzzer configuration found")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Validate LLM configuration
        llm_config = fuzzer_config.get('llm', {})
        if not llm_config.get('model'):
            errors.append("LLM model not specified")
        
        api_key = llm_config.get('api_key', '')
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            if not os.getenv(env_var):
                warnings.append(f"Environment variable {env_var} not set for API key")
        elif not api_key:
            warnings.append("No API key specified for LLM")
        
        # Validate test generation ratios
        test_gen = fuzzer_config.get('test_generation', {})
        ratios = [
            test_gen.get('security_test_ratio', 0.3),
            test_gen.get('edge_case_ratio', 0.2),
            test_gen.get('normal_case_ratio', 0.5)
        ]
        
        if abs(sum(ratios) - 1.0) > 0.01:  # Allow small floating point errors
            warnings.append("Test generation ratios don't sum to 1.0")
        
        # Validate execution parameters
        exec_config = fuzzer_config.get('execution', {})
        max_workers = exec_config.get('max_workers', 4)
        if max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        timeout = exec_config.get('timeout_per_test', 10.0)
        if timeout <= 0:
            errors.append("timeout_per_test must be positive")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: str = "config/config.yaml") -> ConfigLoader:
    """Get global config loader instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def reload_config():
    """Reload global configuration."""
    global _config_loader
    if _config_loader is not None:
        _config_loader.reload()
