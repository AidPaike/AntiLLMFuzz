"""Experiment metadata tracking for reproducible fuzzing sessions."""

import json
import hashlib
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from src.fuzzer.data_models import FuzzerConfig
from src.fuzzer.random_seed_manager import RandomSeedManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    platform: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    architecture: str = field(default_factory=lambda: platform.architecture()[0])
    processor: str = field(default_factory=lambda: platform.processor())
    hostname: str = field(default_factory=lambda: platform.node())


@dataclass
class DocumentInfo:
    """Information about the input document."""
    path: str
    size_bytes: int
    hash_md5: str
    hash_sha256: str
    last_modified: datetime
    encoding: str = "utf-8"
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DocumentInfo':
        """Create DocumentInfo from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentInfo instance
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        # Read file content for hashing
        with open(path, 'rb') as f:
            content = f.read()
        
        # Calculate hashes
        md5_hash = hashlib.md5(content).hexdigest()
        sha256_hash = hashlib.sha256(content).hexdigest()
        
        # Get file stats
        stat = path.stat()
        
        return cls(
            path=str(path.absolute()),
            size_bytes=stat.st_size,
            hash_md5=md5_hash,
            hash_sha256=sha256_hash,
            last_modified=datetime.fromtimestamp(stat.st_mtime)
        )


@dataclass
class DependencyInfo:
    """Information about software dependencies."""
    package_name: str
    version: str
    
    @classmethod
    def get_installed_packages(cls) -> List['DependencyInfo']:
        """Get information about installed packages.
        
        Returns:
            List of DependencyInfo for key packages
        """
        dependencies = []
        
        # Key packages for reproducibility
        key_packages = [
            'openai', 'anthropic', 'requests', 'numpy', 'pyyaml', 
            'spacy', 'javalang', 'pytest', 'hypothesis'
        ]
        
        for package in key_packages:
            try:
                import importlib.metadata
                version = importlib.metadata.version(package)
                dependencies.append(cls(package_name=package, version=version))
            except importlib.metadata.PackageNotFoundError:
                # Package not installed, skip
                continue
            except Exception as e:
                logger.warning(f"Could not get version for {package}: {e}")
        
        return dependencies


@dataclass
class ExperimentMetadata:
    """Comprehensive metadata for experiment reproduction."""
    
    # Experiment identification (required fields first)
    experiment_id: str
    session_id: str
    fuzzer_config: FuzzerConfig
    random_seed_state: Dict[str, Any]
    document_info: DocumentInfo
    
    # Optional fields with defaults
    created_at: datetime = field(default_factory=datetime.now)
    system_info: SystemInfo = field(default_factory=SystemInfo)
    dependencies: List[DependencyInfo] = field(default_factory=DependencyInfo.get_installed_packages)
    
    # Execution information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Results summary
    total_test_cases: int = 0
    validity_rate: float = 0.0
    coverage_percentage: float = 0.0
    defects_found: int = 0
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    version: str = "1.0"
    
    @classmethod
    def create_for_session(cls, session_id: str, document_path: str, 
                          config: FuzzerConfig, seed_manager: RandomSeedManager,
                          experiment_id: Optional[str] = None) -> 'ExperimentMetadata':
        """Create metadata for a new fuzzing session.
        
        Args:
            session_id: Unique session identifier
            document_path: Path to input document
            config: Fuzzer configuration
            seed_manager: Random seed manager
            experiment_id: Optional experiment ID. If None, generates one.
            
        Returns:
            ExperimentMetadata instance
        """
        if experiment_id is None:
            # Generate experiment ID from session and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"exp_{session_id}_{timestamp}"
        
        document_info = DocumentInfo.from_file(document_path)
        
        return cls(
            experiment_id=experiment_id,
            session_id=session_id,
            fuzzer_config=config,
            random_seed_state=seed_manager.export_state(),
            document_info=document_info
        )
    
    def start_execution(self) -> None:
        """Mark the start of experiment execution."""
        self.start_time = datetime.now()
        logger.info(f"Started experiment {self.experiment_id} at {self.start_time}")
    
    def end_execution(self, total_test_cases: int = 0, validity_rate: float = 0.0,
                     coverage_percentage: float = 0.0, defects_found: int = 0) -> None:
        """Mark the end of experiment execution and record results.
        
        Args:
            total_test_cases: Total number of test cases generated
            validity_rate: Validity rate of test cases
            coverage_percentage: Code coverage achieved
            defects_found: Number of defects found
        """
        self.end_time = datetime.now()
        
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        self.total_test_cases = total_test_cases
        self.validity_rate = validity_rate
        self.coverage_percentage = coverage_percentage
        self.defects_found = defects_found
        
        logger.info(f"Completed experiment {self.experiment_id} in {self.duration_seconds:.2f}s")
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_tags(self, tags: List[str]) -> None:
        """Add multiple tags to the experiment.
        
        Args:
            tags: List of tags to add
        """
        for tag in tags:
            self.add_tag(tag)
    
    def set_notes(self, notes: str) -> None:
        """Set experiment notes.
        
        Args:
            notes: Notes about the experiment
        """
        self.notes = notes
    
    def get_reproduction_info(self) -> Dict[str, Any]:
        """Get information needed to reproduce this experiment.
        
        Returns:
            Dictionary with reproduction information
        """
        return {
            'experiment_id': self.experiment_id,
            'fuzzer_config': asdict(self.fuzzer_config),
            'random_seed_state': self.random_seed_state,
            'document_info': asdict(self.document_info),
            'system_info': asdict(self.system_info),
            'dependencies': [asdict(dep) for dep in self.dependencies],
            'created_at': self.created_at.isoformat(),
            'version': self.version
        }
    
    def export_to_file(self, file_path: str) -> None:
        """Export metadata to a JSON file.
        
        Args:
            file_path: Path to save the metadata file
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary with proper serialization
        metadata_dict = self._to_serializable_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported experiment metadata to {output_path}")
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary.
        
        Returns:
            Dictionary that can be serialized to JSON
        """
        def serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None
        
        return {
            'experiment_id': self.experiment_id,
            'session_id': self.session_id,
            'created_at': serialize_datetime(self.created_at),
            'fuzzer_config': asdict(self.fuzzer_config),
            'random_seed_state': self.random_seed_state,
            'document_info': {
                **asdict(self.document_info),
                'last_modified': serialize_datetime(self.document_info.last_modified)
            },
            'system_info': asdict(self.system_info),
            'dependencies': [asdict(dep) for dep in self.dependencies],
            'start_time': serialize_datetime(self.start_time),
            'end_time': serialize_datetime(self.end_time),
            'duration_seconds': self.duration_seconds,
            'total_test_cases': self.total_test_cases,
            'validity_rate': self.validity_rate,
            'coverage_percentage': self.coverage_percentage,
            'defects_found': self.defects_found,
            'tags': self.tags,
            'notes': self.notes,
            'version': self.version
        }
    
    @classmethod
    def import_from_file(cls, file_path: str) -> 'ExperimentMetadata':
        """Import metadata from a JSON file.
        
        Args:
            file_path: Path to the metadata file
            
        Returns:
            ExperimentMetadata instance
        """
        input_path = Path(file_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        return cls._from_dict(metadata_dict)
    
    @classmethod
    def _from_dict(cls, metadata_dict: Dict[str, Any]) -> 'ExperimentMetadata':
        """Create ExperimentMetadata from dictionary.
        
        Args:
            metadata_dict: Dictionary with metadata
            
        Returns:
            ExperimentMetadata instance
        """
        def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(dt_str) if dt_str else None
        
        # Parse fuzzer config
        fuzzer_config = FuzzerConfig.from_dict(metadata_dict['fuzzer_config'])
        
        # Parse document info
        doc_info_dict = metadata_dict['document_info']
        document_info = DocumentInfo(
            path=doc_info_dict['path'],
            size_bytes=doc_info_dict['size_bytes'],
            hash_md5=doc_info_dict['hash_md5'],
            hash_sha256=doc_info_dict['hash_sha256'],
            last_modified=parse_datetime(doc_info_dict['last_modified']),
            encoding=doc_info_dict.get('encoding', 'utf-8')
        )
        
        # Parse system info
        system_info_dict = metadata_dict['system_info']
        system_info = SystemInfo(**system_info_dict)
        
        # Parse dependencies
        dependencies = [
            DependencyInfo(**dep_dict) 
            for dep_dict in metadata_dict['dependencies']
        ]
        
        return cls(
            experiment_id=metadata_dict['experiment_id'],
            session_id=metadata_dict['session_id'],
            created_at=parse_datetime(metadata_dict['created_at']),
            fuzzer_config=fuzzer_config,
            random_seed_state=metadata_dict['random_seed_state'],
            document_info=document_info,
            system_info=system_info,
            dependencies=dependencies,
            start_time=parse_datetime(metadata_dict.get('start_time')),
            end_time=parse_datetime(metadata_dict.get('end_time')),
            duration_seconds=metadata_dict.get('duration_seconds'),
            total_test_cases=metadata_dict.get('total_test_cases', 0),
            validity_rate=metadata_dict.get('validity_rate', 0.0),
            coverage_percentage=metadata_dict.get('coverage_percentage', 0.0),
            defects_found=metadata_dict.get('defects_found', 0),
            tags=metadata_dict.get('tags', []),
            notes=metadata_dict.get('notes', ''),
            version=metadata_dict.get('version', '1.0')
        )
    
    def validate_reproducibility(self, other_document_path: str) -> Dict[str, bool]:
        """Validate if experiment can be reproduced with given document.
        
        Args:
            other_document_path: Path to document to validate against
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'document_exists': False,
            'document_hash_matches': False,
            'document_size_matches': False,
            'system_compatible': True,
            'dependencies_available': True
        }
        
        try:
            # Check if document exists
            other_path = Path(other_document_path)
            results['document_exists'] = other_path.exists()
            
            if results['document_exists']:
                # Check document hash and size
                other_doc_info = DocumentInfo.from_file(other_document_path)
                results['document_hash_matches'] = (
                    other_doc_info.hash_sha256 == self.document_info.hash_sha256
                )
                results['document_size_matches'] = (
                    other_doc_info.size_bytes == self.document_info.size_bytes
                )
            
            # Check system compatibility (basic check)
            current_system = SystemInfo()
            if current_system.platform != self.system_info.platform:
                results['system_compatible'] = False
                logger.warning(f"Platform mismatch: current={current_system.platform}, "
                             f"original={self.system_info.platform}")
            
            # Check dependencies (basic availability check)
            current_deps = DependencyInfo.get_installed_packages()
            current_dep_names = {dep.package_name for dep in current_deps}
            original_dep_names = {dep.package_name for dep in self.dependencies}
            
            missing_deps = original_dep_names - current_dep_names
            if missing_deps:
                results['dependencies_available'] = False
                logger.warning(f"Missing dependencies: {missing_deps}")
        
        except Exception as e:
            logger.error(f"Error validating reproducibility: {e}")
            results['system_compatible'] = False
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment.
        
        Returns:
            Dictionary with experiment summary
        """
        return {
            'experiment_id': self.experiment_id,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'duration_seconds': self.duration_seconds,
            'document_path': self.document_info.path,
            'document_size_mb': round(self.document_info.size_bytes / 1024 / 1024, 2),
            'master_seed': self.random_seed_state.get('master_seed'),
            'total_test_cases': self.total_test_cases,
            'validity_rate': round(self.validity_rate, 3),
            'coverage_percentage': round(self.coverage_percentage, 1),
            'defects_found': self.defects_found,
            'tags': self.tags,
            'system_platform': self.system_info.platform,
            'python_version': self.system_info.python_version
        }


class ExperimentMetadataManager:
    """Manages experiment metadata for multiple sessions."""
    
    def __init__(self, metadata_dir: str = "output/metadata"):
        """Initialize metadata manager.
        
        Args:
            metadata_dir: Directory to store metadata files
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metadata(self, metadata: ExperimentMetadata) -> str:
        """Save experiment metadata to file.
        
        Args:
            metadata: ExperimentMetadata to save
            
        Returns:
            Path to saved metadata file
        """
        filename = f"metadata_{metadata.experiment_id}.json"
        file_path = self.metadata_dir / filename
        
        metadata.export_to_file(str(file_path))
        return str(file_path)
    
    def load_metadata(self, experiment_id: str) -> ExperimentMetadata:
        """Load experiment metadata by ID.
        
        Args:
            experiment_id: Experiment ID to load
            
        Returns:
            ExperimentMetadata instance
        """
        filename = f"metadata_{experiment_id}.json"
        file_path = self.metadata_dir / filename
        
        return ExperimentMetadata.import_from_file(str(file_path))
    
    def list_experiments(self) -> List[str]:
        """List all available experiment IDs.
        
        Returns:
            List of experiment IDs
        """
        experiment_ids = []
        
        for file_path in self.metadata_dir.glob("metadata_*.json"):
            # Extract experiment ID from filename
            filename = file_path.stem
            if filename.startswith("metadata_"):
                experiment_id = filename[9:]  # Remove "metadata_" prefix
                experiment_ids.append(experiment_id)
        
        return sorted(experiment_ids)
    
    def get_experiment_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all experiments.
        
        Returns:
            List of experiment summary dictionaries
        """
        summaries = []
        
        for experiment_id in self.list_experiments():
            try:
                metadata = self.load_metadata(experiment_id)
                summaries.append(metadata.get_summary())
            except Exception as e:
                logger.error(f"Error loading metadata for {experiment_id}: {e}")
        
        return summaries
    
    def create_reproduction_package(self, experiment_id: str, output_dir: str) -> str:
        """Create a reproduction package for an experiment.
        
        Args:
            experiment_id: Experiment ID to package
            output_dir: Directory to create package in
            
        Returns:
            Path to created reproduction package
        """
        metadata = self.load_metadata(experiment_id)
        
        package_dir = Path(output_dir) / f"reproduction_{experiment_id}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = package_dir / "experiment_metadata.json"
        metadata.export_to_file(str(metadata_path))
        
        # Save reproduction script
        script_path = package_dir / "reproduce_experiment.py"
        self._create_reproduction_script(metadata, str(script_path))
        
        # Save configuration
        config_path = package_dir / "fuzzer_config.yaml"
        from src.fuzzer.config_manager import FuzzerConfigManager
        config_manager = FuzzerConfigManager(str(config_path))
        config_manager.save_config(metadata.fuzzer_config)
        
        # Create README
        readme_path = package_dir / "README.md"
        self._create_reproduction_readme(metadata, str(readme_path))
        
        logger.info(f"Created reproduction package at {package_dir}")
        return str(package_dir)
    
    def _create_reproduction_script(self, metadata: ExperimentMetadata, script_path: str) -> None:
        """Create a Python script to reproduce the experiment.
        
        Args:
            metadata: Experiment metadata
            script_path: Path to save the script
        """
        script_content = f'''#!/usr/bin/env python3
"""
Reproduction script for experiment {metadata.experiment_id}
Generated automatically from experiment metadata.
"""

import sys
from pathlib import Path
from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator
from src.fuzzer.config_manager import FuzzerConfigManager
from src.fuzzer.random_seed_manager import get_seed_manager
from src.fuzzer.experiment_metadata import ExperimentMetadata

def main():
    """Reproduce the experiment."""
    
    # Load metadata
    metadata_path = Path(__file__).parent / "experiment_metadata.json"
    metadata = ExperimentMetadata.import_from_file(str(metadata_path))
    
    # Set up random seed manager
    seed_manager = get_seed_manager()
    seed_manager.import_state(metadata.random_seed_state)
    
    # Load configuration
    config_path = Path(__file__).parent / "fuzzer_config.yaml"
    config_manager = FuzzerConfigManager(str(config_path))
    config = config_manager.load_config()
    
    # Check document availability
    document_path = metadata.document_info.path
    if not Path(document_path).exists():
        print(f"Error: Document not found at {{document_path}}")
        print("Please ensure the original document is available at the expected path.")
        sys.exit(1)
    
    # Validate reproducibility
    validation = metadata.validate_reproducibility(document_path)
    if not all(validation.values()):
        print("Warning: Reproducibility validation failed:")
        for check, result in validation.items():
            if not result:
                print(f"  - {{check}}: FAILED")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run fuzzer simulation
    print(f"Reproducing experiment {{metadata.experiment_id}}...")
    print(f"Original results: {{metadata.total_test_cases}} tests, "
          f"{{metadata.validity_rate:.1%}} validity, "
          f"{{metadata.coverage_percentage:.1f}}% coverage")
    
    fuzzer = LLMFuzzerSimulator(config, seed_manager)
    
    try:
        # Run the simulation
        results = fuzzer.run_fuzzing_session(document_path)
        
        # Compare results
        print("\\nReproduction results:")
        print(f"Test cases: {{results.total_test_cases}} (original: {{metadata.total_test_cases}})")
        print(f"Validity rate: {{results.get_validity_rate():.1%}} (original: {{metadata.validity_rate:.1%}})")
        print(f"Coverage: {{results.get_coverage_percentage():.1f}}% (original: {{metadata.coverage_percentage:.1f}}%)")
        print(f"Defects: {{results.get_defect_count()}} (original: {{metadata.defects_found}})")
        
        # Check if results match
        tolerance = 0.01  # 1% tolerance
        matches = (
            abs(results.get_validity_rate() - metadata.validity_rate) < tolerance and
            abs(results.get_coverage_percentage() - metadata.coverage_percentage) < tolerance and
            results.get_defect_count() == metadata.defects_found
        )
        
        if matches:
            print("\\n✅ Reproduction successful! Results match original experiment.")
        else:
            print("\\n⚠️  Results differ from original experiment.")
            print("This may be due to system differences or non-deterministic behavior.")
        
    except Exception as e:
        print(f"\\n❌ Reproduction failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
    def _create_reproduction_readme(self, metadata: ExperimentMetadata, readme_path: str) -> None:
        """Create a README file for the reproduction package.
        
        Args:
            metadata: Experiment metadata
            readme_path: Path to save the README
        """
        readme_content = f'''# Experiment Reproduction Package

## Experiment Information

- **Experiment ID**: {metadata.experiment_id}
- **Session ID**: {metadata.session_id}
- **Created**: {metadata.created_at.strftime("%Y-%m-%d %H:%M:%S") if metadata.created_at else "Unknown"}
- **Duration**: {metadata.duration_seconds:.2f}s (if metadata.duration_seconds else "Unknown")
- **Document**: {metadata.document_info.path}

## Results Summary

- **Test Cases**: {metadata.total_test_cases}
- **Validity Rate**: {metadata.validity_rate:.1%}
- **Coverage**: {metadata.coverage_percentage:.1f}%
- **Defects Found**: {metadata.defects_found}

## System Information

- **Platform**: {metadata.system_info.platform}
- **Python Version**: {metadata.system_info.python_version}
- **Architecture**: {metadata.system_info.architecture}

## Reproduction Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Document Availability**:
   Make sure the original document is available at:
   ```
   {metadata.document_info.path}
   ```
   
   Document hash (SHA256): `{metadata.document_info.hash_sha256}`

3. **Run Reproduction Script**:
   ```bash
   python reproduce_experiment.py
   ```

## Files in this Package

- `experiment_metadata.json`: Complete experiment metadata
- `fuzzer_config.yaml`: Fuzzer configuration used
- `reproduce_experiment.py`: Automated reproduction script
- `README.md`: This file

## Notes

{metadata.notes if metadata.notes else "No additional notes."}

## Tags

{", ".join(metadata.tags) if metadata.tags else "No tags."}

---

Generated by LLM Fuzzer Simulator v{metadata.version}
'''
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)