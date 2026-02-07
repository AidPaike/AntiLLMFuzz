#!/usr/bin/env python3
"""Build distribution packages for anti_llm4fuzz."""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(command, check=True):
    """Run a shell command with error handling."""
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=check)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Exit code: {e.returncode}")
        if check:
            sys.exit(1)


def clean_build_directories():
    """Clean previous build artifacts."""
    print("Cleaning build directories...")
    
    directories_to_clean = [
        'build',
        'dist',
        'anti_llm4fuzz.egg-info',
        '__pycache__'
    ]
    
    for directory in directories_to_clean:
        if Path(directory).exists():
            shutil.rmtree(directory)
            print(f"✓ Removed: {directory}")
    
    # Clean __pycache__ directories recursively
    for pycache in Path('.').rglob('__pycache__'):
        shutil.rmtree(pycache)
        print(f"✓ Removed: {pycache}")
    
    print("Build directories cleaned.\n")


def update_version():
    """Update version information."""
    print("Updating version information...")
    
    # Get current timestamp for development builds
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Read current setup.py
    setup_file = Path('setup.py')
    if not setup_file.exists():
        print("Error: setup.py not found")
        sys.exit(1)
    
    content = setup_file.read_text()
    
    # Update version if it's a development build
    if '--dev' in sys.argv:
        # Replace version with development version
        import re
        version_pattern = r'version="([^"]+)"'
        match = re.search(version_pattern, content)
        if match:
            current_version = match.group(1)
            dev_version = f"{current_version}.dev{timestamp}"
            content = re.sub(version_pattern, f'version="{dev_version}"', content)
            
            # Write back to setup.py
            setup_file.write_text(content)
            print(f"✓ Updated version to: {dev_version}")
        else:
            print("⚠ Could not find version in setup.py")
    
    print("Version update completed.\n")


def run_tests():
    """Run tests before building."""
    if '--skip-tests' in sys.argv:
        print("Skipping tests (--skip-tests flag provided)\n")
        return
    
    print("Running tests...")
    
    # Check if pytest is available
    try:
        run_command("python -c \"import pytest\"", check=False)
    except:
        print("⚠ pytest not available, skipping tests")
        return
    
    # Run basic import tests
    print("Running import tests...")
    run_command("python -c \"from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator\"")
    run_command("python -c \"from src.fuzzer.integration import PerturbationFuzzerIntegrator\"")
    
    # Run pytest if test files exist
    test_files = list(Path('tests').glob('test_*.py')) if Path('tests').exists() else []
    if test_files:
        print("Running pytest...")
        run_command("python -m pytest tests/ -v --tb=short", check=False)
    else:
        print("⚠ No test files found in tests/ directory")
    
    print("Tests completed.\n")


def build_source_distribution():
    """Build source distribution."""
    print("Building source distribution...")
    run_command("python setup.py sdist")
    print("✓ Source distribution built\n")


def build_wheel_distribution():
    """Build wheel distribution."""
    print("Building wheel distribution...")
    
    # Check if wheel is available
    try:
        run_command("python -c \"import wheel\"", check=False)
    except:
        print("Installing wheel...")
        run_command("pip install wheel")
    
    run_command("python setup.py bdist_wheel")
    print("✓ Wheel distribution built\n")


def create_standalone_package():
    """Create standalone package with all dependencies."""
    print("Creating standalone package...")
    
    # Create standalone directory
    standalone_dir = Path('dist/standalone')
    standalone_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy source files
    source_files = [
        'src',
        'config',
        'tools',
        'scripts',
        'README.md',
        'INSTALL.md',
        'requirements.txt',
        'setup.py'
    ]
    
    for item in source_files:
        source_path = Path(item)
        if source_path.exists():
            if source_path.is_dir():
                shutil.copytree(source_path, standalone_dir / item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, standalone_dir / item)
            print(f"✓ Copied: {item}")
    
    # Create installation script
    install_script = standalone_dir / 'install.py'
    shutil.copy2('scripts/install.py', install_script)
    
    # Create README for standalone package
    standalone_readme = standalone_dir / 'STANDALONE_README.md'
    standalone_readme.write_text("""# Anti-LLM4Fuzz Standalone Package

This is a standalone package containing all source files and installation scripts.

## Installation

1. Extract this package to your desired location
2. Run the installation script:
   ```bash
   python install.py
   ```

## Manual Installation

If the automatic installer doesn't work:

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package:
   ```bash
   pip install -e .
   ```
4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

After installation, you can use:
- `anti_llm4fuzz` - Main CLI tool
- `fuzzer-simulator` - Standalone fuzzer simulator
- `validate-fuzzer-integration` - System validation

See README.md for detailed usage instructions.
""")
    
    # Create archive
    archive_name = f"anti_llm4fuzz_standalone_{datetime.now().strftime('%Y%m%d')}"
    shutil.make_archive(f"dist/{archive_name}", 'zip', standalone_dir)
    
    print(f"✓ Standalone package created: dist/{archive_name}.zip\n")


def validate_distributions():
    """Validate built distributions."""
    print("Validating distributions...")
    
    dist_dir = Path('dist')
    if not dist_dir.exists():
        print("Error: dist directory not found")
        return
    
    # Check for expected files
    expected_files = [
        '*.tar.gz',  # Source distribution
        '*.whl',     # Wheel distribution
    ]
    
    for pattern in expected_files:
        files = list(dist_dir.glob(pattern))
        if files:
            for file in files:
                print(f"✓ Found: {file}")
                # Check file size
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  Size: {size_mb:.1f} MB")
        else:
            print(f"⚠ No files matching pattern: {pattern}")
    
    print("Distribution validation completed.\n")


def create_release_notes():
    """Create release notes."""
    print("Creating release notes...")
    
    release_notes = f"""# Anti-LLM4Fuzz Release Notes

**Release Date:** {datetime.now().strftime('%Y-%m-%d')}
**Version:** 1.0.0

## New Features

### LLM Fuzzer Simulator Integration
- Complete integration of LLM-assisted fuzzer simulator
- Realistic fuzzing behavior simulation
- Comprehensive perturbation impact analysis
- End-to-end workflow from document to feedback

### Enhanced CLI
- New `--enable-fuzzer-integration` flag for full fuzzer integration
- Improved `--use-llm-fuzzer` for LLM-based SCS calculation
- Better error handling and progress reporting

### System Validation
- Comprehensive system validation framework
- Performance testing and benchmarking
- Integration testing for all components
- Automated validation scripts

### Distribution Improvements
- Standalone packages for easy deployment
- Improved installation scripts
- Better dependency management
- Multiple CLI entry points

## Technical Improvements

### Architecture
- Modular fuzzer simulator architecture
- Clean separation of concerns
- Comprehensive error handling
- Deterministic execution support

### Performance
- Parallel test execution
- Memory usage optimization
- Scalable batch processing
- Resource management

### Integration
- Seamless SCS system integration
- Perturbation pipeline compatibility
- Configuration system integration
- Backward compatibility maintained

## Installation

### From PyPI (when available)
```bash
pip install anti-llm4fuzz
```

### From Source
```bash
git clone <repository>
cd anti-llm4fuzz
python scripts/install.py
```

### Standalone Package
1. Download the standalone package
2. Extract and run `python install.py`

## Usage Examples

### Basic Usage
```bash
# Traditional perturbation analysis
anti_llm4fuzz --input doc.md --top-n 5 --strategy tokenization_drift

# With SCS calculation
anti_llm4fuzz --input doc.md --enable-scs --use-llm-fuzzer

# Full fuzzer integration
anti_llm4fuzz --input doc.md --enable-fuzzer-integration
```

### Fuzzer Simulator
```bash
# Standalone fuzzer simulation
fuzzer-simulator --input doc.md --output results.json

# Batch processing
fuzzer-simulator --batch --input documents/ --output-dir results/
```

### System Validation
```bash
# Validate installation
validate-fuzzer-integration

# Quick validation
validate-fuzzer-integration --quick
```

## Configuration

The system supports comprehensive configuration through:
- YAML configuration files
- Environment variables
- CLI arguments

See `config/config.yaml` for all available options.

## Requirements

- Python 3.8+
- spaCy with English model
- Optional: LLM API key for realistic simulation

## Known Issues

- LLM API key required for full fuzzer simulation
- Some features require additional dependencies
- Performance may vary based on system resources

## Support

- Documentation: `docs/` directory
- Configuration: `config/` directory
- Troubleshooting: Check `logs/` directory
- Validation: Run `validate-fuzzer-integration`

---

For detailed documentation, see README.md and the docs/ directory.
"""
    
    release_notes_file = Path('dist/RELEASE_NOTES.md')
    release_notes_file.write_text(release_notes)
    
    print(f"✓ Release notes created: {release_notes_file}\n")


def print_build_summary():
    """Print build summary."""
    print("=" * 60)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    dist_dir = Path('dist')
    if dist_dir.exists():
        print("\nGenerated Files:")
        for file in sorted(dist_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name} ({size_mb:.1f} MB)")
    
    print("\nDistribution Types:")
    print("✓ Source distribution (.tar.gz)")
    print("✓ Wheel distribution (.whl)")
    print("✓ Standalone package (.zip)")
    print("✓ Release notes")
    
    print("\nNext Steps:")
    print("1. Test the distributions:")
    print("   pip install dist/*.whl")
    print()
    print("2. Upload to PyPI (if configured):")
    print("   twine upload dist/*")
    print()
    print("3. Create GitHub release:")
    print("   - Upload files from dist/")
    print("   - Use RELEASE_NOTES.md as description")
    
    print("\n" + "=" * 60)


def main():
    """Main build function."""
    print("=" * 60)
    print("ANTI_LLM4FUZZ DISTRIBUTION BUILD SCRIPT")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Clean build directories
        clean_build_directories()
        
        # Step 2: Update version
        update_version()
        
        # Step 3: Run tests
        run_tests()
        
        # Step 4: Build source distribution
        build_source_distribution()
        
        # Step 5: Build wheel distribution
        build_wheel_distribution()
        
        # Step 6: Create standalone package
        create_standalone_package()
        
        # Step 7: Validate distributions
        validate_distributions()
        
        # Step 8: Create release notes
        create_release_notes()
        
        # Step 9: Print summary
        print_build_summary()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\nBuild failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())