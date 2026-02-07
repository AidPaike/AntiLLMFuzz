#!/usr/bin/env python3
"""Installation script for anti_llm4fuzz with fuzzer simulator integration."""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run a shell command with error handling."""
    print(f"Running: {command}")
    
    try:
        if capture_output:
            result = subprocess.run(
                command, 
                shell=True, 
                check=check, 
                capture_output=True, 
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Exit code: {e.returncode}")
        if capture_output and e.stdout:
            print(f"Stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"Stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def check_system_requirements():
    """Check system requirements."""
    print("Checking system requirements...")
    
    # Check Python version
    check_python_version()
    
    # Check pip
    try:
        pip_version = run_command("pip --version", capture_output=True)
        print(f"✓ pip: {pip_version}")
    except:
        print("Error: pip is not available")
        sys.exit(1)
    
    # Check git (optional)
    try:
        git_version = run_command("git --version", capture_output=True, check=False)
        if git_version:
            print(f"✓ git: {git_version}")
        else:
            print("⚠ git not available (optional)")
    except:
        print("⚠ git not available (optional)")
    
    print("System requirements check completed.\n")


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip")
    
    # Install main dependencies
    run_command("pip install -e .")
    
    # Install spaCy model
    print("Downloading spaCy English model...")
    run_command("python -m spacy download en_core_web_sm")
    
    print("Dependencies installation completed.\n")


def install_optional_dependencies():
    """Install optional dependencies."""
    print("Installing optional dependencies...")
    
    # Ask user which optional dependencies to install
    optional_groups = {
        'dev': 'Development tools (pytest, black, mypy)',
        'fuzzer': 'Additional LLM clients (OpenAI, Anthropic)',
        'visualization': 'Visualization tools (matplotlib, plotly)',
        'all': 'All optional dependencies'
    }
    
    print("Available optional dependency groups:")
    for key, description in optional_groups.items():
        print(f"  {key}: {description}")
    
    choice = input("\nEnter dependency groups to install (comma-separated, or 'none'): ").strip()
    
    if choice.lower() == 'none':
        print("Skipping optional dependencies.\n")
        return
    
    groups = [g.strip() for g in choice.split(',') if g.strip()]
    
    for group in groups:
        if group in optional_groups:
            print(f"Installing {group} dependencies...")
            run_command(f"pip install -e .[{group}]")
        else:
            print(f"Warning: Unknown dependency group '{group}'")
    
    print("Optional dependencies installation completed.\n")


def install_jacoco_tools():
    """Download JaCoCo CLI and agent jars."""
    print("Installing JaCoCo tools...")

    jacoco_dir = Path("tools/jacoco")
    jacoco_dir.mkdir(parents=True, exist_ok=True)

    version = "0.8.12"
    cli_jar = jacoco_dir / "jacococli.jar"
    agent_jar = jacoco_dir / "jacocoagent.jar"
    base_url = "https://repo1.maven.org/maven2/org/jacoco"

    if not cli_jar.exists():
        run_command(
            f"curl -L {base_url}/org.jacoco.cli/{version}/org.jacoco.cli-{version}-nodeps.jar -o {cli_jar}"
        )

    if not agent_jar.exists():
        run_command(
            f"curl -L {base_url}/org.jacoco.agent/{version}/org.jacoco.agent-{version}-runtime.jar -o {agent_jar}"
        )

    print("JaCoCo tools installed.\n")



def setup_configuration():
    """Set up configuration files."""
    print("Setting up configuration...")
    
    # Create necessary directories
    directories = [
        'logs',
        'output',
        'validation_results',
        '.kiro/config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Copy default configuration if it doesn't exist
    config_file = Path('config/config.yaml')
    if config_file.exists():
        print(f"✓ Configuration file exists: {config_file}")
    else:
        print(f"⚠ Configuration file not found: {config_file}")
        print("  Please ensure config/config.yaml exists with proper settings")
    
    print("Configuration setup completed.\n")


def setup_environment_variables():
    """Guide user through environment variable setup."""
    print("Environment Variables Setup")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv('HUIYAN_API_KEY')
    if api_key:
        print("✓ HUIYAN_API_KEY is set")
    else:
        print("⚠ HUIYAN_API_KEY is not set")
        print("  This is required for LLM-based fuzzer simulation")
        print("  Set it with: export HUIYAN_API_KEY=your_api_key")
    
    # Other optional environment variables
    optional_vars = {
        'FUZZER_LLM_MODEL': 'LLM model to use (default: gpt-4)',
        'FUZZER_CASES_PER_API': 'Number of test cases per API (default: 20)',
        'FUZZER_PARALLEL_EXECUTION': 'Enable parallel execution (true/false)',
        'FUZZER_RANDOM_SEED': 'Random seed for reproducible results'
    }
    
    print("\nOptional environment variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✓ {var}={value}")
        else:
            print(f"  {var}: {description}")

    print("\nJaCoCo setup:")
    print("  tools/jacoco/jacococli.jar")
    print("  tools/jacoco/jacocoagent.jar")
    print("  scripts/docker/fetch_jacoco.sh (optional helper)")
    
    print("\nEnvironment variables setup completed.\n")



def run_validation():
    """Run system validation."""
    print("Running system validation...")
    
    try:
        # Run basic import test
        print("Testing basic imports...")
        run_command("python -c \"from src.fuzzer.llm_fuzzer_simulator import LLMFuzzerSimulator; print('✓ Fuzzer simulator import successful')\"")
        
        # Run integration validation
        print("Running integration validation...")
        validation_script = Path("tools/validate_system_integration.py")
        if validation_script.exists():
            run_command(f"python {validation_script} --quick")
        else:
            print("⚠ Validation script not found, skipping detailed validation")
        
        print("✓ System validation completed successfully")
        
    except Exception as e:
        print(f"⚠ System validation failed: {e}")
        print("  The system may still work, but some features might not be available")
    
    print()


def create_desktop_shortcuts():
    """Create desktop shortcuts (Windows/Linux)."""
    if platform.system() not in ['Windows', 'Linux']:
        return
    
    choice = input("Create desktop shortcuts? (y/n): ").strip().lower()
    if choice != 'y':
        return
    
    print("Creating desktop shortcuts...")
    
    # This is a simplified implementation
    # In a real deployment, you'd create proper shortcuts
    shortcuts_dir = Path.home() / "Desktop" / "anti_llm4fuzz"
    shortcuts_dir.mkdir(exist_ok=True)
    
    # Create batch files or shell scripts
    if platform.system() == 'Windows':
        # Windows batch file
        batch_content = f"""@echo off
cd /d "{Path.cwd()}"
python -m src.cli %*
pause
"""
        with open(shortcuts_dir / "anti_llm4fuzz.bat", 'w') as f:
            f.write(batch_content)
    else:
        # Linux shell script
        script_content = f"""#!/bin/bash
cd "{Path.cwd()}"
python -m src.cli "$@"
"""
        script_file = shortcuts_dir / "anti_llm4fuzz.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        script_file.chmod(0o755)
    
    print(f"✓ Shortcuts created in: {shortcuts_dir}")
    print()


def print_installation_summary():
    """Print installation summary and next steps."""
    print("=" * 60)
    print("INSTALLATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nInstalled Components:")
    print("✓ Core anti_llm4fuzz package")
    print("✓ LLM Fuzzer Simulator")
    print("✓ Integration modules")
    print("✓ Validation tools")
    print("✓ CLI tools")
    print("✓ JaCoCo jars")

    
    print("\nAvailable Commands:")
    print("  anti_llm4fuzz          - Main CLI tool")
    print("  fuzzer-simulator       - Standalone fuzzer simulator")
    print("  validate-fuzzer-integration - System validation")
    
    print("\nNext Steps:")
    print("1. Set up your API key:")
    print("   export HUIYAN_API_KEY=your_api_key")
    print()
    print("2. Test the installation:")
    print("   anti_llm4fuzz --help")
    print("   fuzzer-simulator --help")
    print()
    print("3. Run a quick test:")
    print("   anti_llm4fuzz --input data/00java_std.md --top-n 3")
    print()
    print("4. Enable fuzzer integration:")
    print("   anti_llm4fuzz --input data/00java_std.md --enable-fuzzer-integration")
    print()
    print("5. Build OpenJDK (for real javac coverage):")
    print("   scripts/docker/build_jdk23.sh")
    print("   scripts/docker/fetch_jacoco.sh")
    print()
    print("6. Validate the system:")
    print("   validate-fuzzer-integration")
    
    print("\nDocumentation:")
    print("  README.md              - Getting started guide")
    print("  docs/                  - Comprehensive documentation")
    print("  config/                - Configuration examples")

    
    print("\nSupport:")
    print("  Check logs/            - For troubleshooting")
    print("  Run with --verbose     - For detailed output")
    
    print("\n" + "=" * 60)


def main():
    """Main installation function."""
    print("=" * 60)
    print("ANTI_LLM4FUZZ INSTALLATION SCRIPT")
    print("LLM Fuzzer Semantic Disruptor with Fuzzer Simulator")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Check system requirements
        check_system_requirements()
        
        # Step 2: Install dependencies
        install_dependencies()
        
        # Step 3: Install optional dependencies
        install_optional_dependencies()
        
        # Step 4: Setup configuration
        setup_configuration()
        
        # Step 5: Setup environment variables
        setup_environment_variables()
        
        # Step 6: Install JaCoCo tools
        install_jacoco_tools()
        
        # Step 7: Run validation
        run_validation()
        
        # Step 8: Create shortcuts (optional)
        create_desktop_shortcuts()
        
        # Step 9: Print summary
        print_installation_summary()

        
        return 0
        
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
