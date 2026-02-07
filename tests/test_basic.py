#!/usr/bin/env python3
"""
Quick verification test for Anti-LLM Fuzzing Disruptor

Run this test to verify the installation is working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all critical modules can be imported."""
    print("Testing imports...")
    
    tests = [
        ("utils", "get_logger"),
        ("data_models", "Token"),
        ("token_prioritizer", "TokenPrioritizer"),
        ("extractors", "DocumentationTokenExtractor"),
        ("strategies.base_strategy", "PerturbationStrategy"),
        ("storage.sqlite_store", "ExperimentStore"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, item_name in tests:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"  ✓ {module_name}.{item_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {module_name}.{item_name}: {e}")
            failed += 1
    
    return passed, failed


def test_token_prioritizer():
    """Test token prioritizer functionality."""
    print("\nTesting token prioritizer...")
    
    try:
        from token_prioritizer import TokenPrioritizer
        from data_models import Token
        
        prioritizer = TokenPrioritizer()
        token = Token(
            text='MessageDigest',
            token_type='function',
            line=1,
            column=0,
            source_file='test.md'
        )
        
        tokens = prioritizer.assign_scores([token])
        assert len(tokens) == 1
        assert tokens[0].text == 'MessageDigest'
        
        print("  ✓ Token prioritizer works")
        return 1, 0
    except Exception as e:
        print(f"  ✗ Token prioritizer failed: {e}")
        return 0, 1


def test_strategies():
    """Test strategy loading."""
    print("\nTesting strategies...")
    
    strategies = [
        "enhanced_contradictory",
        "context_poisoning",
        "contradictory_info",
    ]
    
    passed = 0
    failed = 0
    
    for strategy_name in strategies:
        try:
            module = __import__(
                f"strategies.semantic.{strategy_name}",
                fromlist=[strategy_name.replace('_', ' ').title().replace(' ', '') + "Strategy"]
            )
            print(f"  ✓ {strategy_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {strategy_name}: {e}")
            failed += 1
    
    return passed, failed


def test_data_file():
    """Test that sample data file exists."""
    print("\nTesting data files...")
    
    data_file = os.path.join(
        os.path.dirname(__file__), '..', 'data', '00java_std.md'
    )
    
    if os.path.exists(data_file):
        print(f"  ✓ Sample data file exists")
        return 1, 0
    else:
        print(f"  ✗ Sample data file not found: {data_file}")
        return 0, 1


def main():
    """Run all tests."""
    print("=" * 60)
    print("Anti-LLM Fuzzing Disruptor - Quick Verification")
    print("=" * 60)
    print()
    
    results = []
    results.append(test_imports())
    results.append(test_token_prioritizer())
    results.append(test_strategies())
    results.append(test_data_file())
    
    total_passed = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    
    print()
    print("=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    if total_failed == 0:
        print("\n✓ All tests passed! System is ready to use.")
        print()
        print("Quick start:")
        print("  export PYTHONPATH=src")
        print("  python src/adaptive_feedback_loop.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
