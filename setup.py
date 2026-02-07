#!/usr/bin/env python3
"""
Anti-LLM Fuzzing Disruptor
==========================

Research-grade system to degrade LLM-assisted fuzzing via documentation 
poisoning, adversarial prompting, and coverage/crash tracking.

For more information, see README.md
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="antillmfuzz",
    version="1.0.0",
    author="AntiLLMFuzz Team",
    author_email="contact@antillmfuzz.dev",
    description="Research-grade system to degrade LLM-assisted fuzzing via documentation poisoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antifuzz/antillmfuzz",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.10",
    install_requires=[
        "javalang>=0.13.0",
        "spacy>=3.7.0",
        "requests>=2.31.0",
        "pytest>=7.4.0",
        "hypothesis>=6.92.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "antillmfuzz=antillmfuzz.cli:main",
        ],
    },
)
