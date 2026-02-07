# Contributing to Anti-LLM Fuzzing Disruptor

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install in development mode: `pip install -e ".[dev]"`
5. Download spaCy model: `python -m spacy download en_core_web_sm`

## Code Style

- Follow PEP 8 guidelines
- Use Black for formatting: `black src/ tests/`
- Use type hints where appropriate
- Maximum line length: 100 characters

## Testing

Run tests with pytest:

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=html
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters

## Questions?

Feel free to open an issue for questions or discussion.
