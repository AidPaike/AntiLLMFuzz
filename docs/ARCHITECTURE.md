# LLM Fuzzer Semantic Disruptor - Architecture

This document summarizes the architecture and execution flow of the semantic perturbation pipeline.

## System Overview

```
Input file
  → Extractor (doc/Java/Python)
  → Token prioritizer (security/validation scoring)
  → Strategy selector (semantic/generic/all)
  → Perturbation application (top-N tokens)
  → Output variants + metadata
```

## Core Modules

- `src/extractors/`: language-specific token extraction
- `src/token_prioritizer.py`: security/validation-aware ranking
- `src/strategies/`: semantic + generic perturbation strategies
- `src/scs/`: semantic contribution score (SCS) + hotspot analysis
- `src/feedback/`: feedback simulation for SCS calculations
- `main.py`: pipeline orchestration

## Strategy Layer

- Semantic: tokenization_drift, lexical_disguise, dataflow_misdirection, controlflow_misdirection, documentation_deception, cognitive_manipulation
- Generic: formatting_noise, structural_noise, paraphrasing, cognitive_load

All strategies expose `supported_targets` and `code_safety` flags. The pipeline filters strategies based on whether the input is code or documentation.

## Metadata Output

Each run produces:
- Perturbed files (`output/perturbations_<timestamp>/...`)
- `metadata_<timestamp>.json` with:
  - token ranking & scores
  - applied strategies
  - before/after snippets
  - optional SCS statistics

## Logging Pipeline

- Extraction summary: total tokens, token types
- Priority scoring: counts of security/validation tokens + top-N token summary
- Strategy execution: applied strategy list and variant preview
- Output: number of files, output directory
