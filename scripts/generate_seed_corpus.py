#!/usr/bin/env python3
"""Generate a large seed corpus before/after perturbation for comparison.

Usage:
  PYTHONPATH=src python scripts/generate_seed_corpus.py \
    --doc data/00java_std.md \
    --strategy enhanced_contradictory \
    --intensity 0.6 \
    --count 10000 \
    --out-before seeds_before.txt \
    --out-after seeds_after.txt

Notes:
- This script only generates code via LLM; it does not compile.
- Large counts (e.g., 10k) are slow and LLM-costly; adjust as needed.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.semantic import (
    EnhancedContradictoryStrategy,
    ContextPoisoningStrategy,
)
from src.extractors import DocumentationTokenExtractor
from src.token_prioritizer import TokenPrioritizer
from src.utils.llm_client import LLMClient


def perturb_doc(doc: str, strategy_name: str, intensity: float) -> str:
    extractor = DocumentationTokenExtractor()
    # fallback: write to temp and reuse extractor API
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(doc)
        temp_path = f.name
    try:
        tokens = extractor.extract_tokens(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    prioritizer = TokenPrioritizer()
    tokens = prioritizer.rank_tokens(prioritizer.assign_scores(tokens))
    top_tokens = tokens[: max(1, int(5 * intensity))]

    if strategy_name == "enhanced_contradictory":
        strat = EnhancedContradictoryStrategy()
        return strat.apply(top_tokens[0], doc, intensity=intensity) if top_tokens else doc
    if strategy_name == "context_poison":
        strat = ContextPoisoningStrategy()
        versions = strat.apply_multiple(top_tokens, doc, max_tokens=len(top_tokens))
        return list(versions.values())[0] if versions else doc
    return doc


def generate_seeds(doc: str, count: int, client: LLMClient):
    seeds = []
    base_prompt = f"Based on the following documentation, generate a small Java snippet that demonstrates usage. Keep it under 40 lines. Output code only.\n\n{doc[:1800]}\n"
    for _ in range(count):
        resp = client.simple_completion(base_prompt, temperature=0.4, max_tokens=300)
        seeds.append(resp.strip())
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Generate seed corpora before/after perturbation")
    parser.add_argument("--doc", required=True, help="Input documentation path")
    parser.add_argument("--strategy", default="enhanced_contradictory", choices=["enhanced_contradictory", "context_poison", "reasoning_distraction"], help="Perturbation strategy")
    parser.add_argument("--intensity", type=float, default=0.6, help="Perturbation intensity")
    parser.add_argument("--count", type=int, default=100, help="Number of seeds per variant")
    parser.add_argument("--out-before", default="seeds_before.txt", help="Output file for seeds before perturbation")
    parser.add_argument("--out-after", default="seeds_after.txt", help="Output file for seeds after perturbation")
    args = parser.parse_args()

    doc_path = Path(args.doc)
    doc_text = doc_path.read_text(encoding="utf-8")

    perturbed_doc = perturb_doc(doc_text, args.strategy, args.intensity)

    client = LLMClient(endpoint="http://localhost:11434/api/generate", model="qwen3-java", timeout=120)

    seeds_before = generate_seeds(doc_text, args.count, client)
    seeds_after = generate_seeds(perturbed_doc, args.count, client)

    Path(args.out_before).write_text("\n\n".join(seeds_before), encoding="utf-8")
    Path(args.out_after).write_text("\n\n".join(seeds_after), encoding="utf-8")

    print(f"Generated {len(seeds_before)} seeds before perturbation -> {args.out_before}")
    print(f"Generated {len(seeds_after)} seeds after perturbation  -> {args.out_after}")


if __name__ == "__main__":
    main()
