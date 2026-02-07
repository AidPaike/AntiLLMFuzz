#!/usr/bin/env python3
"""End-to-end pipeline runner.

Steps:
1) Load documentation
2) Run adaptive feedback loop (perturb -> generate tests -> measure coverage) with SQLite logging
3) Optionally write perturbed doc and results to files

Usage:
  PYTHONPATH=src python scripts/run_full_pipeline.py --doc data/00java_std.md \
    --strategy enhanced_contradictory --intensity 0.5 --rounds 5 \
    --out-perturbed output/perturbed_adaptive.md --out-results output/pipeline_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.adaptive_feedback_loop import AdaptiveFeedbackLoop, AdaptiveConfig


def main():
    parser = argparse.ArgumentParser(description="Run full adaptive pipeline")
    parser.add_argument("--doc", required=True, help="Path to input documentation")
    parser.add_argument("--strategy", default="enhanced_contradictory", help="Strategy name for adaptive loop")
    parser.add_argument("--intensity", type=float, default=0.5, help="Initial perturbation intensity (0-1)")
    parser.add_argument("--rounds", type=int, default=5, help="Maximum adaptive rounds")
    parser.add_argument("--target", type=float, default=0.30, help="Target effectiveness (fraction, e.g., 0.30)")
    parser.add_argument("--out-perturbed", default=None, help="Optional file to save best perturbed doc")
    parser.add_argument("--out-results", default=None, help="Optional JSON output path for full results")
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        raise FileNotFoundError(f"Doc not found: {doc_path}")

    original_doc = doc_path.read_text(encoding="utf-8")

    config = AdaptiveConfig(
        target_effectiveness=args.target,
        max_rounds=args.rounds,
        min_compile_rate=0.60,
        intensity_step=0.1,
        strategy_switch_threshold=3,
    )

    loop = AdaptiveFeedbackLoop(config)
    results = loop.run_adaptive_loop(
        original_doc=original_doc,
        strategy_name=args.strategy,
        initial_intensity=args.intensity,
    )

    if args.out_results:
        out_json = Path(args.out_results)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to: {out_json}")

    if args.out_perturbed and results.get("best_result", {}).get("strategy"):
        # Adaptive loop does not return the doc body; reuse last perturbed doc is non-trivial.
        # Users can rerun recompute_rankings.py if a persisted doc is needed; here we just note best parameters.
        out_txt = Path(args.out_perturbed)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "strategy": results.get("best_result", {}).get("strategy"),
            "intensity": results.get("best_result", {}).get("intensity"),
            "note": "Run recompute_rankings.py with the above parameters to materialize perturbed doc."
        }
        out_txt.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Perturbation summary saved to: {out_txt}")


if __name__ == "__main__":
    main()
