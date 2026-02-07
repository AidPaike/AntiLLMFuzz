#!/usr/bin/env python3
"""Recompute token rankings after crashes and re-run perturbations.

Usage:
  python scripts/recompute_rankings.py --doc data/00java_std.md \
    --db data/antifuzz.db --strategy enhanced_contradictory --intensity 0.6 \
    --out output/perturbed_recomputed.md
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.extractors import DocumentationTokenExtractor
from src.token_prioritizer import TokenPrioritizer
from src.strategies.semantic import EnhancedContradictoryStrategy, ContradictoryInfoStrategy


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS coverage_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            round_num INTEGER,
            variant TEXT NOT NULL,
            line_coverage REAL,
            branch_coverage REAL,
            method_coverage REAL,
            compile_rate REAL,
            total_tests INTEGER,
            successful_tests INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def load_crash_weight(db_path: Path) -> float:
    if not db_path.exists():
        return 0.0
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT AVG(CASE WHEN compile_rate IS NOT NULL THEN 1 - compile_rate ELSE 0 END) FROM coverage_snapshots"
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0
    finally:
        conn.close()


def recompute(doc_path: Path, db_path: Path, strategy_name: str, intensity: float) -> str:
    doc_text = doc_path.read_text(encoding="utf-8")
    extractor = DocumentationTokenExtractor()
    tokens = extractor.extract_tokens(str(doc_path))

    prioritizer = TokenPrioritizer()
    tokens = prioritizer.rank_tokens(prioritizer.assign_scores(tokens))

    crash_weight = load_crash_weight(db_path)
    if crash_weight > 0 and tokens:
        offset = max(1, int(len(tokens) * min(crash_weight, 0.5)))
        tokens = tokens[offset:] + tokens[:offset]

    top_tokens = tokens[:max(1, int(5 * intensity))]

    if strategy_name == "enhanced_contradictory":
        strategy = EnhancedContradictoryStrategy()
        perturbed = strategy.apply(top_tokens[0], doc_text, intensity=intensity) if top_tokens else doc_text
    else:
        strategy = ContradictoryInfoStrategy()
        versions = strategy.apply_multiple(top_tokens, doc_text, max_tokens=len(top_tokens))
        perturbed = list(versions.values())[0] if versions else doc_text
    return perturbed


def main():
    parser = argparse.ArgumentParser(description="Recompute rankings after crashes and rerun perturbations")
    parser.add_argument("--doc", required=True, help="Input documentation path")
    parser.add_argument("--db", default="data/antifuzz.db", help="SQLite path for historical runs")
    parser.add_argument("--strategy", default="enhanced_contradictory", choices=["enhanced_contradictory", "contradictory_info"], help="Strategy to apply")
    parser.add_argument("--intensity", type=float, default=0.5, help="Perturbation intensity (0-1)")
    parser.add_argument("--out", default=None, help="Optional output file for perturbed doc")
    args = parser.parse_args()

    perturbed = recompute(Path(args.doc), Path(args.db), args.strategy, args.intensity)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(perturbed, encoding="utf-8")
        print(f"Perturbed doc written to: {out_path}")
    else:
        print(perturbed)


if __name__ == "__main__":
    main()
