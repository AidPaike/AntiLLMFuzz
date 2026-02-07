#!/usr/bin/env python3
"""Import JaCoCo coverage CSV into SQLite snapshots and details.

Usage:
  PYTHONPATH=src python scripts/import_coverage_csv.py \
    --report coverage/report --db data/antifuzz.db --run-id 1 --variant perturbed --round 1 \
    --compile-rate 0.9 --total-tests 10 --successful-tests 9
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.storage.sqlite_store import ExperimentStore


def main():
    parser = argparse.ArgumentParser(description="Import JaCoCo coverage CSV into SQLite")
    parser.add_argument("--report", required=True, help="Directory containing coverage.csv")
    parser.add_argument("--db", default="data/antifuzz.db", help="SQLite DB path")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID in the database")
    parser.add_argument("--variant", required=True, help="Variant name (e.g., original, perturbed)")
    parser.add_argument("--round", type=int, default=None, help="Optional round number")
    parser.add_argument("--compile-rate", type=float, default=None, help="Compile rate for this variant")
    parser.add_argument("--total-tests", type=int, default=None, help="Total tests executed")
    parser.add_argument("--successful-tests", type=int, default=None, help="Successful tests")
    args = parser.parse_args()

    store = ExperimentStore(args.db)
    result = store.import_coverage_csv(
        report_dir=Path(args.report),
        run_id=args.run_id,
        variant=args.variant,
        round_num=args.round,
        compile_rate=args.compile_rate,
        total_tests=args.total_tests,
        successful_tests=args.successful_tests,
    )
    print(f"Imported coverage: line={result['line_coverage']:.2f}% branch={result['branch_coverage']:.2f}% method={result['method_coverage']:.2f}%")


if __name__ == "__main__":
    main()
