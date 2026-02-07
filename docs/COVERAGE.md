# Coverage & Reporting Guide

## Why keep coverage files out of VCS?
JaCoCo CSV/HTML can be large. Do **not** commit coverage/report or /tmp/coverage_*; store locally and import summaries into SQLite.

## Quick Run (JaCoCo)
1) Ensure JDK path: repos/openjdk-jdk23/build_artifacts/jdk.
2) Run a pipeline (examples):
```bash
export PYTHONPATH=src
python src/adaptive_feedback_loop.py
# or
python scripts/run_full_pipeline.py --doc data/00java_std.md --strategy enhanced_contradictory --intensity 0.5 --rounds 5 --target 0.25
```
3) JaCoCo outputs usually appear in /tmp/coverage_*/coverage/report (CSV+HTML) or coverage/report if scripts place them there.

## Import Coverage to SQLite
```bash
export PYTHONPATH=src
python scripts/import_coverage_csv.py --report /path/to/coverage/report --db data/antifuzz.db --run-id 1 --variant perturbed
```
Tables populated:
- coverage_snapshots (line/branch/method coverage, compile_rate, totals)
- coverage_details (per-class line/branch/methods)
- reports (path + optional inline CSV content)

## Viewing Results
```bash
export PYTHONPATH=src
python scripts/export_dashboard.py --db data/antifuzz.db --out output/dashboard.html
```
Open output/dashboard.html in a browser to view runs/rounds/coverage/seeds/prompts/crashes/artifacts.

## Cleaning Up
- Remove /tmp/coverage_* after inspection.
- Keep data/antifuzz.db and output/dashboard.html locally; avoid committing to VCS.

## Troubleshooting
- Coverage = 0: check that generated seeds compiled; fallback seed is injected if none compile, but JaCoCo still needs execution paths.
- Large CSV: importer handles it; avoid in-repo commits. If CSV missing, reports table will note path=None.
