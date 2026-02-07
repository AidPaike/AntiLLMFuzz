#!/usr/bin/env python3
"""Export a static HTML dashboard from SQLite artifacts/logs/runs.

Usage:
  PYTHONPATH=src python scripts/export_dashboard.py --db data/antifuzz.db --out output/dashboard.html
"""

import argparse
import html
import sqlite3
from pathlib import Path


def fetch(conn, query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def table_html(title, cols, rows, limit=200):
    rows = rows[:limit]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f"<h3>{html.escape(title)}</h3><table border='1' cellspacing='0' cellpadding='4'><tr>{head}</tr>{body}</table>"


def main():
    ap = argparse.ArgumentParser(description="Export static dashboard HTML from SQLite")
    ap.add_argument("--db", default="data/antifuzz.db", help="SQLite database path")
    ap.add_argument("--out", default="output/dashboard.html", help="Output HTML file")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    sections = []

    queries = [
        ("Runs", "SELECT id, started_at, strategy, initial_intensity, target_effectiveness, baseline_coverage, model, endpoint FROM runs ORDER BY id DESC"),
        ("Rounds", "SELECT id, run_id, round_num, strategy, intensity, compile_rate, coverage, effectiveness, is_best, created_at FROM rounds ORDER BY id DESC"),
        ("Coverage Snapshots", "SELECT id, run_id, round_num, variant, line_coverage, branch_coverage, method_coverage, compile_rate, total_tests, successful_tests, created_at FROM coverage_snapshots ORDER BY id DESC"),
        ("Seeds", "SELECT id, run_id, round_num, variant, compiled, coverage, substr(content,1,200) AS snippet, created_at FROM seeds ORDER BY id DESC"),
        ("Prompts", "SELECT id, run_id, round_num, strategy, substr(prompt_text,1,200) AS prompt_snippet, substr(response_text,1,200) AS resp_snippet, model, temperature, max_tokens, created_at FROM prompts ORDER BY id DESC"),
        ("Artifacts", "SELECT id, run_id, round_num, type, path, size_bytes, substr(content,1,200) AS snippet, created_at FROM artifacts ORDER BY id DESC"),
        ("Reports", "SELECT id, run_id, round_num, report_type, path, substr(content,1,200) AS snippet, created_at FROM reports ORDER BY id DESC"),
        ("Crashes", "SELECT id, run_id, round_num, test_id, message, substr(stack,1,200) AS stack_snippet, created_at FROM crashes ORDER BY id DESC"),
    ]

    for title, query in queries:
        try:
            cols, rows = fetch(conn, query)
            sections.append(table_html(title, cols, rows))
        except sqlite3.OperationalError:
            continue

    html_doc = """
<!doctype html>
<html><head><meta charset='utf-8'><title>Disruptor Dashboard</title>
<style>body{font-family:Arial,Helvetica,sans-serif;font-size:14px;} table{margin-bottom:24px;} h2{margin-top:32px;}</style>
</head><body>
<h1>Fuzzing Disruptor Dashboard</h1>
<p>Data source: {db}</p>
{sections}
</body></html>
""".format(db=html.escape(args.db), sections="\n".join(sections))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Dashboard written to: {out_path}")


if __name__ == "__main__":
    main()
