"""SQLite experiment store for perturbation runs, coverage metrics, and artifacts."""

import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any
from datetime import datetime


class ExperimentStore:
    def __init__(self, db_path: str = "data/antifuzz.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    initial_intensity REAL,
                    target_effectiveness REAL,
                    baseline_coverage REAL,
                    model TEXT,
                    endpoint TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    round_num INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    compile_rate REAL,
                    coverage REAL,
                    effectiveness REAL,
                    is_best INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    round_num INTEGER NOT NULL,
                    code TEXT,
                    compiled INTEGER,
                    coverage REAL,
                    errors TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
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
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_coverage_run_variant ON coverage_snapshots(run_id, variant)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS coverage_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    round_num INTEGER,
                    variant TEXT NOT NULL,
                    package TEXT,
                    clazz TEXT,
                    line_missed INTEGER,
                    line_covered INTEGER,
                    branch_missed INTEGER,
                    branch_covered INTEGER,
                    method_missed INTEGER,
                    method_covered INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_coverage_details_run_variant ON coverage_details(run_id, variant)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    round_num INTEGER,
                    strategy TEXT,
                    prompt_text TEXT,
                    response_text TEXT,
                    model TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS seeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    round_num INTEGER,
                    variant TEXT,
                    content TEXT,
                    compiled INTEGER,
                    coverage REAL,
                    errors TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    round_num INTEGER,
                    type TEXT,
                    path TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    content TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    round_num INTEGER,
                    content TEXT,
                    path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS crashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    round_num INTEGER,
                    test_id TEXT,
                    message TEXT,
                    stack TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    round_num INTEGER,
                    report_type TEXT,
                    path TEXT,
                    content TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            conn.commit()

    def create_run(
        self,
        strategy: str,
        initial_intensity: float,
        target_effectiveness: float,
        baseline_coverage: float,
        model: str,
        endpoint: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO runs (started_at, strategy, initial_intensity, target_effectiveness, baseline_coverage, model, endpoint)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    strategy,
                    initial_intensity,
                    target_effectiveness,
                    baseline_coverage,
                    model,
                    endpoint,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_round(
        self,
        run_id: int,
        round_num: int,
        strategy: str,
        intensity: float,
        compile_rate: float,
        coverage: float,
        effectiveness: float,
        is_best: bool,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO rounds (run_id, round_num, strategy, intensity, compile_rate, coverage, effectiveness, is_best, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    strategy,
                    intensity,
                    compile_rate,
                    coverage,
                    effectiveness,
                    1 if is_best else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_test_cases(
        self,
        run_id: int,
        round_num: int,
        cases: Iterable[tuple],
    ) -> None:
        # cases: (code, compiled, coverage, errors)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO test_cases (run_id, round_num, code, compiled, coverage, errors)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        round_num,
                        code,
                        1 if compiled else 0,
                        coverage,
                        errors,
                    )
                    for code, compiled, coverage, errors in cases
                ],
            )
            conn.commit()

    def add_coverage_snapshot(
        self,
        run_id: int,
        variant: str,
        line_coverage: Optional[float],
        branch_coverage: Optional[float],
        method_coverage: Optional[float],
        compile_rate: Optional[float],
        total_tests: Optional[int],
        successful_tests: Optional[int],
        round_num: Optional[int] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO coverage_snapshots (
                    run_id, round_num, variant, line_coverage, branch_coverage, method_coverage,
                    compile_rate, total_tests, successful_tests, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    variant,
                    line_coverage,
                    branch_coverage,
                    method_coverage,
                    compile_rate,
                    total_tests,
                    successful_tests,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def import_coverage_csv(
        self,
        report_dir: Path,
        run_id: int,
        variant: str,
        round_num: Optional[int] = None,
        compile_rate: Optional[float] = None,
        total_tests: Optional[int] = None,
        successful_tests: Optional[int] = None,
    ) -> Dict[str, float]:
        csv_path = Path(report_dir) / "coverage.csv"
        if not csv_path.exists():
            return {"line_coverage": 0.0, "branch_coverage": 0.0, "method_coverage": 0.0}

        totals = {
            "line_missed": 0,
            "line_covered": 0,
            "branch_missed": 0,
            "branch_covered": 0,
            "method_missed": 0,
            "method_covered": 0,
        }
        rows: list[tuple] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) < 12:
                    continue
                try:
                    package = parts[1]
                    clazz = parts[2]
                    instr_missed = int(parts[3])
                    instr_cov = int(parts[4])
                    branch_missed = int(parts[5])
                    branch_cov = int(parts[6])
                    line_missed = int(parts[7])
                    line_cov = int(parts[8])
                    method_missed = int(parts[9])
                    method_cov = int(parts[10])
                except (ValueError, IndexError):
                    continue

                totals["line_missed"] += line_missed
                totals["line_covered"] += line_cov
                totals["branch_missed"] += branch_missed
                totals["branch_covered"] += branch_cov
                totals["method_missed"] += method_missed
                totals["method_covered"] += method_cov

                rows.append(
                    (
                        run_id,
                        round_num,
                        variant,
                        package,
                        clazz,
                        line_missed,
                        line_cov,
                        branch_missed,
                        branch_cov,
                        method_missed,
                        method_cov,
                        datetime.utcnow().isoformat(),
                    )
                )

        line_total = totals["line_missed"] + totals["line_covered"]
        branch_total = totals["branch_missed"] + totals["branch_covered"]
        method_total = totals["method_missed"] + totals["method_covered"]

        line_cov_pct = (totals["line_covered"] / line_total * 100) if line_total > 0 else 0.0
        branch_cov_pct = (totals["branch_covered"] / branch_total * 100) if branch_total > 0 else 0.0
        method_cov_pct = (totals["method_covered"] / method_total * 100) if method_total > 0 else 0.0

        with self._connect() as conn:
            cur = conn.cursor()
            if rows:
                cur.executemany(
                    """
                    INSERT INTO coverage_details (
                        run_id, round_num, variant, package, clazz,
                        line_missed, line_covered, branch_missed, branch_covered,
                        method_missed, method_covered, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            cur.execute(
                """
                INSERT INTO coverage_snapshots (
                    run_id, round_num, variant, line_coverage, branch_coverage, method_coverage,
                    compile_rate, total_tests, successful_tests, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    variant,
                    line_cov_pct,
                    branch_cov_pct,
                    method_cov_pct,
                    compile_rate,
                    total_tests,
                    successful_tests,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        return {
            "line_coverage": line_cov_pct,
            "branch_coverage": branch_cov_pct,
            "method_coverage": method_cov_pct,
        }

    def add_prompt(
        self,
        run_id: Optional[int],
        round_num: Optional[int],
        strategy: Optional[str],
        prompt_text: str,
        response_text: str,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        latency_ms: Optional[float],
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO prompts (run_id, round_num, strategy, prompt_text, response_text, model, temperature, max_tokens, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    strategy,
                    prompt_text,
                    response_text,
                    model,
                    temperature,
                    max_tokens,
                    latency_ms,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_seed(
        self,
        run_id: Optional[int],
        round_num: Optional[int],
        variant: Optional[str],
        content: str,
        compiled: bool,
        coverage: Optional[float] = None,
        errors: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO seeds (run_id, round_num, variant, content, compiled, coverage, errors, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    variant,
                    content,
                    1 if compiled else 0,
                    coverage,
                    errors,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_artifact(
        self,
        run_id: Optional[int],
        round_num: Optional[int],
        type: str,
        path: Optional[str] = None,
        sha256: Optional[str] = None,
        size_bytes: Optional[int] = None,
        content: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO artifacts (run_id, round_num, type, path, sha256, size_bytes, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    type,
                    path,
                    sha256,
                    size_bytes,
                    content,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_log(
        self,
        run_id: Optional[int],
        round_num: Optional[int],
        content: Optional[str],
        path: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO logs (run_id, round_num, content, path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    content,
                    path,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_crash(
        self,
        run_id: Optional[int],
        round_num: Optional[int],
        test_id: Optional[str],
        message: Optional[str],
        stack: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO crashes (run_id, round_num, test_id, message, stack, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    test_id,
                    message,
                    stack,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def add_report(
        self,
        run_id: Optional[int],
        round_num: Optional[int],
        report_type: str,
        path: Optional[str],
        content: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO reports (run_id, round_num, report_type, path, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    round_num,
                    report_type,
                    path,
                    content,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
