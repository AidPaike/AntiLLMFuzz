
import os
import time
import json
import shutil
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from src.fuzzer.base_interfaces import BaseTargetSystem
from src.fuzzer.data_models import TestCase, ExecutionResult, CoverageMetrics
from src.utils.logger import get_logger


class JavacTargetSystem(BaseTargetSystem):
    """Execute test cases against real javac with JaCoCo coverage."""

    def __init__(
        self,
        javac_home: str,
        source_root: str,
        jacoco_cli_path: str,
        jacoco_agent_path: str,
        coverage_output_dir: str,
        coverage_scope: str = "javac",
        timeout: float = 10.0,
    ) -> None:
        self.logger = get_logger("JavacTargetSystem")
        self.javac_home = Path(javac_home).resolve()
        self.source_root = Path(source_root).resolve()
        self.jacoco_cli_path = Path(jacoco_cli_path).resolve()
        self.jacoco_agent_path = Path(jacoco_agent_path).resolve()
        self.coverage_output_dir = Path(coverage_output_dir).resolve()
        self.coverage_scope = coverage_scope
        self.timeout = timeout
        self.report_output_dir: Optional[Path] = None

        self.javac_bin = self.javac_home / "bin" / "javac"
        self.java_bin = self.javac_home / "bin" / "java"
        self.coverage_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.javac_bin.exists():
            raise FileNotFoundError(f"javac not found at {self.javac_bin}")
        if not self.java_bin.exists():
            raise FileNotFoundError(f"java not found at {self.java_bin}")
        if not self.jacoco_cli_path.exists():
            raise FileNotFoundError(f"JaCoCo CLI not found at {self.jacoco_cli_path}")
        if not self.jacoco_agent_path.exists():
            raise FileNotFoundError(f"JaCoCo agent not found at {self.jacoco_agent_path}")

        self.exec_file = self.coverage_output_dir / "jacoco.exec"

    def reset_state(self) -> None:
        if self.exec_file.exists():
            self.exec_file.unlink()

    def get_coverage_info(self) -> CoverageMetrics:
        if not self.exec_file.exists():
            return CoverageMetrics()
        return CoverageMetrics()

    def execute_test(self, test_case: TestCase) -> ExecutionResult:
        start = time.time()
        work_dir, source_file = self._prepare_test_source(test_case)
        try:
            success, stderr_text = self._run_javac(work_dir, source_file)
            elapsed = time.time() - start
            return ExecutionResult(
                test_case_id=test_case.id,
                success=success,
                response=None,
                execution_time=elapsed,
                coverage_delta=None,
                errors=[] if success else [stderr_text] if stderr_text else ["javac failed"],
            )
        except Exception as exc:
            elapsed = time.time() - start
            return ExecutionResult(
                test_case_id=test_case.id,
                success=False,
                response=None,
                execution_time=elapsed,
                errors=[str(exc)],
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def get_call_count(self) -> int:
        return 0

    def generate_report(self, output_dir: Optional[Path] = None) -> Path:
        report_dir = output_dir or (self.coverage_output_dir / "report")
        report_dir.mkdir(parents=True, exist_ok=True)
        self.report_output_dir = report_dir
        cmd = [
            str(self.java_bin),
            "-jar",
            str(self.jacoco_cli_path),
            "report",
            str(self.exec_file),
            "--classfiles",
            str(self._javac_classes_dir()),
            "--sourcefiles",
            str(self._javac_source_dir()),
            "--html",
            str(report_dir),
            "--csv",
            str(report_dir / "coverage.csv"),
        ]
        self.logger.info(f"Generating JaCoCo report: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        return report_dir

    def _javac_source_dir(self) -> Path:
        if self.coverage_scope == "all":
            return self.source_root
        return self.source_root / "src" / "jdk.compiler"

    def _javac_classes_dir(self) -> Path:
        build_dir = self.source_root / "build" / "linux-x86_64-server-release" / "jdk" / "modules"
        if not build_dir.exists():
            alt_build = self.javac_home.parent / "build" / "linux-x86_64-server-release" / "jdk" / "modules"
            if alt_build.exists():
                build_dir = alt_build
        if self.coverage_scope == "all":
            return build_dir
        return build_dir / "jdk.compiler"

    def _prepare_test_source(self, test_case: TestCase) -> Tuple[Path, Path]:
        safe_id = re.sub(r"[^A-Za-z0-9_]", "_", test_case.id)
        work_dir = Path(self.coverage_output_dir) / f"case_{safe_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        java_source = None
        if isinstance(test_case.parameters, dict):
            java_source = test_case.parameters.get("java_source")

        if isinstance(java_source, str) and java_source.strip():
            class_name = self._extract_public_class_name(java_source) or f"Test{safe_id}"
            source_file = work_dir / f"{class_name}.java"
            source_file.write_text(java_source, encoding="utf-8")
        else:
            class_name = f"Test{safe_id}"
            source_file = work_dir / f"{class_name}.java"
            source_file.write_text(self._build_java_source(class_name, test_case), encoding="utf-8")
        return work_dir, source_file

    def _build_java_source(self, class_name: str, test_case: TestCase) -> str:
        params = json.dumps(test_case.parameters, ensure_ascii=False)
        escaped = params.replace("\\", "\\\\").replace('"', '\\"')
        return (
            "public class " + class_name + " {\n"
            "  public static void main(String[] args) {\n"
            f"    String params = \"{escaped}\";\n"
            "    if (params.length() > 0) {\n"
            "      System.out.print(\"\");\n"
            "    }\n"
            "  }\n"
            "}\n"
        )

    def _extract_public_class_name(self, java_source: str) -> Optional[str]:
        match = re.search(r"public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", java_source)
        if not match:
            return None
        return match.group(1)

    def _run_javac(self, work_dir: Path, source_file: Path) -> Tuple[bool, str]:
        exec_path = self.exec_file.resolve()
        jacoco_arg = f"-J-javaagent:{self.jacoco_agent_path}=destfile={exec_path},append=true"
        cmd = [
            str(self.javac_bin),
            jacoco_arg,
            "--enable-preview",
            "--release",
            "23",
            str(source_file),
        ]
        self.logger.debug(f"Running javac: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            self.logger.debug(result.stderr.decode(errors="ignore"))
        return result.returncode == 0, result.stderr.decode(errors="ignore")
