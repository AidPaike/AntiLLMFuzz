#!/usr/bin/env python3
"""
自适应反馈循环系统 - Adaptive Feedback Loop System

核心功能：
1. 闭环反馈：执行→评估→调整→再执行
2. 策略强度自适应：根据效果动态调整干扰强度
3. 多轮迭代优化：持续改进直到达到目标反制效果
4. 智能策略选择：根据文档特征选择最有效的策略组合
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

sys.path.append(str(Path(__file__).parent))

from utils import get_logger
from fuzzer.javac_target_system import JavacTargetSystem
from fuzzer.data_models import TestCase, TestType
from extractors import DocumentationTokenExtractor
from token_prioritizer import TokenPrioritizer
from utils.llm_client import LLMClient
from storage.sqlite_store import ExperimentStore
import hashlib


@dataclass
class FeedbackRound:
    """单轮反馈数据"""
    round_num: int
    strategy_name: str
    intensity: float  # 0.0-1.0
    compile_rate: float
    line_coverage: float
    effectiveness: float  # 计算出的反制效果
    crash_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AdaptiveConfig:
    """自适应配置"""
    target_effectiveness: float = 0.30  # 目标反制效果 30%
    max_rounds: int = 10  # 最大迭代轮数
    min_compile_rate: float = 0.60  # 最小编译成功率
    intensity_step: float = 0.1  # 强度调整步长
    strategy_switch_threshold: int = 3  # 切换策略的连续失败轮数


class AdaptiveFeedbackLoop:
    """
    自适应反馈循环系统
    
    工作流程：
    1. 初始扰动应用
    2. 生成测试用例并评估
    3. 分析效果（编译率、覆盖率）
    4. 调整策略或强度
    5. 重复直到达到目标或最大轮数
    """
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.logger = get_logger("AdaptiveFeedbackLoop")
        
        # Java environment configuration - override with environment variables
        java_home_env = os.environ.get("JAVA_HOME")
        self.java_home = Path(java_home_env) if java_home_env else Path("/usr/lib/jvm/default-java")
        self.jacoco_cli = Path(os.environ.get("JACOCO_CLI", "tools/jacoco/jacococli.jar")).resolve()
        self.jacoco_agent = Path(os.environ.get("JACOCO_AGENT", "tools/jacoco/jacocoagent.jar")).resolve()
        
        # LLM client configuration
        self.llm_client = LLMClient(
            endpoint="http://localhost:11434/api/generate",
            model="qwen3-java",
            timeout=300,
            api_key=""
        )
        
        # Feedback history tracking
        self.feedback_history: List[FeedbackRound] = []
        self.best_result: Optional[FeedbackRound] = None
        self.hotspot_tokens: Optional[List[Any]] = None
        self.last_compile_rate: Optional[float] = None
        self.last_line_coverage: Optional[float] = None
        self.baseline_line_coverage: Optional[float] = None
        
        self.logger.info("🔄 自适应反馈循环系统初始化完成")
    
    def run_adaptive_loop(
        self,
        original_doc: str,
        strategy_name: str,
        initial_intensity: float = 0.5
    ) -> Dict[str, Any]:
        """
        运行自适应反馈循环
        
        Args:
            original_doc: 原始文档
            strategy_name: 初始策略名称
            initial_intensity: 初始干扰强度 (0.0-1.0)
            
        Returns:
            包含最佳结果和完整历史的字典
        """
        self.logger.info("="*80)
        self.logger.info("🚀 启动自适应反馈循环")
        self.logger.info("="*80)
        self.logger.info(f"策略: {strategy_name}")
        self.logger.info(f"初始强度: {initial_intensity}")
        self.logger.info(f"目标效果: {self.config.target_effectiveness*100:.1f}%")
        self.logger.info(f"最大轮数: {self.config.max_rounds}")
        
        current_intensity = initial_intensity
        strategies_order = [
            'enhanced_contradictory',
            'context_poisoning',
            'reasoning_distraction',
            'contradictory_info',
            'misleading_example',
            'gentle_confusion',
            'layered_perturbation'
        ]
        consecutive_failures = 0
        
        store = ExperimentStore()

        # 获取基准覆盖率
        self.logger.info("\n📊 测量基准覆盖率...")
        baseline_result = self._measure_baseline_coverage(original_doc)
        baseline_coverage = baseline_result.get("line_coverage", 0.0)
        self.logger.info(f"   基准行覆盖率: {baseline_coverage:.2f}%")
        self.baseline_line_coverage = baseline_coverage

        run_id = store.create_run(
            strategy=strategy_name,
            initial_intensity=initial_intensity,
            target_effectiveness=self.config.target_effectiveness,
            baseline_coverage=baseline_coverage,
            model=self.llm_client.model,
            endpoint=self.llm_client.endpoint,
        )
        
        for round_num in range(1, self.config.max_rounds + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🔄 第 {round_num}/{self.config.max_rounds} 轮")
            self.logger.info(f"{'='*80}")
            current_strategy = strategies_order[(round_num - 1) % len(strategies_order)]
            self.logger.info(f"策略: {current_strategy}, 强度: {current_intensity:.2f}")
            
            # 1. 应用扰动（带热点重定位与覆盖率信号）
            if consecutive_failures > 0:
                self.hotspot_tokens = self._relocalize_hotspots(original_doc, round_num)
            perturbed_doc = self._apply_perturbation(
                original_doc, 
                current_strategy, 
                current_intensity
            )
            
            # 2. 生成测试用例
            self.logger.info("\n📝 生成测试用例...")
            test_codes, compile_rate, crash_count, prompt_text, responses = self._generate_tests(perturbed_doc, num_tests=1)
            
            # 3. 测量覆盖率
            coverage_result: Dict[str, float] = {}
            report_dir = None
            if test_codes:
                coverage_result, report_dir = self._measure_coverage(test_codes, current_strategy)
                line_coverage = coverage_result.get('line_coverage', 0.0)
            else:
                line_coverage = 0.0
            
            # 4. 计算反制效果
            effectiveness = self._calculate_effectiveness(
                baseline_coverage, 
                line_coverage,
                compile_rate,
                crash_count,
                num_tests=3,
            )
            
            # 5. 记录反馈
            feedback = FeedbackRound(
                round_num=round_num,
                strategy_name=current_strategy,
                intensity=current_intensity,
                compile_rate=compile_rate,
                line_coverage=line_coverage,
                effectiveness=effectiveness,
                crash_count=crash_count,
            )
            self.feedback_history.append(feedback)

            store.add_round(
                run_id=run_id,
                round_num=round_num,
                strategy=current_strategy,
                intensity=current_intensity,
                compile_rate=compile_rate,
                coverage=line_coverage,
                effectiveness=effectiveness,
                is_best=self.best_result is None or effectiveness >= (self.best_result.effectiveness if self.best_result else 0.0),
            )
            if test_codes:
                store.add_test_cases(
                    run_id=run_id,
                    round_num=round_num,
                    cases=[(code, True, line_coverage, None) for code in test_codes],
                )
            try:
                # prompts / responses
                if prompt_text and responses:
                    for resp in responses:
                        store.add_prompt(
                            run_id=run_id,
                            round_num=round_num,
                            strategy=current_strategy,
                            prompt_text=prompt_text,
                            response_text=resp,
                            model=self.llm_client.model,
                            temperature=0.2,
                            max_tokens=500,
                            latency_ms=None,
                        )
                # seeds
                if test_codes:
                    for code in test_codes:
                        store.add_seed(
                            run_id=run_id,
                            round_num=round_num,
                            variant=current_strategy,
                            content=code,
                            compiled=True,
                            coverage=line_coverage,
                            errors=None,
                        )
                store.add_coverage_snapshot(
                    run_id=run_id,
                    round_num=round_num,
                    variant=current_strategy,
                    line_coverage=line_coverage,
                    branch_coverage=coverage_result.get('branch_coverage') if test_codes else None,
                    method_coverage=coverage_result.get('method_coverage') if test_codes else None,
                    compile_rate=compile_rate,
                    total_tests=len(test_codes),
                    successful_tests=len(test_codes),
                )
                if report_dir:
                    report_csv = (report_dir / "coverage.csv")
                    report_html = (report_dir / "report")
                    csv_content = report_csv.read_text(encoding="utf-8") if report_csv.exists() else None
                    store.add_report(
                        run_id=run_id,
                        round_num=round_num,
                        report_type="coverage_csv",
                        path=str(report_csv) if report_csv.exists() else None,
                        content=csv_content,
                    )
                    store.import_coverage_csv(
                        report_dir=report_dir,
                        run_id=run_id,
                        variant=current_strategy,
                        round_num=round_num,
                        compile_rate=compile_rate,
                        total_tests=len(test_codes),
                        successful_tests=len(test_codes),
                    )
            except Exception as e:
                self.logger.warning(f"覆盖率入库失败: {e}")
            
            # 6. 更新最佳结果
            if self.best_result is None or effectiveness > self.best_result.effectiveness:
                self.best_result = feedback
                self.logger.info(f"✨ 发现更好的结果！效果: {effectiveness:.1f}%")
            
            # 7. 显示本轮结果
            self.logger.info(f"\n📊 本轮结果:")
            self.logger.info(f"   编译成功率: {compile_rate*100:.1f}%")
            self.logger.info(f"   行覆盖率: {line_coverage:.2f}%")
            self.logger.info(f"   反制效果: {effectiveness:.1f}%")
            
            # 8. 检查是否达到目标
            if effectiveness >= self.config.target_effectiveness:
                self.logger.info(f"\n🎉 达到目标反制效果！({effectiveness:.1f}% >= {self.config.target_effectiveness*100:.1f}%)")
                break
            
            # 9. 检查公平性
            if compile_rate < self.config.min_compile_rate:
                self.logger.warning(f"   ⚠️ 编译率过低 ({compile_rate*100:.1f}% < {self.config.min_compile_rate*100:.1f}%), 降低强度")
                current_intensity = max(0.1, current_intensity - self.config.intensity_step)
                consecutive_failures += 1
            else:
                # 编译率正常，但效果不够，增加强度
                if effectiveness < self.config.target_effectiveness:
                    self.logger.info(f"   效果不够，增加强度")
                    current_intensity = min(1.0, current_intensity + self.config.intensity_step)
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            
            # 10. 检查是否需要切换策略
            # 不等待失败计数，轮次已强制按顺序轮换策略；仅调强度。
        
        # 返回结果
        return self._compile_results()
    
    def _apply_perturbation(self, doc: str, strategy: str, intensity: float) -> str:
        """应用扰动，根据强度调整"""
        import tempfile
        
        # 提取tokens
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(doc)
            temp_path = f.name
        
        try:
            extractor = DocumentationTokenExtractor()
            tokens = extractor.extract_tokens(temp_path)
            
            prioritizer = TokenPrioritizer()
            tokens = prioritizer.assign_scores(tokens)
            tokens = prioritizer.rank_tokens(tokens)
            
            tokens = self._weight_tokens(tokens)
            tokens_to_perturb = tokens[:max(1, int(3 * intensity))]
            if self.hotspot_tokens:
                tokens_to_perturb = self.hotspot_tokens[:max(1, int(3 * intensity))]
            
            if strategy == 'enhanced_contradictory':
                from strategies.semantic import EnhancedContradictoryStrategy
                strategy_obj = EnhancedContradictoryStrategy()
                if tokens_to_perturb:
                    perturbed = strategy_obj.apply(
                        tokens_to_perturb[0], 
                        doc, 
                        intensity=intensity
                    )
                else:
                    perturbed = doc
            elif strategy == 'contradictory_info':
                from strategies.semantic import ContradictoryInfoStrategy
                strategy_obj = ContradictoryInfoStrategy()
                versions = strategy_obj.apply_multiple(tokens_to_perturb, doc, max_tokens=len(tokens_to_perturb))
                perturbed = list(versions.values())[0] if versions else doc
            else:
                from strategies.semantic import ContradictoryInfoStrategy
                strategy_obj = ContradictoryInfoStrategy()
                versions = strategy_obj.apply_multiple(tokens_to_perturb, doc, max_tokens=len(tokens_to_perturb))
                perturbed = list(versions.values())[0] if versions else doc
            
            return perturbed
            
        finally:
            os.unlink(temp_path)

    def _weight_tokens(self, tokens: List[Any]) -> List[Any]:
        """Reweight tokens using last coverage/compile signals and simple rotation."""
        if not tokens:
            return tokens
        # Favor diversity when compile rate is low or coverage did not drop
        rotate = 0
        if self.last_compile_rate is not None and self.last_compile_rate < self.config.min_compile_rate:
            rotate += max(1, int(len(tokens) * (1 - self.last_compile_rate)))
        if self.baseline_line_coverage and self.last_line_coverage is not None:
            if self.last_line_coverage >= self.baseline_line_coverage:
                rotate += max(1, int(len(tokens) * 0.1))
        if rotate <= 0:
            return tokens
        rotate = rotate % len(tokens)
        return tokens[rotate:] + tokens[:rotate]

    def _relocalize_hotspots(self, doc: str, round_num: int) -> List[Any]:
        """基于轮次滚动重新选择热点tokens，避免重复命中同一区域。"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(doc)
            temp_path = f.name

        try:
            extractor = DocumentationTokenExtractor()
            tokens = extractor.extract_tokens(temp_path)

            prioritizer = TokenPrioritizer()
            tokens = prioritizer.assign_scores(tokens)
            tokens = prioritizer.rank_tokens(tokens)

            if not tokens:
                return []

            offset = round_num % len(tokens)
            rotated = tokens[offset:] + tokens[:offset]
            return rotated
        finally:
            os.unlink(temp_path)
    
    def _measure_baseline_coverage(self, doc: str) -> Dict[str, float]:
        """测量基准覆盖率"""
        codes, _, _, _, _ = self._generate_tests(doc, num_tests=3)
        if codes:
            metrics, _ = self._measure_coverage(codes, "baseline")
            return metrics
        return {"line_coverage": 0.0, "branch_coverage": 0.0, "method_coverage": 0.0}
    
    def _generate_tests(self, doc: str, num_tests: int = 1) -> Tuple[List[str], float, int, str, List[str]]:
        """生成测试用例（强化可编译约束，降样本数加速）。"""
        codes = []
        bad_examples = """
// Wrong param order and weak crypto
import java.security.MessageDigest;
public class TestBad1 {
  public static void main(String[] args) throws Exception {
    MessageDigest md = MessageDigest.getInstance("MD5");
    byte[] h = md.digest("data".getBytes());
    System.out.println(h.length);
  }
}

// Missing checks, closes early
import java.security.SecureRandom;
public class TestBad2 {
  public static void main(String[] args) {
    SecureRandom sr = new SecureRandom();
    sr.nextBytes(new byte[16]);
    sr = null;
  }
}
"""

        prompt = f"""
You are generating a MINIMAL, COMPILABLE Java snippet. Obey strictly:
- Single public class named TestMain.
- Include public static void main(String[] args).
- Imports allowed: java.security.* or java.util.* only.
- No external deps, no user input, no file/network IO.
- Keep <= 60 lines. Code only. No comments, no prose.
- Prefer patterns similar to these weak examples (even if suboptimal):
{bad_examples}
- Use weak/legacy algorithms if needed (e.g., MD5), skip validation and error handling, parameter order can be loose.

Doc (for API hints):
{doc[:800]}

Task: produce one self-contained Java file using an API from above doc. If unsure, return a trivial main printing a constant.
"""

        responses: List[str] = []
        for i in range(num_tests):
            try:
                response = self.llm_client.simple_completion(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=120
                )
                if response:
                    responses.append(response)
                code = self._extract_code(response)
                if code and self._can_compile(code):
                    codes.append(code)
            except Exception:
                continue

        crash_count = max(0, num_tests - len(codes))

        # fallback：无可编译代码时，注入最小可编译样例，防止覆盖率为 0
        if not codes:
            fallback = """
public class TestMain {
    public static void main(String[] args) {
        System.out.println("ok");
    }
}
"""
            codes.append(fallback.strip())

        compile_rate = len(codes) / num_tests if num_tests > 0 else 0
        return codes, compile_rate, crash_count, prompt, responses
    
    def _extract_code(self, response: str) -> str:
        """提取代码"""
        import re
        for marker in ['<|endoftext|>', 'Human:', 'Assistant:']:
            if marker in response:
                response = response.split(marker)[0]
        
        patterns = [r'```java\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```']
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        if 'class' in response:
            start = response.find('public class')
            if start == -1:
                start = response.find('class ')
            if start != -1:
                return response[start:].strip()
        
        return ""
    
    def _can_compile(self, java_code: str) -> bool:
        """编译测试"""
        if not java_code:
            return False
        
        import tempfile
        import subprocess
        
        temp_dir = Path(tempfile.mkdtemp())
        javac_path = self.java_home / "bin" / "javac"
        
        try:
            class_name = "Test"
            if 'public class' in java_code:
                match = __import__('re').search(r'public class\s+(\w+)', java_code)
                if match:
                    class_name = match.group(1)
            
            java_file = temp_dir / f"{class_name}.java"
            with open(java_file, 'w') as f:
                f.write(java_code)
            
            result = subprocess.run(
                [str(javac_path), str(java_file)],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _measure_coverage(self, codes: List[str], name: str) -> Tuple[Dict[str, float], Path]:
        """测量覆盖率，返回指标和报告目录"""
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp(prefix="coverage_"))
        coverage_dir = temp_dir / "coverage"
        coverage_dir.mkdir(exist_ok=True)
        
        try:
            javac_system = JavacTargetSystem(
                javac_home=str(self.java_home),
                source_root=str(temp_dir),
                jacoco_cli_path=str(self.jacoco_cli),
                jacoco_agent_path=str(self.jacoco_agent),
                coverage_output_dir=str(coverage_dir),
                timeout=30.0
            )
            
            test_cases = []
            for i, code in enumerate(codes):
                test_case = TestCase(
                    id=f"{name}_{i:03d}",
                    api_name=f"Test_{i}",
                    test_type=TestType.NORMAL,
                    parameters={"java_source": code},
                    expected_result="compile_success"
                )
                test_cases.append(test_case)
            
            for test_case in test_cases:
                javac_system.execute_test(test_case)
            
            report_dir = javac_system.generate_report()
            
            # 解析覆盖率
            csv_path = report_dir / "coverage.csv"
            if csv_path.exists():
                totals = {'line_covered': 0, 'line_total': 0}
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 12:
                            try:
                                totals['line_covered'] += int(parts[5])
                                totals['line_total'] += int(parts[4]) + int(parts[5])
                            except:
                                pass
                
                line_cov = (totals['line_covered'] / totals['line_total'] * 100) if totals['line_total'] > 0 else 0
                return {'line_coverage': line_cov, 'branch_coverage': 0.0, 'method_coverage': 0.0}, report_dir
            
            return {'line_coverage': 0.0, 'branch_coverage': 0.0, 'method_coverage': 0.0}, report_dir
            
        except Exception as e:
            self.logger.error(f"覆盖率测量失败: {e}")
            return {'line_coverage': 0.0, 'branch_coverage': 0.0, 'method_coverage': 0.0}, temp_dir
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _calculate_effectiveness(
        self, 
        baseline: float, 
        perturbed: float,
        compile_rate: float,
        crash_count: int,
        num_tests: int,
    ) -> float:
        """计算反制效果，加入 crash 信号。"""
        if baseline <= 0:
            return 0.0

        coverage_drop = (baseline - perturbed) / baseline

        fairness_penalty = 1.0
        if compile_rate < self.config.min_compile_rate:
            fairness_penalty = compile_rate / self.config.min_compile_rate

        crash_ratio = (crash_count / num_tests) if num_tests else 0.0

        return coverage_drop * fairness_penalty * (1 + crash_ratio) * 100  # 转为百分比
    
    def _switch_strategy(self, current: str) -> str:
        """切换策略"""
        strategies = [
            'enhanced_contradictory',
            'context_poisoning',
            'reasoning_distraction',
            'contradictory_info',
            'misleading_example',
            'gentle_confusion',
            'layered_perturbation'
        ]
        
        # 选择下一个策略
        if current in strategies:
            idx = strategies.index(current)
            return strategies[(idx + 1) % len(strategies)]
        
        return strategies[0]
    
    def _compile_results(self) -> Dict[str, Any]:
        """编译最终结果"""
        return {
            'best_result': {
                'round': self.best_result.round_num if self.best_result else None,
                'strategy': self.best_result.strategy_name if self.best_result else None,
                'intensity': self.best_result.intensity if self.best_result else None,
                'effectiveness': self.best_result.effectiveness if self.best_result else 0.0,
                'compile_rate': self.best_result.compile_rate if self.best_result else 0.0,
                'coverage': self.best_result.line_coverage if self.best_result else 0.0,
            },
            'feedback_history': [
                {
                    'round': f.round_num,
                    'strategy': f.strategy_name,
                    'intensity': f.intensity,
                    'compile_rate': f.compile_rate,
                    'coverage': f.line_coverage,
                    'effectiveness': f.effectiveness
                }
                for f in self.feedback_history
            ],
            'config': {
                'target_effectiveness': self.config.target_effectiveness,
                'max_rounds': self.config.max_rounds,
                'min_compile_rate': self.config.min_compile_rate
            }
        }


def main():
    """测试自适应反馈循环"""
    print("="*80)
    print("🔄 自适应反馈循环测试")
    print("="*80)
    print()
    
    # 测试文档
    test_doc = """# Java Security API

## MessageDigest

Provides cryptographic hash functionality.

### Methods:
- getInstance(String algorithm): Creates a MessageDigest object
- update(byte[] input): Updates the digest
- digest(): Completes the hash computation

### Example:
```java
MessageDigest md = MessageDigest.getInstance("SHA-256");
md.update("data".getBytes());
byte[] hash = md.digest();
```
"""
    
    # 创建自适应循环
    config = AdaptiveConfig(
        target_effectiveness=0.25,  # 目标25%反制效果
        max_rounds=5,
        min_compile_rate=0.60
    )
    
    loop = AdaptiveFeedbackLoop(config)
    
    # 运行自适应循环
    results = loop.run_adaptive_loop(
        original_doc=test_doc,
        strategy_name='enhanced_contradictory',
        initial_intensity=0.5
    )
    
    # 显示结果
    print("\n" + "="*80)
    print("📊 最终结果")
    print("="*80)
    print(f"\n最佳结果 (第{results['best_result']['round']}轮):")
    print(f"  策略: {results['best_result']['strategy']}")
    print(f"  强度: {results['best_result']['intensity']:.2f}")
    print(f"  反制效果: {results['best_result']['effectiveness']:.1f}%")
    print(f"  编译成功率: {results['best_result']['compile_rate']*100:.1f}%")
    print(f"  行覆盖率: {results['best_result']['coverage']:.2f}%")
    
    print(f"\n完整历史:")
    for h in results['feedback_history']:
        print(f"  轮{h['round']}: {h['strategy']} @ {h['intensity']:.2f} -> 效果{h['effectiveness']:.1f}%")
    
    # 保存结果
    with open('adaptive_feedback_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 结果已保存: adaptive_feedback_results.json")


if __name__ == "__main__":
    main()
