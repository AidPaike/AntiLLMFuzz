#!/usr/bin/env python3
"""反制效果演示脚本 - 对比原始文档和扰动文档的LLM模糊测试效果"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm_client_manager import get_llm_manager
from src.utils.error_handler import ProgressReporter, print_startup_banner


class AntiLLMFuzzDemo:
    """反制LLM模糊测试演示器"""
    
    def __init__(self):
        self.llm_manager = get_llm_manager()
        self.results = {}
    
    def run_demo(self, original_file: str, perturbed_files: List[str]):
        """运行完整的反制效果演示"""
        print_startup_banner()
        print("🎯 反制LLM模糊测试效果演示")
        print("="*80)
        
        reporter = ProgressReporter(4, "反制效果测试")
        
        # 1. 测试原始文档
        reporter.start_step("测试原始文档的LLM理解能力")
        original_result = self._test_document_understanding(original_file, "原始文档")
        reporter.complete_step(True, f"原始文档测试完成 - 理解度: {original_result['understanding_score']:.1f}%")
        
        # 2. 测试扰动文档
        reporter.start_step("测试扰动文档的LLM理解能力")
        perturbed_results = []
        for i, perturbed_file in enumerate(perturbed_files[:3]):  # 测试前3个
            result = self._test_document_understanding(perturbed_file, f"扰动文档{i+1}")
            perturbed_results.append(result)
        
        avg_perturbed_score = sum(r['understanding_score'] for r in perturbed_results) / len(perturbed_results)
        reporter.complete_step(True, f"扰动文档测试完成 - 平均理解度: {avg_perturbed_score:.1f}%")
        
        # 3. 生成测试用例对比
        reporter.start_step("对比测试用例生成能力")
        test_case_comparison = self._compare_test_generation(original_file, perturbed_files[0])
        reporter.complete_step(True, "测试用例生成对比完成")
        
        # 4. 分析反制效果
        reporter.start_step("分析反制效果")
        effectiveness = self._analyze_effectiveness(original_result, perturbed_results, test_case_comparison)
        reporter.complete_step(True, f"反制效果分析完成 - 干扰度: {effectiveness['disruption_rate']:.1f}%")
        
        reporter.finish(True)
        
        # 显示详细结果
        self._display_results(original_result, perturbed_results, test_case_comparison, effectiveness)
        
        return effectiveness
    
    def _test_document_understanding(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """测试LLM对文档的理解能力"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 测试1: 关键信息提取
            extraction_prompt = f"""
请分析以下Java API文档，提取关键信息：

{content[:2000]}...

请回答：
1. 这个文档描述了哪些主要的Java安全相关API？
2. 列出所有提到的安全相关包名
3. 识别出认证和授权相关的功能

请用JSON格式回答。
"""
            
            extraction_response = self.llm_manager.generate(extraction_prompt)
            
            # 测试2: 代码生成
            generation_prompt = f"""
基于以下文档内容，生成一个使用Java安全API的示例代码：

{content[:1500]}...

要求：
1. 使用文档中提到的安全相关API
2. 包含认证或加密功能
3. 代码要完整可编译
"""
            
            generation_response = self.llm_manager.generate(generation_prompt)
            
            # 评估理解质量
            understanding_score = self._evaluate_understanding(
                content, extraction_response, generation_response
            )
            
            return {
                'doc_type': doc_type,
                'file_path': file_path,
                'understanding_score': understanding_score,
                'extraction_response': extraction_response,
                'generation_response': generation_response,
                'content_length': len(content)
            }
            
        except Exception as e:
            print(f"   ⚠️ {doc_type}测试失败: {e}")
            return {
                'doc_type': doc_type,
                'file_path': file_path,
                'understanding_score': 0.0,
                'error': str(e)
            }
    
    def _evaluate_understanding(self, original_content: str, extraction: str, generation: str) -> float:
        """评估LLM理解质量"""
        score = 0.0
        
        # 检查关键词识别 (40分)
        security_keywords = [
            'java.security', 'authentication', 'authorization', 'jgss', 'sasl',
            'security', 'credential', 'provider', 'GSS-API', 'SASL'
        ]
        
        found_keywords = 0
        for keyword in security_keywords:
            if keyword.lower() in extraction.lower():
                found_keywords += 1
        
        score += (found_keywords / len(security_keywords)) * 40
        
        # 检查代码生成质量 (30分)
        code_indicators = ['import', 'class', 'public', 'java.security', 'new ']
        code_quality = 0
        for indicator in code_indicators:
            if indicator in generation:
                code_quality += 1
        
        score += (code_quality / len(code_indicators)) * 30
        
        # 检查响应完整性 (30分)
        if len(extraction) > 50:  # 有实质性回答
            score += 15
        if len(generation) > 100:  # 生成了代码
            score += 15
        
        return min(score, 100.0)
    
    def _compare_test_generation(self, original_file: str, perturbed_file: str) -> Dict[str, Any]:
        """对比测试用例生成能力"""
        try:
            # 读取文档
            with open(original_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            with open(perturbed_file, 'r', encoding='utf-8') as f:
                perturbed_content = f.read()
            
            # 生成测试用例的提示词
            test_prompt_template = """
基于以下Java API文档，生成3个安全测试用例：

{content}

要求：
1. 测试认证功能
2. 测试授权检查
3. 测试输入验证

每个测试用例包含：测试目标、输入数据、期望结果
"""
            
            # 对原始文档生成测试用例
            original_tests = self.llm_manager.generate(
                test_prompt_template.format(content=original_content[:2000])
            )
            
            # 对扰动文档生成测试用例
            perturbed_tests = self.llm_manager.generate(
                test_prompt_template.format(content=perturbed_content[:2000])
            )
            
            # 分析差异
            return {
                'original_tests': original_tests,
                'perturbed_tests': perturbed_tests,
                'original_length': len(original_tests),
                'perturbed_length': len(perturbed_tests),
                'quality_degradation': self._calculate_test_quality_degradation(
                    original_tests, perturbed_tests
                )
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_test_quality_degradation(self, original: str, perturbed: str) -> float:
        """计算测试质量下降程度"""
        # 检查测试用例的关键元素
        test_elements = ['测试', 'test', '验证', 'assert', '输入', '期望', '结果']
        
        original_score = sum(1 for elem in test_elements if elem in original.lower())
        perturbed_score = sum(1 for elem in test_elements if elem in perturbed.lower())
        
        if original_score == 0:
            return 0.0
        
        degradation = max(0, (original_score - perturbed_score) / original_score * 100)
        return degradation
    
    def _analyze_effectiveness(self, original: Dict, perturbed_list: List[Dict], 
                             test_comparison: Dict) -> Dict[str, Any]:
        """分析反制效果"""
        # 计算理解能力下降
        original_score = original.get('understanding_score', 0)
        avg_perturbed_score = sum(p.get('understanding_score', 0) for p in perturbed_list) / len(perturbed_list)
        
        understanding_degradation = max(0, (original_score - avg_perturbed_score) / original_score * 100) if original_score > 0 else 0
        
        # 计算测试生成质量下降
        test_degradation = test_comparison.get('quality_degradation', 0)
        
        # 综合干扰度
        disruption_rate = (understanding_degradation + test_degradation) / 2
        
        return {
            'disruption_rate': disruption_rate,
            'understanding_degradation': understanding_degradation,
            'test_generation_degradation': test_degradation,
            'original_understanding': original_score,
            'perturbed_understanding': avg_perturbed_score,
            'effectiveness_level': self._get_effectiveness_level(disruption_rate)
        }
    
    def _get_effectiveness_level(self, disruption_rate: float) -> str:
        """获取效果等级"""
        if disruption_rate >= 70:
            return "🔥 极高 - 严重干扰LLM理解"
        elif disruption_rate >= 50:
            return "🎯 高 - 显著影响LLM性能"
        elif disruption_rate >= 30:
            return "⚡ 中等 - 一定程度干扰"
        elif disruption_rate >= 10:
            return "💡 轻微 - 少量影响"
        else:
            return "😐 无效 - 几乎无影响"
    
    def _display_results(self, original: Dict, perturbed_list: List[Dict], 
                        test_comparison: Dict, effectiveness: Dict):
        """显示详细结果"""
        print("\n" + "="*80)
        print("🎯 反制效果分析报告")
        print("="*80)
        
        # 1. LLM理解能力对比
        print("\n📊 LLM文档理解能力对比:")
        print(f"   原始文档理解度: {original.get('understanding_score', 0):.1f}%")
        
        for i, result in enumerate(perturbed_list):
            score = result.get('understanding_score', 0)
            degradation = max(0, original.get('understanding_score', 0) - score)
            print(f"   扰动文档{i+1}理解度: {score:.1f}% (下降 {degradation:.1f}%)")
        
        # 2. 测试生成能力对比
        print("\n🧪 测试用例生成能力对比:")
        if 'quality_degradation' in test_comparison:
            print(f"   测试质量下降: {test_comparison['quality_degradation']:.1f}%")
            print(f"   原始测试长度: {test_comparison.get('original_length', 0)} 字符")
            print(f"   扰动测试长度: {test_comparison.get('perturbed_length', 0)} 字符")
        
        # 3. 综合效果评估
        print("\n🎯 综合反制效果:")
        print(f"   总体干扰度: {effectiveness['disruption_rate']:.1f}%")
        print(f"   效果等级: {effectiveness['effectiveness_level']}")
        print(f"   理解能力下降: {effectiveness['understanding_degradation']:.1f}%")
        print(f"   测试生成下降: {effectiveness['test_generation_degradation']:.1f}%")
        
        # 4. 技术分析
        print("\n🔬 技术分析:")
        if effectiveness['disruption_rate'] > 30:
            print("   ✅ 扰动策略有效干扰了LLM的tokenization")
            print("   ✅ 成功降低了LLM对安全API的理解能力")
            print("   ✅ 影响了基于文档的测试用例生成质量")
        else:
            print("   ⚠️ 扰动效果有限，可能需要更强的扰动策略")
        
        # 5. 使用的LLM信息
        print("\n🤖 测试环境:")
        print(f"   LLM后端: {self.llm_manager.current_client.name}")
        if self.llm_manager.is_mock_mode():
            print("   ⚠️ 注意: 当前使用Mock模式，结果为模拟数据")
            print("   💡 安装真实LLM可获得更准确的反制效果")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    demo = AntiLLMFuzzDemo()
    
    # 查找最新的扰动文件
    output_dirs = list(Path('output').glob('perturbations_*'))
    if not output_dirs:
        print("❌ 未找到扰动文件，请先运行扰动生成")
        print("💡 运行: python main.py --input data/00java_std.md --strategy semantic")
        return 1
    
    latest_dir = max(output_dirs, key=lambda p: p.stat().st_mtime)
    perturbed_files = list(latest_dir.glob('*.md'))
    
    if not perturbed_files:
        print(f"❌ 在 {latest_dir} 中未找到扰动文件")
        return 1
    
    print(f"🔍 找到 {len(perturbed_files)} 个扰动文件")
    
    # 运行反制效果演示
    original_file = "data/00java_std.md"
    effectiveness = demo.run_demo(original_file, [str(f) for f in perturbed_files])
    
    # 保存结果
    results_file = f"output/anti_llm_fuzz_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(effectiveness, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存到: {results_file}")
    
    return 0


if __name__ == "__main__":
    from datetime import datetime
    sys.exit(main())