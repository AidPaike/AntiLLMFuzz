#!/usr/bin/env python3
"""
矛盾信息策略 - 提供相互矛盾的使用说明

核心思想：
- 在文档的不同部分提供相互矛盾的信息
- 让LLM无法确定正确的使用方式
- 例如：一个地方说"应该重用实例"，另一个地方说"每次创建新实例"
- LLM会困惑并生成不一致的测试
"""

import re
import random
from typing import List, Dict, Any, Optional
from src.strategies.base_strategy import PerturbationStrategy
from src.data_models import Token


class ContradictoryInfoStrategy(PerturbationStrategy):
    """
    矛盾信息策略 - 提供相互矛盾的使用说明
    
    策略：
    1. 在文档不同位置提供矛盾的使用建议
    2. 让LLM无法确定正确的API用法
    3. 示例代码展示不同的调用模式
    4. 混淆参数的有效范围
    """
    
    def __init__(self):
        super().__init__(
            name="contradictory_info",
            description="Provide contradictory usage information in different sections",
            category="semantic",
            supported_targets=["documentation"],
            supported_languages=["documentation", "java", "python"],
            code_safety="safe"
        )
        
        # 矛盾的用法对
        self.contradictions = [
            {
                'topic': '实例创建',
                'statement1': '每次调用都应该创建新的实例',
                'statement2': '实例应该被重用以提高性能',
                'example1': 'MessageDigest md = MessageDigest.getInstance("SHA-256");  // 每次新建',
                'example2': 'private static MessageDigest md;  // 复用实例',
            },
            {
                'topic': '异常处理',
                'statement1': '所有异常都应该被捕获并处理',
                'statement2': '运行时异常不需要显式捕获',
                'example1': 'try { ... } catch (Exception e) { ... }',
                'example2': '// 无需try-catch，直接调用',
            },
            {
                'topic': '线程安全',
                'statement1': '该类是线程安全的，可以在多线程环境中共享',
                'statement2': '每个线程应该有自己的实例',
                'example1': 'public static final Cipher cipher = ...;  // 共享',
                'example2': 'ThreadLocal<Cipher> cipher = ...;  // 线程本地',
            },
            {
                'topic': '输入验证',
                'statement1': '输入参数必须进行严格的null检查',
                'statement2': '框架会自动处理null值',
                'example1': 'if (input == null) throw new ...;',
                'example2': '// 直接处理，无需检查',
            },
            {
                'topic': '资源释放',
                'statement1': '使用完毕后必须显式关闭资源',
                'statement2': '依赖垃圾回收自动清理资源',
                'example1': 'finally { stream.close(); }',
                'example2': '// 无需关闭，自动回收',
            },
        ]
        
        # 矛盾的参数范围
        self.contradictory_ranges = [
            {
                'param': 'key size',
                'range1': '128 bits (faster)',
                'range2': '256 bits (required for security)',
            },
            {
                'param': 'iteration count',
                'range1': '1000 (minimum)',
                'range2': '100000 (recommended)',
            },
            {
                'param': 'salt length',
                'range1': '8 bytes',
                'range2': '16 bytes or more',
            },
        ]
    
    def apply(self, token: Token, content: str, **kwargs) -> str:
        """应用矛盾信息"""
        modified = content
        
        operator = kwargs.get('operator', 'add_contradictions')
        
        if operator == 'add_contradictions':
            modified = self._add_contradictory_statements(modified, token)
        elif operator == 'confuse_ranges':
            modified = self._add_contradictory_ranges(modified)
        elif operator == 'full':
            modified = self._add_contradictory_statements(modified, token)
            modified = self._add_contradictory_ranges(modified)
        
        return modified
    
    def _add_contradictory_statements(self, content: str, token: Token) -> str:
        """添加矛盾的用法说明"""
        lines = content.split('\n')
        
        # 在文档的不同部分插入矛盾信息
        # 前半部分放第一个说法，后半部分放矛盾的说法
        mid_point = len(lines) // 2
        
        # 选择几个矛盾对
        selected = random.sample(self.contradictions, min(2, len(self.contradictions)))
        
        for i, contradiction in enumerate(selected):
            if i == 0:
                # 在文档前半部分插入第一个说法
                insert_pos = mid_point // 2
                block = [
                    "",
                    f"> **注意**: {contradiction['statement1']}",
                    f"> 示例: `{contradiction['example1']}`",
                    "",
                ]
            else:
                # 在文档后半部分插入矛盾的说法
                insert_pos = mid_point + (len(lines) - mid_point) // 2
                block = [
                    "",
                    f"> **重要**: {contradiction['statement2']}",
                    f"> 示例: `{contradiction['example2']}`",
                    "",
                ]
            
            lines = lines[:insert_pos] + block + lines[insert_pos:]
        
        return '\n'.join(lines)
    
    def _add_contradictory_ranges(self, content: str) -> str:
        """添加矛盾的参数范围"""
        lines = content.split('\n')
        modified = []
        
        added_ranges = []
        for line in lines:
            modified.append(line)
            
            # 在参数描述后添加矛盾的范围说明
            if 'Parameters:' in line or '**参数**' in line:
                for range_info in self.contradictory_ranges:
                    if range_info['param'] not in added_ranges:
                        block = [
                            "",
                            f"- {range_info['param']}: ",
                            f"  - 最低要求: {range_info['range1']}",
                            f"  - 安全建议: {range_info['range2']}",
                        ]
                        modified.extend(block)
                        added_ranges.append(range_info['param'])
                        break
        
        return '\n'.join(modified)
    
    def apply_multiple(
        self,
        tokens: List[Token],
        content: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, str]:
        """应用多种变体"""
        if max_tokens is None:
            max_tokens = len(tokens)
        
        perturbed_versions = {}
        tokens_to_perturb = tokens[:max_tokens]
        
        operators = ['add_contradictions', 'confuse_ranges', 'full']
        
        for i, token in enumerate(tokens_to_perturb):
            for operator in operators:
                variant_name = f"{self.name}_{operator}_token{i+1}"
                perturbed_content = self.apply(
                    token,
                    content,
                    operator=operator,
                    **kwargs
                )
                perturbed_versions[variant_name] = perturbed_content
        
        return perturbed_versions
