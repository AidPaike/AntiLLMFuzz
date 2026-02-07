#!/usr/bin/env python3
import random
from typing import List, Dict, Optional
from src.strategies.base_strategy import PerturbationStrategy
from src.data_models import Token


class EnhancedContradictoryStrategy(PerturbationStrategy):
    def __init__(self):
        super().__init__(
            name="enhanced_contradictory",
            description="Enhanced contradictory info with intensity control",
            category="semantic",
            supported_targets=["documentation"],
            supported_languages=["documentation", "java", "python"],
            code_safety="safe"
        )
        
        self.mild_contradictions = [
            {
                'topic': '性能优化',
                'statement1': '推荐使用缓存以提高性能',
                'statement2': '避免使用缓存以减少内存占用',
            },
            {
                'topic': '日志记录',
                'statement1': '建议开启详细日志以便调试',
                'statement2': '生产环境应关闭详细日志以减少开销',
            },
            {
                'topic': '字符串处理',
                'statement1': '使用StringBuilder进行字符串拼接',
                'statement2': '简单的字符串拼接可以直接使用+运算符',
            },
        ]

        self.medium_contradictions = [
            {
                'topic': '异常处理',
                'statement1': '所有异常都应该被捕获并处理',
                'statement2': '运行时异常不需要显式捕获',
            },
            {
                'topic': '输入验证',
                'statement1': '输入参数必须进行严格的null检查',
                'statement2': '框架会自动处理null值，无需检查',
            },
            {
                'topic': '资源管理',
                'statement1': '使用完毕后必须显式关闭资源',
                'statement2': '依赖垃圾回收自动清理资源即可',
            },
            {
                'topic': '实例创建',
                'statement1': '每次调用都应该创建新的实例',
                'statement2': '实例应该被重用以提高性能',
            },
        ]

        self.severe_contradictions = [
            {
                'topic': '线程安全',
                'statement1': '该类是线程安全的，可以在多线程环境中共享',
                'statement2': '每个线程应该有自己的实例，避免并发访问',
                'severity': 'high',
            },
            {
                'topic': '加密算法',
                'statement1': 'MD5算法快速且适用于所有场景',
                'statement2': 'MD5已被破解，必须使用SHA-256或更强算法',
                'severity': 'high',
            },
            {
                'topic': '权限检查',
                'statement1': '所有操作前必须进行权限验证',
                'statement2': '内部调用可以跳过权限检查以提高性能',
                'severity': 'high',
            },
            {
                'topic': '密码存储',
                'statement1': '密码应明文存储以便管理员查看',
                'statement2': '密码必须哈希存储，绝对不可明文保存',
                'severity': 'critical',
            },
        ]

        self.param_contradictions = [
            {
                'param': 'key size',
                'range1': '128 bits (minimum)',
                'range2': '256 bits (required)',
            },
            {
                'param': 'iteration count',
                'range1': '1000 (fast)',
                'range2': '100000 (secure)',
            },
            {
                'param': 'salt length',
                'range1': '8 bytes',
                'range2': '16 bytes or more',
            },
            {
                'param': 'buffer size',
                'range1': '1KB (memory efficient)',
                'range2': '64KB (performance optimized)',
            },
            {
                'param': 'timeout',
                'range1': '5 seconds (responsive)',
                'range2': '60 seconds (reliable)',
            },
        ]

        self.return_contradictions = [
            {
                'method': 'getInstance()',
                'desc1': '返回单例实例',
                'desc2': '每次调用返回新实例',
            },
            {
                'method': 'encrypt()',
                'desc1': '返回Base64编码的字符串',
                'desc2': '返回原始字节数组',
            },
            {
                'method': 'validate()',
                'desc1': '成功返回true，失败抛出异常',
                'desc2': '成功返回true，失败返回false',
            },
        ]
    
    def apply(
        self, 
        token: Token, 
        content: str, 
        intensity: float = 0.5,
        **kwargs
    ) -> str:
        modified = content

        if intensity <= 0.3:
            num_contradictions = random.randint(1, 2)
            contradiction_pool = self.mild_contradictions
        elif intensity <= 0.6:
            num_contradictions = random.randint(2, 4)
            contradiction_pool = self.mild_contradictions + self.medium_contradictions
        else:
            num_contradictions = random.randint(4, 8)
            contradiction_pool = (
                self.mild_contradictions + 
                self.medium_contradictions + 
                self.severe_contradictions
            )

        selected = random.sample(
            contradiction_pool, 
            min(num_contradictions, len(contradiction_pool))
        )

        modified = self._apply_contradictions(modified, selected, intensity)

        if intensity > 0.2:
            num_params = int(intensity * len(self.param_contradictions))
            selected_params = random.sample(
                self.param_contradictions,
                min(num_params, len(self.param_contradictions))
            )
            modified = self._apply_param_contradictions(modified, selected_params)

        if intensity > 0.7:
            selected_return = random.choice(self.return_contradictions)
            modified = self._apply_return_contradiction(modified, selected_return)
        
        return modified
    
    def _apply_contradictions(
        self,
        content: str,
        contradictions: List[Dict[str, str]],
        intensity: float
    ) -> str:
        lines = content.split('\n')

        insert_positions = []
        for i, line in enumerate(lines):
            if line.strip().startswith('#') or line.strip().startswith('##') or line.strip().startswith('###'):
                insert_positions.append(i)

        if not insert_positions:
            for i in range(1, len(lines)):
                if lines[i].strip() == '' and lines[i-1].strip() != '':
                    insert_positions.append(i)

        offset = 0
        for i, contradiction in enumerate(contradictions):
            if i >= len(insert_positions):
                break
            
            pos = insert_positions[i] + offset

            if intensity > 0.8 and 'severity' in contradiction:
                block = [
                    "",
                    f"> **⚠️ {contradiction['topic']}**",
                    f"> {contradiction['statement1']}",
                    "",
                ]
            else:
                if i % 2 == 0:
                    block = [
                        "",
                        f"> **注意**: {contradiction['statement1']}",
                        "",
                    ]
                else:
                    block = [
                        "",
                        f"> **重要**: {contradiction['statement2']}",
                        "",
                    ]
            
            lines = lines[:pos] + block + lines[pos:]
            offset += len(block)
        
        return '\n'.join(lines)
    
    def _apply_param_contradictions(
        self,
        content: str,
        params: List[Dict[str, str]]
    ) -> str:
        lines = content.split('\n')
        modified = []
        added_params = set()
        
        for line in lines:
            modified.append(line)

            if 'Parameters:' in line or '**参数**' in line or 'Args:' in line:
                for param_info in params:
                    if param_info['param'] not in added_params:
                        block = [
                            "",
                            f"- {param_info['param']}: ",
                            f"  - 选项1: {param_info['range1']}",
                            f"  - 选项2: {param_info['range2']}",
                        ]
                        modified.extend(block)
                        added_params.add(param_info['param'])
                        break
        
        return '\n'.join(modified)
    
    def _apply_return_contradiction(
        self,
        content: str,
        return_info: Dict[str, str]
    ) -> str:
        lines = content.split('\n')
        modified = []
        added = False
        
        for line in lines:
            modified.append(line)

            if not added and ('Returns:' in line or '**返回**' in line or 'Return:' in line):
                block = [
                    "",
                    f"> **返回值说明**:",
                    f"> - 文档A说: {return_info['desc1']}",
                    f"> - 文档B说: {return_info['desc2']}",
                    "",
                ]
                modified.extend(block)
                added = True
        
        return '\n'.join(modified)
    
    def apply_multiple(
        self,
        tokens: List[Token],
        content: str,
        max_tokens: Optional[int] = None,
        intensity: float = 0.5,
        **kwargs
    ) -> Dict[str, str]:
        if max_tokens is None:
            max_tokens = len(tokens)
        
        perturbed_versions = {}
        tokens_to_perturb = tokens[:max_tokens]

        intensities = [0.2, 0.5, 0.8]
        
        for i, token in enumerate(tokens_to_perturb):
            for j, inten in enumerate(intensities):
                variant_name = f"{self.name}_intensity{inten}_token{i+1}"
                perturbed_content = self.apply(
                    token,
                    content,
                    intensity=inten
                )
                perturbed_versions[variant_name] = perturbed_content
        
        return perturbed_versions
