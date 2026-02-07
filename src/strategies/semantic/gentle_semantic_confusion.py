#!/usr/bin/env python3
"""
改进的语义混淆策略 - 温和版本

关键改进：
- 保留API类名和方法名（确保可编译）
- 只混淆描述性文本和注释
- 扭曲语义但不破坏语法
"""

import re
import random
from typing import List, Dict, Any, Optional
from src.strategies.base_strategy import PerturbationStrategy
from src.data_models import Token


class GentleSemanticConfusionStrategy(PerturbationStrategy):
    """
    温和的语义混淆策略 - 保留API名称，混淆描述
    
    核心思想：
    1. 保留所有Java API类名和方法名（确保LLM生成可编译代码）
    2. 混淆描述性文本（功能描述、参数说明、返回值描述）
    3. 扭曲语义但不破坏语法
    4. 让LLM生成语法正确但逻辑错误的测试
    """
    
    def __init__(self):
        super().__init__(
            name="gentle_semantic_confusion",
            description="Gently confuse descriptive text while preserving API names",
            category="semantic",
            supported_targets=["documentation"],
            supported_languages=["documentation", "java", "python"],
            code_safety="safe"
        )
        
        # 描述性词汇的混淆映射（保留API名称）
        self.description_mappings = {
            # 功能描述
            'cryptographic hash': ['data transformation', 'content encoding', 'information processing'],
            'security policy': ['access guidelines', 'permission rules', 'usage constraints'],
            'random number': ['unpredictable value', 'arbitrary sequence', 'non-deterministic data'],
            'encrypt': ['transform', 'process', 'encode'],
            'decrypt': ['reverse', 'decode', 'extract'],
            
            # 操作描述
            'checks': ['examines', 'inspects', 'verifies'],
            'validates': ['confirms', 'assesses', 'reviews'],
            'throws': ['signals', 'indicates', 'reports'],
            'returns': ['provides', 'gives', 'yields'],
            
            # 安全相关描述（保留异常类名）
            'security exception': ['access concern', 'permission issue', 'restriction notice'],
            'invalid argument': ['input concern', 'parameter issue', 'value problem'],
            'illegal state': ['unexpected condition', 'improper status', 'wrong context'],
            
            # 程度描述
            'must': ['should consider', 'is recommended to', 'ought to'],
            'always': ['typically', 'usually', 'generally'],
            'never': ['rarely', 'seldom', 'avoid when possible'],
            'required': ['suggested', 'advised', 'encouraged'],
            
            # 结果描述
            'hash': ['result', 'output', 'value'],
            'encrypted': ['processed', 'transformed', 'encoded'],
            'decrypted': ['recovered', 'extracted', 'decoded'],
        }
        
        # 模糊的边界描述（误导LLM关于边界条件）
        self.vague_boundaries = [
            (r'输入长度必须在\s*\d+\s*到\s*\d+', '输入长度应在合理范围内'),
            (r'参数不能为null', '参数应该有效'),
            (r'必须大于\s*\d+', '应该是正值'),
            (r'必须小于\s*\d+', '不应过大'),
        ]
        
        # 误导性的功能描述
        self.misleading_functions = {
            'getInstance': ['获取共享实例', '返回全局对象'],
            'update': ['追加数据', '合并内容'],
            'digest': ['获取结果', '完成计算'],
            'nextBytes': ['生成序列', '填充数组'],
        }
    
    def apply(self, token: Token, content: str, **kwargs) -> str:
        """应用温和的语义混淆"""
        modified = content
        
        # 1. 混淆描述性词汇（保留API名称）
        modified = self._confuse_descriptions(modified)
        
        # 2. 模糊边界条件
        modified = self._vague_boundaries(modified)
        
        # 3. 误导性功能描述
        modified = self._misleading_function_desc(modified)
        
        # 4. 扭曲最佳实践
        modified = self._distort_best_practices(modified)
        
        return modified
    
    def _confuse_descriptions(self, content: str) -> str:
        """混淆描述性文本，保留API名称"""
        modified = content
        
        # 按长度降序排序，避免部分替换
        sorted_terms = sorted(
            self.description_mappings.keys(),
            key=len,
            reverse=True
        )
        
        for term in sorted_terms:
            if term.lower() in modified.lower():
                replacements = self.description_mappings[term]
                replacement = random.choice(replacements)
                
                # 大小写敏感替换
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                modified = pattern.sub(replacement, modified)
        
        return modified
    
    def _vague_boundaries(self, content: str) -> str:
        """将精确的边界条件改为模糊描述"""
        modified = content
        
        for pattern, replacement in self.vague_boundaries:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        
        return modified
    
    def _misleading_function_desc(self, content: str) -> str:
        """误导性的方法功能描述"""
        modified = content
        
        for method, misleading_descs in self.misleading_functions.items():
            if method in content:
                # 找到方法描述并替换
                pattern = rf'({re.escape(method)}[^(]*\([^)]*\)[^:]*:\s*)([^.\n]+)'
                
                def replace_desc(match):
                    prefix = match.group(1)
                    new_desc = random.choice(misleading_descs)
                    return prefix + new_desc
                
                modified = re.sub(pattern, replace_desc, modified, flags=re.IGNORECASE)
        
        return modified
    
    def _distort_best_practices(self, content: str) -> str:
        """扭曲最佳实践建议"""
        # 将强烈建议改为弱建议
        distortions = [
            (r'应该始终', '通常建议'),
            (r'必须', '建议'),
            (r'最佳实践', '一种方法'),
            (r'强烈推荐', '可以考虑'),
            (r'永远不要', '尽量避免'),
        ]
        
        modified = content
        for pattern, replacement in distortions:
            modified = re.sub(pattern, replacement, modified)
        
        return modified
    
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
        
        for i, token in enumerate(tokens_to_perturb):
            variant_name = f"{self.name}_token{i+1}"
            perturbed_content = self.apply(token, content, **kwargs)
            perturbed_versions[variant_name] = perturbed_content
        
        return perturbed_versions
