"""
语义混淆策略 (Semantic Confusion Strategy)

基于NLP领域的EuphemAttack和委婉语重写技术：
- 将关键API术语替换为模糊或误导性的同义词
- 使用委婉、隐含或反讽的表达方式来描述关键操作
- 保持文档对人类可读，但LLM难以理解精确语义
- 绕过LLM的关键词检测和语义理解

技术来源：
- EuphemAttack: 委婉语重写绕过安全检测
- Doublespeak: 表示劫持攻击
- 对抗样本中的同义词替换
"""

import re
import random
from typing import List, Dict, Any, Optional
from src.strategies.base_strategy import PerturbationStrategy
from src.data_models import Token


class SemanticConfusionStrategy(PerturbationStrategy):
    """
    语义混淆策略 - 通过委婉语和同义词替换干扰LLM语义理解
    
    核心思想：
    1. 识别关键API术语（方法名、类名、技术概念）
    2. 用模糊或误导性的表述替换
    3. 使用委婉语描述敏感/关键操作
    4. 保持语法正确但语义模糊化
    """
    
    def __init__(self):
        super().__init__(
            name="semantic_confusion",
            description="Replace key API terms with vague or misleading euphemisms",
            category="semantic",
            supported_targets=["documentation"],
            supported_languages=["documentation", "java", "python"],
            code_safety="safe"
        )
        
        # API术语的混淆映射
        self.term_substitutions = {
            # 安全相关
            'SecurityManager': ['access controller', 'permission handler', 'security monitor'],
            'checkPermission': ['verify access', 'validate rights', 'confirm privileges'],
            'checkRead': ['attempt access', 'try to view', 'request reading'],
            'checkWrite': ['attempt modification', 'try to change', 'request writing'],
            
            # 加密相关
            'MessageDigest': ['data processor', 'hash generator', 'checksum tool'],
            'getInstance': ['create', 'obtain', 'retrieve'],
            'digest': ['process', 'transform', 'compute result'],
            'SecureRandom': ['random source', 'entropy provider', 'number generator'],
            'nextBytes': ['generate data', 'produce output', 'create sequence'],
            
            # 输入验证
            'validate': ['inspect', 'examine', 'check'],
            'IllegalArgumentException': ['issue', 'problem', 'concern'],
            'SecurityException': ['restriction', 'limitation', 'boundary'],
            
            # 通用术语
            'throws': ['might indicate', 'could signal', 'may suggest'],
            'exception': ['situation', 'condition', 'state'],
            'error': ['unexpected outcome', 'unusual result', 'special case'],
            'return': ['provide', 'give back', 'yield'],
            'parameter': ['input value', 'provided data', 'given information'],
        }
        
        # 委婉语模板
        self.euphemism_templates = {
            'security_check': [
                "确保操作符合预期",
                "验证请求的合理性",
                "确认行为的适当性",
            ],
            'error_handling': [
                "处理特殊情况",
                "应对非正常状态",
                "管理边界条件",
            ],
            'validation': [
                "确认输入的合理性",
                "检查数据的适用性",
                "验证信息的有效性",
            ],
            'cryptographic': [
                "对数据进行转换处理",
                "应用特定的算法操作",
                "执行标准的计算流程",
            ],
        }
        
        # 模糊化表述
        self.vague_expressions = [
            "在某些条件下",
            "根据具体情况",
            "视应用场景而定",
            "通常而言",
            "在大多数情况下",
            "除非另有说明",
        ]
    
    def apply(self, token: Token, content: str, **kwargs) -> str:
        """
        应用语义混淆
        
        策略：
        1. 替换关键API术语为模糊表述
        2. 将精确异常描述改为委婉语
        3. 添加模糊化限定词
        """
        modified_content = content
        
        # 获取操作符
        operator = kwargs.get('operator', 'term_substitution')
        
        if operator == 'term_substitution':
            modified_content = self._apply_term_substitution(content)
        elif operator == 'euphemism':
            modified_content = self._apply_euphemism(content, token)
        elif operator == 'vague_modifier':
            modified_content = self._apply_vague_modifiers(content)
        elif operator == 'full_confusion':
            # 组合所有操作
            modified_content = self._apply_term_substitution(content)
            modified_content = self._apply_euphemism(modified_content, token)
            modified_content = self._apply_vague_modifiers(modified_content)
        
        return modified_content
    
    def _apply_term_substitution(self, content: str) -> str:
        """替换关键术语为模糊表述"""
        modified = content
        
        # 按长度降序排序，避免部分替换问题
        sorted_terms = sorted(
            self.term_substitutions.keys(),
            key=len,
            reverse=True
        )
        
        for term in sorted_terms:
            if term in modified:
                # 随机选择一个替换词
                substitutions = self.term_substitutions[term]
                replacement = random.choice(substitutions)
                
                # 替换，但保留原始大小写模式
                modified = self._smart_replace(modified, term, replacement)
        
        return modified
    
    def _smart_replace(self, text: str, old: str, new: str) -> str:
        """智能替换，尝试保持大小写一致性"""
        # 如果原词首字母大写，替换词也首字母大写
        if old[0].isupper() and new[0].islower():
            new = new[0].upper() + new[1:]
        
        # 使用正则进行词边界替换
        pattern = r'\b' + re.escape(old) + r'\b'
        return re.sub(pattern, new, text)
    
    def _apply_euphemism(self, content: str, token: Token) -> str:
        """应用委婉语描述"""
        modified = content
        
        # 识别不同类别的描述并替换为委婉语
        euphemism_mappings = [
            (r'检查权限|安全检查', 'security_check'),
            (r'异常处理|错误处理', 'error_handling'),
            (r'验证|校验', 'validation'),
            (r'加密|哈希|签名', 'cryptographic'),
        ]
        
        for pattern, category in euphemism_mappings:
            if re.search(pattern, content):
                templates = self.euphemism_templates.get(category, [])
                if templates:
                    euphemism = random.choice(templates)
                    # 替换匹配的部分
                    modified = re.sub(pattern, euphemism, modified, count=1)
        
        return modified
    
    def _apply_vague_modifiers(self, content: str) -> str:
        """添加模糊化限定词"""
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_lines.append(line)
            
            # 在精确描述后添加模糊限定
            if self._is_precise_description(line) and random.random() < 0.3:
                vague = random.choice(self.vague_expressions)
                modified_lines.append(f" ({vague})")
        
        return '\n'.join(modified_lines)
    
    def _is_precise_description(self, line: str) -> bool:
        """判断是否是精确描述行"""
        precise_patterns = [
            r'必须|一定|始终',
            r'返回\s*(true|false|\d+)',
            r'抛出\s*\w+Exception',
            r'输入\s*(必须|应该|需要)',
        ]
        
        for pattern in precise_patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def apply_multiple(
        self,
        tokens: List[Token],
        content: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, str]:
        """应用多种语义混淆变体"""
        if max_tokens is None:
            max_tokens = len(tokens)
        
        perturbed_versions = {}
        tokens_to_perturb = tokens[:max_tokens]
        
        # 定义操作符
        operators = ['term_substitution', 'euphemism', 'vague_modifier', 'full_confusion']
        
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
