"""
分层扰动策略 (Layered Perturbation Strategy)

基于NLP领域的"分层文档"概念和Oracle专家的建议：
- 将文档分为"规范层"(短而准确)和"非规范层"(长而噪)
- 在规范层中保留核心API签名和契约
- 在非规范层中分散、混淆和稀释关键信息
- 使LLM难以从非规范层提取可测试的预言机

技术来源：
- Oracle建议的"dual-layer documentation"
- NLP领域的semantic structure manipulation
"""

import re
import random
from typing import List, Dict, Any, Optional
from src.strategies.base_strategy import PerturbationStrategy
from src.data_models import Token


class LayeredPerturbationStrategy(PerturbationStrategy):
    """
    分层扰动策略 - 通过分层文档结构干扰LLM信息提取
    
    核心思想：
    1. 识别并保留规范信息（API签名、参数类型）
    2. 将详细的语义信息分散到多个非规范段落
    3. 在非规范层中插入噪声、交叉引用和模糊表述
    4. 保持文档对人类可读，但LLM难以提取精确测试信息
    """
    
    def __init__(self):
        super().__init__(
            name="layered_perturbation",
            description="Split document into normative (concise) and non-normative (verbose/noisy) layers",
            category="semantic",
            supported_targets=["documentation"],
            supported_languages=["documentation", "java", "python"],
            code_safety="safe"
        )
        
        # 非规范层的噪声模板
        self.distraction_templates = [
            "值得注意的是，{topic}的实现可能因具体环境而异。",
            "在某些情况下，开发者可能需要考虑{topic}的替代方案。",
            "关于{topic}的更多细节，请参考相关文档或社区讨论。",
            "实践中，{topic}的行为可能受到多种因素影响。",
            "需要特别说明的是，{topic}并非在所有场景下都适用。",
        ]
        
        # 模糊化表述模板
        self.vague_templates = [
            "参数应该在合理范围内",
            "返回值通常是预期的结果",
            "可能抛出某些异常",
            "执行时间取决于具体情况",
            "线程安全性需要根据实际情况判断",
        ]
        
        # 交叉引用噪声
        self.cross_ref_templates = [
            "（参见后文关于{topic}的讨论）",
            "（这一行为与{topic}相关，详见其他章节）",
            "（更多信息请参考{topic}的实现说明）",
        ]
    
    def apply(self, token: Token, content: str, **kwargs) -> str:
        """
        应用分层扰动
        
        策略：
        1. 识别API定义（方法签名、类定义）作为规范层
        2. 将详细说明转换为非规范层
        3. 在非规范层中分散关键信息
        """
        modified_content = content
        
        # 获取操作符
        operator = kwargs.get('operator', 'split_layer')
        
        if operator == 'split_layer':
            modified_content = self._apply_layer_split(content, token)
        elif operator == 'vague_description':
            modified_content = self._apply_vague_descriptions(content, token)
        elif operator == 'cross_reference':
            modified_content = self._apply_cross_references(content, token)
        elif operator == 'dense_noise':
            # 组合所有操作
            modified_content = self._apply_layer_split(content, token)
            modified_content = self._apply_vague_descriptions(modified_content, token)
            modified_content = self._apply_cross_references(modified_content, token)
        
        return modified_content
    
    def _apply_layer_split(self, content: str, token: Token) -> str:
        """将文档分层为规范和半规范部分"""
        lines = content.split('\n')
        normative_lines = []
        non_normative_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # 识别规范信息（API定义、参数列表等）
            if self._is_normative(stripped):
                normative_lines.append(line)
            else:
                # 非规范信息添加到非规范层
                non_normative_lines.append(line)
                
                # 在非规范信息之间插入噪声
                if random.random() < 0.3:  # 30%概率插入噪声
                    noise = self._generate_noise(token.text)
                    non_normative_lines.append(f"\n> **补充说明**: {noise}\n")
        
        # 重组文档：规范层 + 分隔符 + 非规范层
        result = []
        result.append("## 规范定义 (Normative)\n")
        result.extend(normative_lines)
        result.append("\n---\n")
        result.append("## 详细说明 (Non-Normative)\n")
        result.append("以下信息仅供参考，不构成严格规范：\n")
        result.extend(non_normative_lines)
        
        return '\n'.join(result)
    
    def _is_normative(self, line: str) -> bool:
        """判断一行是否包含规范信息"""
        normative_patterns = [
            r'^public\s+',  # public方法
            r'^private\s+',  # private方法
            r'^class\s+',  # 类定义
            r'@\w+',  # 注解
            r'^\s*-\s+\w+\s*\(',  # 方法列表项
            r'\b(?:String|int|boolean|void|Object)\b.*\(',  # 带类型的方法
        ]
        
        for pattern in normative_patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def _apply_vague_descriptions(self, content: str, token: Token) -> str:
        """将精确描述替换为模糊表述"""
        # 替换精确数值和边界
        vague_content = content
        
        # 替换"抛出XxxException"为模糊表述
        vague_content = re.sub(
            r'抛出\s*(\w+Exception)',
            lambda m: random.choice(self.vague_templates),
            vague_content
        )
        
        # 替换精确的输入范围
        vague_content = re.sub(
            r'输入必须在\s*\[?\d+\s*,\s*\d+\]?',
            "输入参数应在合理范围内",
            vague_content
        )
        
        # 替换精确返回值
        vague_content = re.sub(
            r'返回\s*(true|false|\d+|"[^"]*")',
            "返回预期的结果",
            vague_content
        )
        
        return vague_content
    
    def _apply_cross_references(self, content: str, token: Token) -> str:
        """添加交叉引用噪声，分散注意力"""
        lines = content.split('\n')
        modified_lines = []
        
        for i, line in enumerate(lines):
            modified_lines.append(line)
            
            # 在关键信息后添加交叉引用
            if self._is_key_information(line) and random.random() < 0.4:
                ref = random.choice(self.cross_ref_templates).format(topic=token.text)
                modified_lines.append(f" {ref}")
        
        return '\n'.join(modified_lines)
    
    def _is_key_information(self, line: str) -> bool:
        """判断是否是关键信息行"""
        key_patterns = [
            r'Example|示例',
            r'Note:|注意',
            r'Important:|重要',
            r'Returns?|返回',
            r'Parameters?|参数',
        ]
        
        for pattern in key_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_noise(self, topic: str) -> str:
        """生成关于主题的噪声文本"""
        template = random.choice(self.distraction_templates)
        return template.format(topic=topic)
    
    def apply_multiple(
        self,
        tokens: List[Token],
        content: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, str]:
        """应用多种分层扰动变体"""
        if max_tokens is None:
            max_tokens = len(tokens)
        
        perturbed_versions = {}
        tokens_to_perturb = tokens[:max_tokens]
        
        # 定义操作符组合
        operators = ['split_layer', 'vague_description', 'cross_reference', 'dense_noise']
        
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
