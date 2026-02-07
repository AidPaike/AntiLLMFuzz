"""Dynamic priority manager for balanced token selection."""

import math
import random
from typing import List, Dict, Any, Optional
from src.data_models import Token
from src.utils import get_logger


class DynamicPriorityManager:
    """动态平衡优先级管理器。
    
    实现多维度优先级计算，平衡利用(exploit)和探索(explore)，
    防止局部最优和分数膨胀。
    
    核心功能:
    - 多维度优先级计算 (基础分、SCS分、反馈分、历史表现、探索奖励)
    - 动态权重调整 (根据实验阶段自动调整权重)
    - 混合选择策略 (80%贪婪选择 + 20%探索选择)
    - 新颖度衰减机制 (防止重复选择同一Token)
    - 分数膨胀控制 (Sigmoid衰减和上限控制)
    - 性能监控 (选择多样性、探索比例等指标)
    
    使用示例:
        >>> manager = DynamicPriorityManager()
        >>> selected = manager.select_tokens_for_perturbation(tokens, 5)
        >>> for token in selected:
        ...     impact = simulate_perturbation(token)
        ...     manager.update_after_perturbation(token, impact)
    
    配置参数:
        exploration_rate: 探索率 (默认0.2，即20%随机选择)
        max_perturbations_per_token: 单Token最大扰动次数 (默认3)
        novelty_decay_rate: 新颖度衰减率 (默认0.1)
        time_recovery_rounds: 时间恢复轮数 (默认5)
        max_feedback_score: 反馈分数上限 (默认20.0)
    """
    
    # 常量定义
    MIN_PRIORITY_SCORE = 0.01
    BASE_SCORE_NORMALIZER = 20.0
    SCS_SCORE_NORMALIZER = 100.0
    HISTORY_LIMIT = 5
    DECAY_THRESHOLD = 50.0
    DECAY_STEEPNESS = 10.0
    MIN_DECAY_FACTOR = 0.5
    RECENT_SELECTIONS_WINDOW = 5
    
    # 默认配置
    DEFAULT_CONFIG = {
        "exploration_rate": 0.2,           # 探索率 (20%随机选择)
        "max_perturbations_per_token": 3,  # 单Token最大扰动次数
        "novelty_decay_rate": 0.1,         # 新颖度衰减率
        "time_recovery_rounds": 5,          # 时间恢复轮数
        "max_feedback_score": 20.0,        # 反馈分数上限
        "feedback_decay_factor": 0.8,      # 反馈衰减因子
        
        # 阶段权重配置
        "stage_weights": {
            "initial": {
                "base": 0.4,
                "scs": 0.5,
                "feedback": 0.0,
                "historical": 0.0,
                "exploration": 0.1
            },
            "feedback": {
                "base": 0.2,
                "scs": 0.3,
                "feedback": 0.3,
                "historical": 0.1,
                "exploration": 0.1
            },
            "mature": {
                "base": 0.1,
                "scs": 0.2,
                "feedback": 0.2,
                "historical": 0.3,
                "exploration": 0.2
            }
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化动态优先级管理器。
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.current_iteration = 0
        self.current_stage = "initial"
        self.has_feedback = False
        
        # 历史记录
        self.perturbation_history = {}  # {token_text: [impact1, impact2, ...]}
        self.selection_history = []     # 选择历史
        
        # 缓存
        self._weights_cache = {}        # 缓存权重计算结果
        self._last_weights_stage = None # 上次计算权重的阶段
        
        self.logger = get_logger()
        
        # 从配置中提取常用参数
        self.exploration_rate = self.config["exploration_rate"]
        self.max_perturbations = self.config["max_perturbations_per_token"]
        self.novelty_decay_rate = self.config["novelty_decay_rate"]
        self.time_recovery_rounds = self.config["time_recovery_rounds"]
        self.max_feedback_score = self.config["max_feedback_score"]
    
    def calculate_priority(self, token: Token) -> float:
        """计算Token的动态优先级。
        
        Args:
            token: 要计算优先级的Token
            
        Returns:
            最终优先级分数
        """
        # 1. 归一化各维度分数
        normalized_scores = self._normalize_token_scores(token)
        
        # 2. 计算加权分数
        weighted_score = self._calculate_weighted_score(token, normalized_scores)
        
        # 3. 应用调整因子
        final_score = self._apply_adjustment_factors(token, weighted_score)
        
        return max(self.MIN_PRIORITY_SCORE, final_score)
    
    def _normalize_token_scores(self, token: Token) -> Dict[str, float]:
        """归一化Token的各维度分数到 [0, 1]。
        
        Args:
            token: Token对象
            
        Returns:
            归一化分数字典
        """
        return {
            "base": min(1.0, token.priority_score / self.BASE_SCORE_NORMALIZER),
            "scs": token.scs_score / self.SCS_SCORE_NORMALIZER,
            "feedback": min(1.0, token.feedback_score / self.max_feedback_score),
            "historical": token.get_historical_performance(self.perturbation_history),
            "exploration": token.get_exploration_bonus(self.max_perturbations)
        }
    
    def _calculate_weighted_score(self, token: Token, normalized_scores: Dict[str, float]) -> float:
        """计算加权分数。
        
        Args:
            token: Token对象
            normalized_scores: 归一化分数字典
            
        Returns:
            加权分数
        """
        weights = self._get_current_weights()
        
        return (
            weights["base"] * normalized_scores["base"] +
            weights["scs"] * normalized_scores["scs"] +
            weights["feedback"] * normalized_scores["feedback"] +
            weights["historical"] * normalized_scores["historical"] +
            weights["exploration"] * normalized_scores["exploration"]
        )
    
    def _apply_adjustment_factors(self, token: Token, weighted_score: float) -> float:
        """应用新颖度和衰减调整因子。
        
        Args:
            token: Token对象
            weighted_score: 加权分数
            
        Returns:
            调整后的最终分数
        """
        # 应用新颖度惩罚
        novelty_score = token.get_novelty_score(
            self.current_iteration,
            self.novelty_decay_rate,
            self.time_recovery_rounds
        )
        
        # 应用衰减因子 (防止分数膨胀)
        decay_factor = self._calculate_decay_factor(token)
        
        return weighted_score * novelty_score * decay_factor
    
    def select_tokens_for_perturbation(self, tokens: List[Token], n: int) -> List[Token]:
        """选择Token进行扰动。
        
        使用混合策略：贪婪选择 + 探索选择
        
        Args:
            tokens: 候选Token列表
            n: 要选择的Token数量
            
        Returns:
            选中的Token列表
        """
        if not tokens:
            return []
        
        if len(tokens) <= n:
            return tokens.copy()
        
        # 1. 计算所有Token的优先级
        priorities = []
        for token in tokens:
            priority = self.calculate_priority(token)
            priorities.append((token, priority))
        
        # 2. 按优先级排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 应用混合选择策略
        selected = self._apply_mixed_selection_strategy(priorities, n)
        
        # 4. 记录选择历史
        self.selection_history.append({
            'iteration': self.current_iteration,
            'selected_tokens': [t.text for t in selected],
            'selection_strategy': 'mixed',
            'greedy_count': greedy_n,
            'explore_count': len(selected) - greedy_n
        })
        
        self.logger.debug(
            f"Selected {len(selected)} tokens: {greedy_n} greedy + "
            f"{len(selected) - greedy_n} explore"
        )
        
        return selected
    
    def _apply_mixed_selection_strategy(self, priorities: List[tuple], n: int) -> List[Token]:
        """应用混合选择策略 (贪婪 + 探索)。
        
        Args:
            priorities: 排序后的(token, priority)元组列表
            n: 要选择的Token数量
            
        Returns:
            选中的Token列表
        """
        selected = []
        
        # 贪婪选择 (选择高优先级)
        greedy_n = int(n * (1 - self.exploration_rate))
        for i in range(min(greedy_n, len(priorities))):
            selected.append(priorities[i][0])
        
        # 探索选择 (随机选择剩余Token)
        remaining_tokens = [p[0] for p in priorities[greedy_n:]]
        if remaining_tokens and len(selected) < n:
            explore_n = n - len(selected)
            explored = random.sample(
                remaining_tokens,
                min(explore_n, len(remaining_tokens))
            )
            selected.extend(explored)
        
        return selected
    
    def update_after_perturbation(self, token: Token, impact: float):
        """扰动后更新Token和历史记录。
        
        Args:
            token: 被扰动的Token
            impact: 扰动影响分数 (0.0-1.0)
            
        Raises:
            ValueError: If impact is not in valid range [0.0, 1.0]
            TypeError: If token is not a Token instance
        """
        # 输入验证
        if not isinstance(token, Token):
            raise TypeError(f"Expected Token instance, got {type(token)}")
        
        if not 0.0 <= impact <= 1.0:
            raise ValueError(f"Impact must be between 0.0 and 1.0, got {impact}")
        
        # 1. 更新Token状态
        token.update_after_perturbation(self.current_iteration, impact)
        
        # 2. 更新历史记录
        if token.text not in self.perturbation_history:
            self.perturbation_history[token.text] = []
        
        self.perturbation_history[token.text].append(impact)
        
        # 3. 限制历史长度 (只保留最近几次)
        if len(self.perturbation_history[token.text]) > self.HISTORY_LIMIT:
            self.perturbation_history[token.text] = \
                self.perturbation_history[token.text][-self.HISTORY_LIMIT:]
        
        self.logger.debug(
            f"Updated token '{token.text}': count={token.perturbation_count}, "
            f"impact={impact:.3f}, avg_impact={token.average_impact:.3f}"
        )
    
    def apply_feedback_boost(self, token: Token, boost: float) -> float:
        """应用带衰减的反馈提升。
        
        Args:
            token: 要提升的Token
            boost: 原始提升量
            
        Returns:
            实际应用的提升量
        """
        # 计算当前总分 (用于衰减)
        current_total = (
            token.priority_score + 
            token.scs_score + 
            token.feedback_score
        )
        
        # 应用衰减 (分数越高，衰减越强)
        decay_factor = 1.0 / (1.0 + current_total / self.DECAY_THRESHOLD)
        final_boost = boost * decay_factor * self.config["feedback_decay_factor"]
        
        # 应用上限
        old_feedback_score = token.feedback_score
        token.feedback_score = min(
            token.feedback_score + final_boost,
            self.max_feedback_score
        )
        
        actual_boost = token.feedback_score - old_feedback_score
        
        self.logger.debug(
            f"Applied feedback boost to '{token.text}': "
            f"{old_feedback_score:.2f} -> {token.feedback_score:.2f} "
            f"(requested={boost:.2f}, actual={actual_boost:.2f})"
        )
        
        return actual_boost
    
    def update_stage(self, iteration: int, has_feedback: bool):
        """更新当前阶段。
        
        Args:
            iteration: 当前迭代次数
            has_feedback: 是否有反馈数据
        """
        self.current_iteration = iteration
        self.has_feedback = has_feedback
        
        # 根据迭代次数和反馈情况确定阶段
        if not has_feedback:
            new_stage = "initial"
        elif iteration < 5:
            new_stage = "feedback"
        else:
            new_stage = "mature"
        
        if new_stage != self.current_stage:
            self.logger.info(f"Stage changed: {self.current_stage} -> {new_stage}")
            self.current_stage = new_stage
    
    def _get_current_weights(self) -> Dict[str, float]:
        """获取当前阶段的权重配置。
        
        Returns:
            权重字典
        """
        # 使用缓存避免重复查找
        if self._last_weights_stage != self.current_stage:
            self._weights_cache = self.config["stage_weights"][self.current_stage]
            self._last_weights_stage = self.current_stage
        
        return self._weights_cache
    
    def _calculate_decay_factor(self, token: Token) -> float:
        """计算衰减因子，防止分数无限膨胀。
        
        Args:
            token: Token对象
            
        Returns:
            衰减因子 (0.5-1.0)
        """
        # 计算当前总分
        current_total = (
            token.priority_score + 
            token.scs_score + 
            token.feedback_score
        )
        
        # Sigmoid衰减 (分数越高，衰减越强)
        decay = 1.0 / (1.0 + math.exp((current_total - self.DECAY_THRESHOLD) / self.DECAY_STEEPNESS))
        
        # 确保最多衰减50%
        return max(self.MIN_DECAY_FACTOR, decay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前的性能指标。
        
        Returns:
            指标字典
        """
        if not self.selection_history:
            return {}
        
        recent_selections = self.selection_history[-self.RECENT_SELECTIONS_WINDOW:]  # 最近几次选择
        
        # 计算探索比例
        total_selections = sum(len(s['selected_tokens']) for s in recent_selections)
        total_explorations = sum(s['explore_count'] for s in recent_selections)
        exploration_ratio = total_explorations / total_selections if total_selections > 0 else 0
        
        # 计算选择多样性 (基于Token类型)
        all_selected = []
        for s in recent_selections:
            all_selected.extend(s['selected_tokens'])
        
        diversity = self._calculate_diversity(all_selected)
        
        return {
            'current_stage': self.current_stage,
            'current_iteration': self.current_iteration,
            'exploration_ratio': exploration_ratio,
            'selection_diversity': diversity,
            'total_perturbations': sum(len(impacts) for impacts in self.perturbation_history.values()),
            'unique_tokens_perturbed': len(self.perturbation_history)
        }
    
    def _calculate_diversity(self, selected_tokens: List[str]) -> float:
        """计算选择多样性 (Shannon熵)。
        
        Args:
            selected_tokens: 选中的Token文本列表
            
        Returns:
            多样性分数 (0.0-log2(n))
        """
        if not selected_tokens:
            return 0.0
        
        # 统计频率
        token_counts = {}
        for token in selected_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # 计算Shannon熵
        total = len(selected_tokens)
        entropy = 0.0
        
        for count in token_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def reset(self):
        """重置管理器状态 (用于新一轮实验)。"""
        self.current_iteration = 0
        self.current_stage = "initial"
        self.has_feedback = False
        self.perturbation_history.clear()
        self.selection_history.clear()
        
        # 清空缓存
        self._weights_cache.clear()
        self._last_weights_stage = None
        
        self.logger.info("Dynamic priority manager reset")