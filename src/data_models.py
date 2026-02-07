"""Data models for the LLM Fuzzer Semantic Disruptor."""

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from src.priority.dynamic_manager import DynamicPriorityManager


# Constants for priority calculations
class PriorityConstants:
    """Constants for priority calculation algorithms."""
    
    # Novelty score parameters
    MIN_NOVELTY_SCORE = 0.2
    NOVELTY_COUNT_WEIGHT = 0.6
    NOVELTY_TIME_WEIGHT = 0.4
    DEFAULT_DECAY_RATE = 0.1
    DEFAULT_RECOVERY_ROUNDS = 5
    
    # Exploration bonus parameters
    MIN_EXPLORATION_BONUS = 0.1
    MAX_EXPLORATION_BONUS = 1.0
    EXPLORATION_DECAY_FACTOR = 0.5
    DEFAULT_MAX_PERTURBATIONS = 3
    
    # Historical performance parameters
    DEFAULT_PERFORMANCE_SCORE = 0.5
    HISTORICAL_WEIGHT_BASE = 0.5


@dataclass
class Token:
    """Represents a token extracted from code or documentation.
    
    Enhanced with dynamic priority management capabilities including:
    - Historical tracking of perturbations
    - Dynamic priority calculations
    - Performance caching
    
    Attributes:
        text: The actual token text
        line: Line number in source file (1-indexed)
        column: Column position in source file (0-indexed)
        token_type: Type classification ('function', 'variable', 'literal', etc.)
        source_file: Path to the source file
        priority_score: Base priority score (0.0+)
        scs_score: Semantic Contribution Score (0.0-100.0)
        module_type: Module classification ('A', 'B', 'C', 'D')
        perturbation_count: Number of times this token has been perturbed
        last_perturbed_iter: Iteration when last perturbed (-1 if never)
        cumulative_impact: Sum of all perturbation impacts
        feedback_score: Score from LLM feedback analysis
    """
    # 原有字段 (保持向后兼容)
    text: str
    line: int
    column: int
    token_type: str  # 'function', 'variable', 'literal', 'conditional', 'noun', 'verb', 'phrase'
    source_file: str
    priority_score: float = 0.0  # 基础优先级分数
    scs_score: float = 0.0       # SCS分数
    module_type: str = ""        # 'A', 'B', 'C', 'D'
    
    # 新增历史跟踪字段
    perturbation_count: int = 0           # 被扰动次数
    last_perturbed_iter: int = -1         # 最后扰动的迭代 (-1表示从未扰动)
    cumulative_impact: float = 0.0        # 累积影响分数
    feedback_score: float = 0.0           # 反馈分数 (独立于priority_score)
    
    # 缓存字段 (不包含在序列化中)
    _cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    
    # 运行时计算的属性 (不存储，动态计算)
    def get_novelty_score(self, current_iter: int, 
                         decay_rate: float = PriorityConstants.DEFAULT_DECAY_RATE, 
                         recovery_rounds: int = PriorityConstants.DEFAULT_RECOVERY_ROUNDS) -> float:
        """计算新颖度分数 (0.2-1.0)。
        
        新颖度随扰动次数衰减，随时间恢复。
        
        Args:
            current_iter: 当前迭代次数
            decay_rate: 衰减率
            recovery_rounds: 恢复轮数
            
        Returns:
            新颖度分数 (0.2-1.0)
        """
        # 基于扰动次数的指数衰减
        count_penalty = math.exp(-decay_rate * self.perturbation_count)
        
        # 基于时间的恢复
        if self.last_perturbed_iter >= 0:
            time_gap = current_iter - self.last_perturbed_iter
            time_recovery = min(1.0, time_gap / recovery_rounds)
        else:
            time_recovery = 1.0  # 从未扰动，完全新颖
        
        # 综合计算 (使用常量权重)
        novelty = (count_penalty * PriorityConstants.NOVELTY_COUNT_WEIGHT + 
                  time_recovery * PriorityConstants.NOVELTY_TIME_WEIGHT)
        
        # 确保最低分数，避免完全归零
        return max(PriorityConstants.MIN_NOVELTY_SCORE, novelty)
    
    def get_exploration_bonus(self, max_perturbations: int = PriorityConstants.DEFAULT_MAX_PERTURBATIONS) -> float:
        """计算探索奖励分数 (0.1-1.0)。
        
        未扰动或扰动次数少的Token获得奖励。
        
        Args:
            max_perturbations: 最大扰动次数阈值
            
        Returns:
            探索奖励分数 (0.1-1.0)
        """
        if self.perturbation_count == 0:
            return PriorityConstants.MAX_EXPLORATION_BONUS  # 未扰动过，最高奖励
        elif self.perturbation_count < max_perturbations:
            # 线性衰减
            decay_ratio = (self.perturbation_count / max_perturbations) * PriorityConstants.EXPLORATION_DECAY_FACTOR
            return PriorityConstants.MAX_EXPLORATION_BONUS - decay_ratio
        else:
            # 扰动过多，最低奖励
            return PriorityConstants.MIN_EXPLORATION_BONUS
    
    def get_historical_performance(self, perturbation_history: Dict[str, List[float]]) -> float:
        """获取历史表现分数 (0.0-1.0)。
        
        基于历史扰动影响的加权平均。
        
        Args:
            perturbation_history: {token_text: [impact1, impact2, ...]}
            
        Returns:
            历史表现分数 (0.0-1.0)
        """
        if self.text not in perturbation_history:
            return PriorityConstants.DEFAULT_PERFORMANCE_SCORE  # 默认中等表现
        
        impacts = perturbation_history[self.text]
        if not impacts:
            return PriorityConstants.DEFAULT_PERFORMANCE_SCORE
        
        # 计算加权平均 (近期权重更高)
        weights = [PriorityConstants.HISTORICAL_WEIGHT_BASE ** i for i in range(len(impacts))]
        weights.reverse()  # 最近的权重最高
        
        total_weight = sum(weights)
        if total_weight == 0:
            return PriorityConstants.DEFAULT_PERFORMANCE_SCORE
        
        weighted_avg = sum(w * i for w, i in zip(weights, impacts)) / total_weight
        return max(0.0, min(1.0, weighted_avg))
    
    def get_final_priority(self, manager: 'DynamicPriorityManager') -> float:
        """计算最终优先级分数。
        
        由DynamicPriorityManager调用，整合所有维度的分数。
        
        Args:
            manager: 动态优先级管理器
            
        Returns:
            最终优先级分数
        """
        return manager.calculate_priority(self)
    
    def update_after_perturbation(self, iteration: int, impact: float):
        """扰动后更新Token状态。
        
        Args:
            iteration: 当前迭代次数
            impact: 扰动影响分数 (0.0-1.0)
            
        Raises:
            ValueError: If iteration is negative or impact is out of range
        """
        if iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {iteration}")
        if not 0.0 <= impact <= 1.0:
            raise ValueError(f"Impact must be between 0.0 and 1.0, got {impact}")
        
        self.perturbation_count += 1
        self.last_perturbed_iter = iteration
        self.cumulative_impact += impact
        self._invalidate_cache()
    
    def reset_perturbation_history(self):
        """重置扰动历史 (用于新一轮实验)。"""
        self.perturbation_count = 0
        self.last_perturbed_iter = -1
        self.cumulative_impact = 0.0
        self.feedback_score = 0.0
        self._invalidate_cache()
    
    @property
    def average_impact(self) -> float:
        """获取平均扰动影响。
        
        Returns:
            平均影响分数，如果从未扰动则返回0.0
        """
        if self.perturbation_count == 0:
            return 0.0
        return self.cumulative_impact / self.perturbation_count
    
    @property
    def is_unexplored(self) -> bool:
        """检查Token是否未被扰动过。
        
        Returns:
            True if token has never been perturbed
        """
        return self.perturbation_count == 0
    
    @property
    def total_score(self) -> float:
        """获取所有分数的总和。
        
        Returns:
            基础分数、SCS分数和反馈分数的总和
        """
        return self.priority_score + self.scs_score + self.feedback_score
    
    def _invalidate_cache(self):
        """清空缓存，在状态改变时调用。"""
        self._cache.clear()
    
    def _get_cached_or_compute(self, key: str, compute_func, *args, **kwargs):
        """获取缓存值或计算新值。"""
        if key not in self._cache:
            self._cache[key] = compute_func(*args, **kwargs)
        return self._cache[key]
    
    def __str__(self):
        return (f"Token('{self.text}', type={self.token_type}, line={self.line}, "
                f"priority={self.priority_score:.2f}, scs={self.scs_score:.2f}, "
                f"perturbed={self.perturbation_count})")

    def __repr__(self):
        return self.__str__()
