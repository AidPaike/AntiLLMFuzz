"""Priority calculation strategies for Token objects."""

from abc import ABC, abstractmethod
from typing import Dict, List
import math


class PriorityCalculationStrategy(ABC):
    """Abstract base class for priority calculation strategies."""
    
    @abstractmethod
    def calculate(self, token: 'Token', **kwargs) -> float:
        """Calculate priority score for a token."""
        pass


class NoveltyScoreStrategy(PriorityCalculationStrategy):
    """Strategy for calculating novelty scores."""
    
    def __init__(self, min_score: float = 0.2, count_weight: float = 0.6, 
                 time_weight: float = 0.4, decay_rate: float = 0.1):
        self.min_score = min_score
        self.count_weight = count_weight
        self.time_weight = time_weight
        self.decay_rate = decay_rate
    
    def calculate(self, token: 'Token', current_iter: int, recovery_rounds: int = 5) -> float:
        """Calculate novelty score based on perturbation history and time."""
        # 基于扰动次数的指数衰减
        count_penalty = math.exp(-self.decay_rate * token.perturbation_count)
        
        # 基于时间的恢复
        if token.last_perturbed_iter >= 0:
            time_gap = current_iter - token.last_perturbed_iter
            time_recovery = min(1.0, time_gap / recovery_rounds)
        else:
            time_recovery = 1.0  # 从未扰动，完全新颖
        
        # 综合计算
        novelty = (count_penalty * self.count_weight + 
                  time_recovery * self.time_weight)
        
        return max(self.min_score, novelty)


class ExplorationBonusStrategy(PriorityCalculationStrategy):
    """Strategy for calculating exploration bonuses."""
    
    def __init__(self, min_bonus: float = 0.1, max_bonus: float = 1.0, 
                 decay_factor: float = 0.5):
        self.min_bonus = min_bonus
        self.max_bonus = max_bonus
        self.decay_factor = decay_factor
    
    def calculate(self, token: 'Token', max_perturbations: int = 3) -> float:
        """Calculate exploration bonus based on perturbation count."""
        if token.perturbation_count == 0:
            return self.max_bonus  # 未扰动过，最高奖励
        elif token.perturbation_count < max_perturbations:
            # 线性衰减
            decay_ratio = (token.perturbation_count / max_perturbations) * self.decay_factor
            return self.max_bonus - decay_ratio
        else:
            # 扰动过多，最低奖励
            return self.min_bonus


class HistoricalPerformanceStrategy(PriorityCalculationStrategy):
    """Strategy for calculating historical performance scores."""
    
    def __init__(self, default_score: float = 0.5, weight_base: float = 0.5):
        self.default_score = default_score
        self.weight_base = weight_base
    
    def calculate(self, token: 'Token', perturbation_history: Dict[str, List[float]]) -> float:
        """Calculate historical performance based on weighted average of past impacts."""
        if token.text not in perturbation_history:
            return self.default_score
        
        impacts = perturbation_history[token.text]
        if not impacts:
            return self.default_score
        
        # 计算加权平均 (近期权重更高)
        weights = [self.weight_base ** i for i in range(len(impacts))]
        weights.reverse()  # 最近的权重最高
        
        total_weight = sum(weights)
        if total_weight == 0:
            return self.default_score
        
        weighted_avg = sum(w * i for w, i in zip(weights, impacts)) / total_weight
        return max(0.0, min(1.0, weighted_avg))