"""Performance metrics for priority management system."""

import math
from typing import List, Dict, Any
from collections import Counter
from src.data_models import Token


class PriorityMetrics:
    """优先级系统性能监控。
    
    跟踪选择多样性、探索比例、分数分布等关键指标。
    """
    
    def __init__(self):
        """初始化指标收集器。"""
        self.metrics_history = []
        self.selection_history = []
    
    def track_selection(self, selected_tokens: List[Token], 
                       all_tokens: List[Token], 
                       iteration: int,
                       selection_method: str = "unknown") -> Dict[str, float]:
        """跟踪一次Token选择的指标。
        
        Args:
            selected_tokens: 选中的Token列表
            all_tokens: 所有候选Token列表
            iteration: 当前迭代次数
            selection_method: 选择方法
            
        Returns:
            本次选择的指标字典
        """
        metrics = {
            'iteration': iteration,
            'selection_method': selection_method,
            'selected_count': len(selected_tokens),
            'total_count': len(all_tokens)
        }
        
        # 1. 选择多样性
        metrics['diversity_index'] = self.calculate_diversity_index(selected_tokens)
        
        # 2. 探索比例
        metrics['exploration_ratio'] = self.calculate_exploration_ratio(selected_tokens)
        
        # 3. 分数分布
        score_stats = self.calculate_score_distribution(all_tokens)
        metrics.update(score_stats)
        
        # 4. 新颖度统计
        novelty_stats = self.calculate_novelty_stats(selected_tokens, iteration)
        metrics.update(novelty_stats)
        
        # 5. 类型分布
        type_stats = self.calculate_type_distribution(selected_tokens)
        metrics.update(type_stats)
        
        # 记录历史
        self.metrics_history.append(metrics)
        self.selection_history.append([t.text for t in selected_tokens])
        
        return metrics
    
    def calculate_diversity_index(self, tokens: List[Token]) -> float:
        """计算选择多样性 (Shannon熵)。
        
        基于Token文本的多样性。
        
        Args:
            tokens: Token列表
            
        Returns:
            多样性指数 (0.0 到 log2(n))
        """
        if not tokens:
            return 0.0
        
        # 统计Token文本频率
        text_counts = Counter(token.text for token in tokens)
        total = len(tokens)
        
        # 计算Shannon熵
        entropy = 0.0
        for count in text_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_exploration_ratio(self, tokens: List[Token]) -> float:
        """计算探索比例。
        
        未扰动过的Token比例。
        
        Args:
            tokens: Token列表
            
        Returns:
            探索比例 (0.0-1.0)
        """
        if not tokens:
            return 0.0
        
        unexplored_count = sum(1 for token in tokens if token.perturbation_count == 0)
        return unexplored_count / len(tokens)
    
    def calculate_score_distribution(self, tokens: List[Token]) -> Dict[str, float]:
        """计算分数分布统计。
        
        Args:
            tokens: Token列表
            
        Returns:
            分数统计字典
        """
        if not tokens:
            return {
                'score_mean': 0.0,
                'score_std': 0.0,
                'score_min': 0.0,
                'score_max': 0.0,
                'score_range': 0.0
            }
        
        # 收集所有优先级分数
        scores = [token.priority_score for token in tokens]
        
        # 计算统计量
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = math.sqrt(variance)
        min_score = min(scores)
        max_score = max(scores)
        
        return {
            'score_mean': mean_score,
            'score_std': std_score,
            'score_min': min_score,
            'score_max': max_score,
            'score_range': max_score - min_score
        }
    
    def calculate_novelty_stats(self, tokens: List[Token], 
                               current_iter: int) -> Dict[str, float]:
        """计算新颖度统计。
        
        Args:
            tokens: Token列表
            current_iter: 当前迭代次数
            
        Returns:
            新颖度统计字典
        """
        if not tokens:
            return {
                'novelty_mean': 0.0,
                'novelty_min': 0.0,
                'novelty_max': 0.0
            }
        
        # 计算所有Token的新颖度
        novelties = [token.get_novelty_score(current_iter) for token in tokens]
        
        return {
            'novelty_mean': sum(novelties) / len(novelties),
            'novelty_min': min(novelties),
            'novelty_max': max(novelties)
        }
    
    def calculate_type_distribution(self, tokens: List[Token]) -> Dict[str, float]:
        """计算Token类型分布。
        
        Args:
            tokens: Token列表
            
        Returns:
            类型分布字典
        """
        if not tokens:
            return {}
        
        # 统计类型频率
        type_counts = Counter(token.token_type for token in tokens)
        total = len(tokens)
        
        # 转换为比例
        type_ratios = {}
        for token_type, count in type_counts.items():
            type_ratios[f'type_ratio_{token_type}'] = count / total
        
        return type_ratios
    
    def get_convergence_metrics(self, window_size: int = 5) -> Dict[str, float]:
        """计算收敛指标。
        
        分析最近几次选择的稳定性。
        
        Args:
            window_size: 分析窗口大小
            
        Returns:
            收敛指标字典
        """
        if len(self.selection_history) < window_size:
            return {
                'convergence_rate': 0.0,
                'selection_stability': 0.0,
                'diversity_trend': 0.0
            }
        
        recent_selections = self.selection_history[-window_size:]
        recent_metrics = self.metrics_history[-window_size:]
        
        # 1. 选择稳定性 (重复选择的Token比例)
        all_tokens = set()
        overlap_counts = []
        
        for i, selection in enumerate(recent_selections):
            all_tokens.update(selection)
            if i > 0:
                prev_selection = set(recent_selections[i-1])
                curr_selection = set(selection)
                overlap = len(prev_selection & curr_selection)
                total = len(prev_selection | curr_selection)
                overlap_counts.append(overlap / total if total > 0 else 0)
        
        selection_stability = sum(overlap_counts) / len(overlap_counts) if overlap_counts else 0
        
        # 2. 多样性趋势
        diversities = [m['diversity_index'] for m in recent_metrics]
        if len(diversities) >= 2:
            diversity_trend = (diversities[-1] - diversities[0]) / (len(diversities) - 1)
        else:
            diversity_trend = 0.0
        
        # 3. 收敛率 (基于分数变化)
        if len(recent_metrics) >= 2:
            score_changes = []
            for i in range(1, len(recent_metrics)):
                prev_mean = recent_metrics[i-1]['score_mean']
                curr_mean = recent_metrics[i]['score_mean']
                if prev_mean > 0:
                    change = abs(curr_mean - prev_mean) / prev_mean
                    score_changes.append(change)
            
            convergence_rate = 1.0 - (sum(score_changes) / len(score_changes)) if score_changes else 1.0
        else:
            convergence_rate = 0.0
        
        return {
            'convergence_rate': max(0.0, min(1.0, convergence_rate)),
            'selection_stability': selection_stability,
            'diversity_trend': diversity_trend
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告。
        
        Returns:
            汇总报告字典
        """
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.metrics_history[-1]
        convergence_metrics = self.get_convergence_metrics()
        
        # 计算历史趋势
        all_diversities = [m['diversity_index'] for m in self.metrics_history]
        all_exploration_ratios = [m['exploration_ratio'] for m in self.metrics_history]
        
        return {
            'status': 'active',
            'total_iterations': len(self.metrics_history),
            'latest_metrics': latest_metrics,
            'convergence_metrics': convergence_metrics,
            'trends': {
                'diversity_mean': sum(all_diversities) / len(all_diversities),
                'diversity_latest': all_diversities[-1],
                'exploration_mean': sum(all_exploration_ratios) / len(all_exploration_ratios),
                'exploration_latest': all_exploration_ratios[-1]
            },
            'recommendations': self._generate_recommendations(latest_metrics, convergence_metrics)
        }
    
    def _generate_recommendations(self, latest_metrics: Dict[str, float], 
                                convergence_metrics: Dict[str, float]) -> List[str]:
        """生成优化建议。
        
        Args:
            latest_metrics: 最新指标
            convergence_metrics: 收敛指标
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 多样性检查
        if latest_metrics['diversity_index'] < 1.0:
            recommendations.append("选择多样性较低，建议增加探索率")
        
        # 探索比例检查
        if latest_metrics['exploration_ratio'] < 0.1:
            recommendations.append("探索比例过低，可能陷入局部最优")
        
        # 收敛检查
        if convergence_metrics['convergence_rate'] > 0.9:
            recommendations.append("系统已收敛，可以考虑重置或调整参数")
        
        # 稳定性检查
        if convergence_metrics['selection_stability'] > 0.8:
            recommendations.append("选择过于稳定，建议增加随机性")
        
        if not recommendations:
            recommendations.append("系统运行正常，指标健康")
        
        return recommendations
    
    def reset(self):
        """重置指标历史。"""
        self.metrics_history.clear()
        self.selection_history.clear()