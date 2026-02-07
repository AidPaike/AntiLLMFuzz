"""Hotspot analyzer for ranking and analyzing tokens by SCS scores."""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from src.data_models import Token


class HotspotAnalyzer:
    """Analyzes, ranks, and visualizes semantic hotspots based on SCS scores.
    
    The analyzer provides methods to rank tokens, calculate statistics,
    and identify high-impact semantic regions.
    """
    
    def __init__(self, tokens: List[Token]):
        """Initialize analyzer with tokens.
        
        Args:
            tokens: List of Token objects with SCS scores
        """
        self.tokens = tokens
    
    def rank_by_scs(
        self,
        use_priority_fallback: bool = True
    ) -> List[Token]:
        """Rank tokens by SCS score in descending order.
        
        Args:
            use_priority_fallback: If True, use priority score as secondary sort key
            
        Returns:
            Sorted list of tokens (highest SCS first)
        """
        if use_priority_fallback:
            # Primary sort by SCS, secondary by priority
            return sorted(
                self.tokens,
                key=lambda t: (t.scs_score, t.priority_score),
                reverse=True
            )
        else:
            # Sort only by SCS
            return sorted(
                self.tokens,
                key=lambda t: t.scs_score,
                reverse=True
            )
    
    def get_top_n(
        self,
        n: int,
        use_priority_fallback: bool = True
    ) -> List[Token]:
        """Get top N tokens by SCS score.
        
        Args:
            n: Number of tokens to return
            use_priority_fallback: If True, use priority score as secondary sort key
            
        Returns:
            List of top N tokens
        """
        ranked = self.rank_by_scs(use_priority_fallback)
        return ranked[:min(n, len(ranked))]
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate summary statistics for SCS scores.
        
        Returns:
            Dictionary with mean, median, max, min, std_dev
        """
        if not self.tokens:
            return {
                'mean': 0.0,
                'median': 0.0,
                'max': 0.0,
                'min': 0.0,
                'std_dev': 0.0,
                'count': 0
            }
        
        scs_scores = [t.scs_score for t in self.tokens]
        
        stats = {
            'mean': statistics.mean(scs_scores),
            'median': statistics.median(scs_scores),
            'max': max(scs_scores),
            'min': min(scs_scores),
            'count': len(scs_scores)
        }
        
        # Calculate standard deviation (need at least 2 values)
        if len(scs_scores) >= 2:
            stats['std_dev'] = statistics.stdev(scs_scores)
        else:
            stats['std_dev'] = 0.0
        
        return stats
    
    def get_tokens_by_threshold(
        self,
        threshold: float,
        comparison: str = "greater"
    ) -> List[Token]:
        """Get tokens with SCS scores above/below threshold.
        
        Args:
            threshold: SCS score threshold
            comparison: 'greater', 'less', 'equal'
            
        Returns:
            List of tokens matching criteria
        """
        if comparison == "greater":
            return [t for t in self.tokens if t.scs_score > threshold]
        elif comparison == "less":
            return [t for t in self.tokens if t.scs_score < threshold]
        elif comparison == "equal":
            return [t for t in self.tokens if abs(t.scs_score - threshold) < 0.01]
        else:
            raise ValueError(f"Unknown comparison: {comparison}")
    
    def get_tokens_by_type(self) -> Dict[str, List[Token]]:
        """Group tokens by token type.
        
        Returns:
            Dictionary mapping token type to list of tokens
        """
        by_type: Dict[str, List[Token]] = {}
        
        for token in self.tokens:
            token_type = token.token_type
            if token_type not in by_type:
                by_type[token_type] = []
            by_type[token_type].append(token)
        
        # Sort each type's tokens by SCS score
        for token_type in by_type:
            by_type[token_type].sort(key=lambda t: t.scs_score, reverse=True)
        
        return by_type
    
    def get_tokens_by_line(self) -> Dict[int, List[Token]]:
        """Group tokens by line number.
        
        Returns:
            Dictionary mapping line number to list of tokens
        """
        by_line: Dict[int, List[Token]] = {}
        
        for token in self.tokens:
            line = token.line
            if line not in by_line:
                by_line[line] = []
            by_line[line].append(token)
        
        # Sort each line's tokens by SCS score
        for line in by_line:
            by_line[line].sort(key=lambda t: t.scs_score, reverse=True)
        
        return by_line
    
    def get_max_scs_per_line(self) -> Dict[int, float]:
        """Get maximum SCS score for each line.
        
        Returns:
            Dictionary mapping line number to max SCS score
        """
        by_line = self.get_tokens_by_line()
        return {
            line: max(t.scs_score for t in tokens)
            for line, tokens in by_line.items()
        }
    
    def format_token_display(
        self,
        token: Token,
        show_scs: bool = True,
        show_priority: bool = True
    ) -> str:
        """Format token for display.
        
        Args:
            token: Token to format
            show_scs: Include SCS score
            show_priority: Include priority score
            
        Returns:
            Formatted string
        """
        parts = []
        
        if show_scs:
            parts.append(f"[SCS: {token.scs_score:.1f}]")
        
        if show_priority:
            parts.append(f"[Priority: {token.priority_score:.1f}]")
        
        parts.append(f"{token.token_type:8s}")
        parts.append(f"| {token.text}")
        parts.append(f"(line {token.line})")
        
        return " ".join(parts)
    
    def print_ranking(
        self,
        top_n: Optional[int] = None,
        show_priority: bool = True
    ) -> None:
        """Print ranked tokens to console.
        
        Args:
            top_n: Number of top tokens to show (None = all)
            show_priority: Include priority scores
        """
        ranked = self.rank_by_scs()
        
        if top_n is not None:
            ranked = ranked[:top_n]
            print(f"\nTop {len(ranked)} Tokens by SCS Score:")
        else:
            print(f"\nAll {len(ranked)} Tokens by SCS Score:")
        
        print("-" * 80)
        
        for i, token in enumerate(ranked, 1):
            display = self.format_token_display(
                token,
                show_scs=True,
                show_priority=show_priority
            )
            print(f"{i:3d}. {display}")
    
    def print_statistics(self) -> None:
        """Print SCS statistics to console."""
        stats = self.get_statistics()
        
        print("\nSCS Statistics:")
        print("-" * 80)
        print(f"Total tokens:    {stats['count']}")
        print(f"Mean SCS:        {stats['mean']:.2f}")
        print(f"Median SCS:      {stats['median']:.2f}")
        print(f"Max SCS:         {stats['max']:.2f}")
        print(f"Min SCS:         {stats['min']:.2f}")
        print(f"Std Dev:         {stats['std_dev']:.2f}")
        
        # Count tokens in different ranges
        high = len(self.get_tokens_by_threshold(70.0, "greater"))
        medium = len(self.get_tokens_by_threshold(40.0, "greater")) - high
        low = stats['count'] - high - medium
        
        print(f"\nSCS Distribution:")
        print(f"  High (>70):    {high}")
        print(f"  Medium (40-70): {medium}")
        print(f"  Low (<40):     {low}")
    
    def generate_visualization(
        self,
        source_file: str,
        output_path: str
    ) -> None:
        """Generate annotated source file with hotspot markers.
        
        Args:
            source_file: Path to original source file
            output_path: Path for annotated output
            
        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Read source file
        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get max SCS per line
        max_scs_per_line = self.get_max_scs_per_line()
        
        # Generate annotated output
        output_lines = []
        output_lines.append(f"{source_path.name} (Semantic Hotspots)\n")
        output_lines.append("=" * 80 + "\n")
        
        for line_num, line_content in enumerate(lines, start=1):
            # Get SCS score for this line
            scs_score = max_scs_per_line.get(line_num, 0.0)
            
            # Generate visual indicator
            indicator = self._get_visual_indicator(scs_score)
            
            # Format line with annotation
            if scs_score > 0:
                annotation = f"[SCS: {scs_score:5.1f}] {indicator}"
            else:
                annotation = " " * 16  # Empty space for lines without tokens
            
            # Preserve original content
            annotated_line = f"{line_num:3d} | {line_content.rstrip():60s} {annotation}\n"
            output_lines.append(annotated_line)
        
        # Add legend
        output_lines.append("\n" + "=" * 80 + "\n")
        output_lines.append("Legend: ● (0-25) ●● (25-50) ●●● (50-75) ●●●● (75-100)\n")
        
        # Write output
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
    
    def _get_visual_indicator(self, scs_score: float) -> str:
        """Get visual indicator based on SCS score.
        
        Args:
            scs_score: SCS score (0-100)
            
        Returns:
            Visual indicator string
        """
        if scs_score >= 75.0:
            return "●●●●"
        elif scs_score >= 50.0:
            return "●●●"
        elif scs_score >= 25.0:
            return "●●"
        elif scs_score > 0:
            return "●"
        else:
            return ""
    
    def export_analysis(
        self,
        output_path: str,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        calculation_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Export SCS analysis to JSON.
        
        Args:
            output_path: Path for JSON output
            baseline_metrics: Optional baseline metrics dict
            calculation_params: Optional calculation parameters dict
        """
        # Organize tokens by type
        tokens_by_type = self.get_tokens_by_type()
        
        # Build JSON structure
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_file': self.tokens[0].source_file if self.tokens else '',
                'total_tokens': len(self.tokens)
            },
            'statistics': self.get_statistics(),
            'tokens_by_type': {},
            'tokens_by_line': {}
        }
        
        # Add baseline metrics if provided
        if baseline_metrics:
            export_data['baseline_metrics'] = baseline_metrics
        
        # Add calculation parameters if provided
        if calculation_params:
            export_data['calculation_params'] = calculation_params
        
        # Organize tokens by type
        for token_type, tokens in tokens_by_type.items():
            export_data['tokens_by_type'][token_type] = [
                self._token_to_dict(token) for token in tokens
            ]
        
        # Organize tokens by line
        tokens_by_line = self.get_tokens_by_line()
        for line_num, tokens in sorted(tokens_by_line.items()):
            export_data['tokens_by_line'][str(line_num)] = [
                self._token_to_dict(token) for token in tokens
            ]
        
        # Write JSON
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _token_to_dict(self, token: Token) -> Dict[str, Any]:
        """Convert Token to dictionary for JSON export.
        
        Args:
            token: Token to convert
            
        Returns:
            Dictionary representation
        """
        return {
            'text': token.text,
            'line': token.line,
            'column': token.column,
            'type': token.token_type,
            'priority_score': round(token.priority_score, 2),
            'scs_score': round(token.scs_score, 2)
        }
