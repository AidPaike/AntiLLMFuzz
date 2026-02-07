"""Cleaner argument parsing following project patterns."""

import argparse
from typing import Dict, Any
from .constants import DEFAULTS, APP


class ArgumentParserBuilder:
    """Builder for creating argument parser with fluent interface."""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="LLM Fuzzer Semantic Disruptor - Token Extraction and Perturbation Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        self._add_core_arguments()
        self._add_optional_arguments()
    
    def _add_core_arguments(self) -> None:
        """Add core required arguments."""
        self.parser.add_argument(
            '--input', '-i',
            type=str,
            default=DEFAULTS.INPUT_FILE,
            help='输入文件路径（文档、Java或Python文件）'
        )
        
        self.parser.add_argument(
            '--top-n', '-n',
            type=int,
            default=DEFAULTS.TOP_N,
            help=f'要扰动的顶级token数量（默认：{DEFAULTS.TOP_N}）'
        )
        
        self.parser.add_argument(
            '--strategy', '-s',
            type=str,
            default=DEFAULTS.STRATEGY,
            help=f'要使用的扰动策略（默认：{DEFAULTS.STRATEGY}）'
        )
    
    def _add_optional_arguments(self) -> None:
        """Add optional configuration arguments."""
        self.parser.add_argument(
            '--output-dir', '-o',
            type=str,
            default=DEFAULTS.OUTPUT_DIR,
            help=f'基础输出目录（默认：{DEFAULTS.OUTPUT_DIR}）'
        )
        
        self.parser.add_argument(
            '--log-level',
            type=str,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default=DEFAULTS.LOG_LEVEL,
            help=f'日志级别（默认：{DEFAULTS.LOG_LEVEL}）'
        )
        
        self.parser.add_argument(
            '--version', '-V',
            action='version',
            version=APP.VERSION
        )
        
        self.parser.add_argument(
            '--test-llm',
            action='store_true',
            help='测试LLM API连接并退出'
        )
        
        self.parser.add_argument(
            '--enable-fuzzer-integration',
            action='store_true',
            help='启用完整的模糊测试器集成和扰动影响分析'
        )
    
    def _get_usage_examples(self) -> str:
        """Get formatted usage examples."""
        return """
使用示例:
  # 使用tokenization_drift策略提取和扰动文档
  python main.py --input data/doc.md --top-n 5 --strategy tokenization_drift
  
  # 对Java文件应用所有策略
  python main.py --input src/Main.java --top-n 3 --strategy all
  
  # 仅应用语义策略
  python main.py --input data/doc.md --strategy semantic
  
  # 使用认知操控策略并启用调试日志
  python main.py --input script.py --strategy cognitive_manipulation --log-level DEBUG
        """
    
    def build(self) -> argparse.ArgumentParser:
        """Build and return the configured parser."""
        return self.parser


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments using builder pattern."""
    parser = ArgumentParserBuilder().build()
    return parser.parse_args()