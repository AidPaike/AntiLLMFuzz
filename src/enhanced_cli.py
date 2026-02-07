"""重构的CLI接口 - 提供用户友好的命令行体验"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.error_handler import (
    print_startup_banner, 
    print_completion_summary,
    GracefulErrorHandler,
    ProgressReporter,
    graceful_error
)
from src.utils.llm_client_manager import get_llm_manager


class EnhancedCLI:
    """增强的CLI接口"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            description="🎯 LLM Fuzzer Semantic Disruptor",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
🚀 快速开始:
  %(prog)s --demo --input data/00java_std.md           # 演示模式
  %(prog)s --input your_file.md --strategy semantic   # 语义策略
  %(prog)s --validate                                  # 环境验证

📚 更多示例:
  %(prog)s --input doc.md --top-n 10 --strategy all   # 完整测试
  %(prog)s --input code.java --strategy tokenization_drift --visualize

🔗 项目主页: https://github.com/project/llm-fuzzer-disruptor
            """
        )
        
        # 基础参数
        parser.add_argument(
            '--input', '-i',
            type=str,
            help='输入文件路径 (支持 .md, .java, .py)'
        )
        
        parser.add_argument(
            '--output-dir', '-o',
            type=str,
            default='output',
            help='输出目录 (默认: output)'
        )
        
        parser.add_argument(
            '--top-n', '-n',
            type=int,
            default=5,
            help='要扰动的顶级token数量 (默认: 5)'
        )
        
        # 策略选择
        strategies = [
            'tokenization_drift', 'lexical_disguise', 'dataflow_misdirection',
            'controlflow_misdirection', 'documentation_deception', 'cognitive_manipulation',
            'formatting_noise', 'structural_noise', 'paraphrasing', 'cognitive_load',
            'semantic', 'generic', 'all', 'auto'
        ]
        
        parser.add_argument(
            '--strategy', '-s',
            choices=strategies,
            default='auto',
            help='扰动策略 (默认: auto - 自动选择最佳策略)'
        )
        
        # 模式选择
        parser.add_argument(
            '--demo',
            action='store_true',
            help='演示模式: 使用最佳参数快速展示功能'
        )
        
        parser.add_argument(
            '--validate',
            action='store_true',
            help='验证环境配置并退出'
        )
        
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='交互模式: 逐步引导用户操作'
        )
        
        # 高级选项
        parser.add_argument(
            '--visualize',
            action='store_true',
            help='生成可视化报告'
        )
        
        parser.add_argument(
            '--enable-scs',
            action='store_true',
            help='启用SCS (语义贡献分数) 计算'
        )
        
        parser.add_argument(
            '--use-mock-llm',
            action='store_true',
            help='强制使用模拟LLM (用于测试)'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='自定义配置文件路径'
        )
        
        # 日志和调试
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='日志级别 (默认: INFO)'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='静默模式: 只显示错误信息'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='详细模式: 显示详细进度信息'
        )
        
        # 版本信息
        parser.add_argument(
            '--version',
            action='version',
            version='LLM Fuzzer Semantic Disruptor v1.0.0'
        )
        
        return parser
    
    @graceful_error("cli")
    def run(self, args: Optional[List[str]] = None) -> int:
        """运行CLI应用"""
        parsed_args = self.parser.parse_args(args)
        
        # 不在静默模式下显示横幅
        if not parsed_args.quiet:
            print_startup_banner()
        
        # 环境验证模式
        if parsed_args.validate:
            return self._run_validation()
        
        # 交互模式
        if parsed_args.interactive:
            return self._run_interactive_mode()
        
        # 检查必需参数
        if not parsed_args.input:
            if parsed_args.demo:
                # 演示模式使用默认文件
                parsed_args.input = "data/00java_std.md"
                parsed_args.strategy = "tokenization_drift"
                parsed_args.top_n = 3
                parsed_args.visualize = True
                print("🎬 演示模式: 使用默认参数")
            else:
                self.parser.error("需要指定输入文件 (--input) 或使用 --demo 模式")
        
        # 验证输入文件
        if not Path(parsed_args.input).exists():
            GracefulErrorHandler.handle_file_not_found(
                parsed_args.input,
                suggestions=["data/00java_std.md", "examples/sample.java"]
            )
            return 1
        
        # 运行主要处理流程
        return self._run_perturbation(parsed_args)
    
    def _run_validation(self) -> int:
        """运行环境验证"""
        from scripts.validate_environment import main as validate_main
        return validate_main()
    
    def _run_interactive_mode(self) -> int:
        """运行交互模式"""
        print("🤝 交互模式")
        print("我将引导您完成扰动测试的设置...\n")
        
        # 1. 选择输入文件
        input_file = self._interactive_select_input()
        if not input_file:
            return 1
        
        # 2. 选择策略
        strategy = self._interactive_select_strategy(input_file)
        
        # 3. 选择参数
        top_n = self._interactive_select_top_n()
        
        # 4. 确认并运行
        print(f"\n📋 配置摘要:")
        print(f"   输入文件: {input_file}")
        print(f"   策略: {strategy}")
        print(f"   Token数量: {top_n}")
        
        confirm = input("\n确认运行? (y/N): ").lower().strip()
        if confirm != 'y':
            print("❌ 已取消")
            return 0
        
        # 构造参数并运行
        from types import SimpleNamespace
        args = SimpleNamespace(
            input=input_file,
            strategy=strategy,
            top_n=top_n,
            output_dir='output',
            visualize=True,
            enable_scs=False,
            use_mock_llm=False,
            config=None,
            log_level='INFO',
            quiet=False,
            verbose=True
        )
        
        return self._run_perturbation(args)
    
    def _interactive_select_input(self) -> Optional[str]:
        """交互式选择输入文件"""
        print("1️⃣ 选择输入文件:")
        
        # 扫描可用文件
        available_files = []
        for pattern in ['data/*.md', 'examples/*.java', 'examples/*.py']:
            available_files.extend(Path('.').glob(pattern))
        
        if available_files:
            print("   可用文件:")
            for i, file in enumerate(available_files[:5], 1):
                print(f"   {i}. {file}")
            print("   0. 自定义路径")
            
            choice = input("\n请选择 (1-5, 0): ").strip()
            
            if choice == '0':
                custom_path = input("请输入文件路径: ").strip()
                return custom_path if Path(custom_path).exists() else None
            elif choice.isdigit() and 1 <= int(choice) <= len(available_files):
                return str(available_files[int(choice) - 1])
        
        # 没有可用文件或选择无效
        custom_path = input("请输入文件路径: ").strip()
        return custom_path if Path(custom_path).exists() else None
    
    def _interactive_select_strategy(self, input_file: str) -> str:
        """交互式选择策略"""
        print("\n2️⃣ 选择扰动策略:")
        
        file_ext = Path(input_file).suffix.lower()
        
        if file_ext == '.md':
            strategies = [
                ('tokenization_drift', '🎯 Token边界干扰 (推荐)'),
                ('documentation_deception', '📝 文档欺骗'),
                ('cognitive_manipulation', '🧠 认知操控'),
                ('semantic', '🔄 所有语义策略'),
                ('all', '🌟 所有策略')
            ]
        else:
            strategies = [
                ('tokenization_drift', '🎯 Token边界干扰'),
                ('lexical_disguise', '🎭 词汇伪装'),
                ('semantic', '🔄 所有语义策略 (推荐)'),
                ('generic', '📄 通用策略'),
                ('all', '🌟 所有策略')
            ]
        
        for i, (strategy, desc) in enumerate(strategies, 1):
            print(f"   {i}. {desc}")
        
        choice = input(f"\n请选择 (1-{len(strategies)}): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(strategies):
            return strategies[int(choice) - 1][0]
        
        return 'auto'  # 默认自动选择
    
    def _interactive_select_top_n(self) -> int:
        """交互式选择token数量"""
        print("\n3️⃣ 选择要扰动的token数量:")
        print("   1. 3个 (快速测试)")
        print("   2. 5个 (推荐)")
        print("   3. 10个 (详细分析)")
        print("   4. 自定义")
        
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            return 3
        elif choice == '2':
            return 5
        elif choice == '3':
            return 10
        elif choice == '4':
            custom = input("请输入数量 (1-50): ").strip()
            if custom.isdigit() and 1 <= int(custom) <= 50:
                return int(custom)
        
        return 5  # 默认值
    
    @graceful_error("perturbation")
    def _run_perturbation(self, args) -> int:
        """运行扰动处理"""
        # 初始化进度报告器
        reporter = ProgressReporter(8, "扰动处理")
        
        try:
            # 导入主要模块
            reporter.start_step("初始化系统")
            from main import PerturbationPipeline
            reporter.complete_step(True, "系统初始化完成")
            
            # 创建流水线
            reporter.start_step("创建处理流水线")
            
            # 构造args对象以匹配原始接口
            import argparse
            pipeline_args = argparse.Namespace(
                input=args.input,
                output_dir=args.output_dir,
                top_n=args.top_n,
                strategy=args.strategy,
                log_level=args.log_level,
                enable_scs=getattr(args, 'enable_scs', False),
                enable_fuzzer_integration=False,
                crm_profile=getattr(args, 'crm_profile', 'generic'),
                test_llm=False,
                version=False
            )
            
            pipeline = PerturbationPipeline(pipeline_args)
            reporter.complete_step(True, "流水线创建完成")
            
            # 配置LLM
            if hasattr(args, 'use_mock_llm') and args.use_mock_llm:
                reporter.start_step("配置Mock LLM")
                # 强制使用Mock模式的逻辑
                reporter.complete_step(True, "Mock LLM已配置")
            else:
                reporter.start_step("检查LLM后端")
                llm_manager = get_llm_manager()
                if llm_manager.is_mock_mode():
                    print("   ⚠️ 使用模拟LLM模式")
                reporter.complete_step(True, "LLM后端已就绪")
            
            # 运行处理
            reporter.start_step("执行扰动处理")
            exit_code = pipeline.execute()
            if exit_code == 0:
                reporter.complete_step(True, "扰动处理完成")
                # 构造结果字典
                result = {
                    'input_file': args.input,
                    'output_dir': args.output_dir,
                    'output_files': [],  # 需要从pipeline获取
                    'metadata_file': None
                }
            else:
                reporter.complete_step(False, "扰动处理失败")
                return exit_code
            
            # 生成可视化
            if hasattr(args, 'visualize') and args.visualize:
                reporter.start_step("生成可视化报告")
                self._generate_visualization(result)
                reporter.complete_step(True, "可视化报告已生成")
            
            # 完成
            reporter.finish(True)
            
            # 显示结果摘要
            if not args.quiet:
                print_completion_summary({
                    'output_files': result.get('output_files', []),
                    'output_dir': result.get('output_dir', ''),
                    'metadata_file': result.get('metadata_file', '')
                })
            
            return 0
            
        except Exception as e:
            reporter.complete_step(False, f"处理失败: {e}")
            reporter.finish(False)
            
            if args.verbose:
                import traceback
                print(f"\n🔍 详细错误信息:\n{traceback.format_exc()}")
            
            return 1
    
    def _generate_visualization(self, result):
        """生成可视化报告"""
        try:
            import subprocess
            
            if 'output_files' in result and result['output_files']:
                original_file = result.get('input_file')
                perturbed_file = result['output_files'][0]
                
                # 调用现有的可视化脚本
                cmd = ['python', 'tools/visualize_perturbation.py', original_file, perturbed_file]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"   📊 可视化完成: {perturbed_file}")
                
        except ImportError:
            print("   ⚠️ 可视化功能不可用 (缺少依赖)")
        except Exception as e:
            print(f"   ⚠️ 可视化生成失败: {e}")


def main(args: Optional[List[str]] = None) -> int:
    """CLI主入口点"""
    cli = EnhancedCLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())