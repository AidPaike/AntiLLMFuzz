"""优雅的错误处理和用户指导系统"""

import logging
import sys
import traceback
from typing import Optional, Callable, Any, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class UserFriendlyError(Exception):
    """用户友好的错误类"""
    
    def __init__(self, message: str, suggestions: Optional[list] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.suggestions = suggestions or []
        self.error_code = error_code


class GracefulErrorHandler:
    """优雅错误处理器"""
    
    @staticmethod
    def handle_llm_failure(operation_name: str, fallback_action: Optional[Callable] = None) -> Any:
        """处理LLM相关失败"""
        logger.warning(f"⚠️  {operation_name} 需要LLM支持，但当前不可用")
        
        print(f"""
🤖 LLM功能不可用

💡 解决方案:
1. 安装Ollama:
   curl -fsSL https://ollama.ai/install.sh | sh
   
2. 下载推荐模型:
   ollama pull qwen2.5-coder
   
3. 或者配置OpenAI API:
   export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   
4. 继续使用模拟模式 (功能受限但可用)

📚 详细文档: https://github.com/project/docs/llm-setup.md
        """)
        
        if fallback_action:
            print("🔄 使用模拟模式继续...")
            return fallback_action()
        
        return None
    
    @staticmethod
    def handle_dependency_missing(dependency: str, install_command: Optional[str] = None):
        """处理依赖缺失"""
        print(f"""
❌ 缺少依赖: {dependency}

💡 解决方案:
""")
        
        if install_command:
            print(f"   {install_command}")
        else:
            print(f"   pip install {dependency}")
        
        print(f"""
📚 完整安装指南: docs/INSTALL.md
        """)
    
    @staticmethod
    def handle_configuration_error(config_file: str, missing_keys: Optional[list] = None):
        """处理配置错误"""
        print(f"""
⚙️  配置文件问题: {config_file}

💡 解决方案:
1. 检查配置文件是否存在
2. 验证配置格式 (YAML语法)
""")
        
        if missing_keys:
            print("3. 添加缺少的配置项:")
            for key in missing_keys:
                print(f"   - {key}")
        
        print(f"""
📝 配置模板: config/config.yaml.template
🔧 配置验证: python scripts/validate_config.py
        """)
        
        if missing_keys:
            print("3. 添加缺少的配置项:")
            for key in missing_keys:
                print(f"   - {key}")
        
        print(f"""
📝 配置模板: config/config.yaml.template
🔧 配置验证: python scripts/validate_config.py
        """)
    
    @staticmethod
    def handle_file_not_found(file_path: str, suggestions: Optional[list] = None):
        """处理文件未找到"""
        print(f"""
📁 文件未找到: {file_path}

💡 解决方案:
1. 检查文件路径是否正确
2. 确认文件是否存在
""")
        
        if suggestions:
            print("3. 可能的替代文件:")
            for suggestion in suggestions:
                print(f"   - {suggestion}")
        
        print(f"""
📚 示例文件: data/examples/
        """)


def graceful_error(error_type: str = "general"):
    """装饰器：为函数添加优雅的错误处理"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                GracefulErrorHandler.handle_file_not_found(str(e))
                raise UserFriendlyError(
                    f"文件未找到: {e}",
                    suggestions=["检查文件路径", "确认文件存在"]
                )
            except ImportError as e:
                missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
                GracefulErrorHandler.handle_dependency_missing(missing_module)
                raise UserFriendlyError(
                    f"缺少依赖: {missing_module}",
                    suggestions=[f"pip install {missing_module}"]
                )
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行失败: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
                raise
        return wrapper
    return decorator


class ProgressReporter:
    """进度报告器 - 提供用户友好的进度反馈"""
    
    def __init__(self, total_steps: int, description: str = "处理中"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.step_descriptions: Dict[int, str] = {}
    
    def set_step_description(self, step: int, description: str):
        """设置步骤描述"""
        self.step_descriptions[step] = description
    
    def start_step(self, step_name: Optional[str] = None):
        """开始新步骤"""
        self.current_step += 1
        
        if step_name:
            desc = step_name
        else:
            desc = self.step_descriptions.get(self.current_step, f"步骤 {self.current_step}")
        
        progress = (self.current_step / self.total_steps) * 100
        print(f"\r[{progress:5.1f}%] {desc}...", end="", flush=True)
    
    def complete_step(self, success: bool = True, message: Optional[str] = None):
        """完成当前步骤"""
        status = "✅" if success else "❌"
        if message:
            print(f"\r{status} {message}")
        else:
            print(f"\r{status} 完成")
    
    def finish(self, success: bool = True):
        """完成所有步骤"""
        if success:
            print(f"\n🎉 {self.description} 完成！")
        else:
            print(f"\n💥 {self.description} 失败")


class ValidationHelper:
    """验证助手 - 提供环境和配置验证"""
    
    @staticmethod
    def check_python_version(min_version: tuple = (3, 8)) -> bool:
        """检查Python版本"""
        current = sys.version_info[:2]
        if current >= min_version:
            print(f"✅ Python版本: {'.'.join(map(str, current))}")
            return True
        else:
            print(f"❌ Python版本过低: {'.'.join(map(str, current))} < {'.'.join(map(str, min_version))}")
            print(f"💡 请升级到Python {'.'.join(map(str, min_version))}+")
            return False
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """检查依赖包"""
        required_packages = {
            'spacy': 'spacy>=3.7.2',
            'javalang': 'javalang>=0.13.0',
            'yaml': 'pyyaml>=6.0.1',
            'requests': 'requests>=2.31.0'
        }
        
        results = {}
        for package, requirement in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {requirement}")
                results[package] = True
            except ImportError:
                print(f"❌ {requirement}")
                results[package] = False
        
        return results
    
    @staticmethod
    def check_spacy_model(model: str = "en_core_web_sm") -> bool:
        """检查spaCy模型"""
        try:
            import spacy
            nlp = spacy.load(model)
            print(f"✅ spaCy模型: {model}")
            return True
        except OSError:
            print(f"❌ spaCy模型缺失: {model}")
            print(f"💡 安装命令: python -m spacy download {model}")
            return False
        except ImportError:
            print("❌ spaCy未安装")
            print("💡 安装命令: pip install spacy")
            return False
    
    @staticmethod
    def check_llm_backends() -> Dict[str, bool]:
        """检查LLM后端"""
        from src.utils.llm_client_manager import LLMClientManager
        
        manager = LLMClientManager()
        status = manager.get_status()
        
        print("🤖 LLM后端状态:")
        results = {}
        for client_info in status['available_clients']:
            name = client_info['name']
            available = client_info['available']
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {name}")
            results[name] = available
        
        current = status.get('current_client')
        if current:
            print(f"🎯 当前使用: {current}")
        
        return results
    
    @staticmethod
    def check_test_data() -> bool:
        """检查测试数据"""
        import os
        test_file = "data/00java_std.md"
        
        if os.path.exists(test_file):
            print(f"✅ 测试数据: {test_file}")
            return True
        else:
            print(f"❌ 测试数据缺失: {test_file}")
            print("💡 请确保data目录包含示例文件")
            return False


def print_startup_banner():
    """打印启动横幅"""
    print("""
🎯 LLM Fuzzer Semantic Disruptor
================================================================================
💡 首次使用？运行: anti_llm4fuzz --demo --input data/00java_std.md
🔧 环境检查: python scripts/validate_environment.py
📚 帮助文档: anti_llm4fuzz --help
================================================================================
    """)


def print_completion_summary(results: Dict[str, Any]):
    """打印完成摘要"""
    print("""
================================================================================
🎉 处理完成！
================================================================================""")
    
    if 'output_files' in results:
        print(f"📁 输出文件: {len(results['output_files'])} 个")
        for file in results['output_files'][:3]:  # 显示前3个
            print(f"   - {file}")
        if len(results['output_files']) > 3:
            print(f"   ... 还有 {len(results['output_files']) - 3} 个文件")
    
    if 'output_dir' in results:
        print(f"📂 输出目录: {results['output_dir']}")
    
    if 'metadata_file' in results:
        print(f"📊 元数据: {results['metadata_file']}")
    
    print("\n💡 下一步:")
    print("   - 查看生成的文件")
    print("   - 运行可视化: python tools/visualize_perturbation.py")
    print("   - 尝试其他策略: --strategy semantic")
    print("================================================================================")