#!/usr/bin/env python3
"""环境验证脚本 - 检查项目运行环境"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.error_handler import ValidationHelper, print_startup_banner


def main():
    """主验证函数"""
    print_startup_banner()
    
    print("🔍 环境验证开始...\n")
    
    all_checks_passed = True
    
    # 1. Python版本检查
    print("1️⃣ Python环境检查")
    if not ValidationHelper.check_python_version():
        all_checks_passed = False
    print()
    
    # 2. 依赖包检查
    print("2️⃣ 依赖包检查")
    deps_result = ValidationHelper.check_dependencies()
    if not all(deps_result.values()):
        all_checks_passed = False
        missing = [pkg for pkg, status in deps_result.items() if not status]
        print(f"💡 安装缺失依赖: pip install {' '.join(missing)}")
    print()
    
    # 3. spaCy模型检查
    print("3️⃣ spaCy模型检查")
    if not ValidationHelper.check_spacy_model():
        all_checks_passed = False
    print()
    
    # 4. LLM后端检查
    print("4️⃣ LLM后端检查")
    llm_results = ValidationHelper.check_llm_backends()
    # LLM不是必需的，只要有Mock就行
    if not any(llm_results.values()):
        print("⚠️ 没有可用的LLM后端，但Mock模式可用")
    print()
    
    # 5. 测试数据检查
    print("5️⃣ 测试数据检查")
    if not ValidationHelper.check_test_data():
        all_checks_passed = False
    print()
    
    # 总结
    print("="*80)
    if all_checks_passed:
        print("🎉 环境验证通过！可以开始使用项目")
        print("\n💡 快速开始:")
        print("   anti_llm4fuzz --demo --input data/00java_std.md")
    else:
        print("⚠️ 环境验证发现问题，请根据上述提示修复")
        print("\n📚 详细安装指南: docs/INSTALL.md")
    print("="*80)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())