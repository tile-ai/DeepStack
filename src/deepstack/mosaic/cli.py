#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepStack 命令行接口
"""

import argparse
import sys
from pathlib import Path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DeepStack - 3D建模和可视化工具包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  mosaic --version                    # 显示版本信息
  mosaic --help                       # 显示帮助信息
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="DeepStack 0.1.0"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示项目信息"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print("DeepStack - 3D建模和可视化工具包")
        print("版本: 0.1.0")
        print("功能模块:")
        print("  - 数据模块 (data)")
        print("  - LLM模块 (llm)")
        print("  - NOC模块 (noc)")
        print("  - 并行化模块 (parallelism)")
        print("  - 性能模块 (perf)")
        print("  - 工具模块 (utils)")
        print("  - 架构模块 (arch)")
        return 0
    
    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
