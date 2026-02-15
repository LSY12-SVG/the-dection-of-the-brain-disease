#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑肿瘤检测Web应用启动脚本
简化的启动入口，方便用户使用
"""

import sys
import os
from pathlib import Path

def main():
    """主函数"""
    print("=" * 60)
    print("🧠 脑肿瘤智能检测系统启动器")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"   当前版本：{sys.version}")
        sys.exit(1)
    
    # 检查依赖
    try:
        import flask
        import torch
        import timm
        import PIL
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖缺失：{e}")
        print("   请运行：pip install -r requirements.txt")
        sys.exit(1)
    
    # 检查模型文件
    model_path = Path("weights_eff/best.pt")
    if not model_path.exists():
        print(f"❌ 模型文件不存在：{model_path}")
        print("   请确保模型文件在正确位置")
        sys.exit(1)
    
    print(f"✅ 模型文件存在：{model_path}")
    
    # 启动应用
    try:
        from app import init_app, app, config
        
        print("\n🚀 正在启动Web应用...")
        print(f"   访问地址：http://{config.HOST}:{config.PORT}")
        print(f"   模型架构：{config.MODEL_NAME}")
        print(f"   图像尺寸：{config.IMG_SIZE}x{config.IMG_SIZE}")
        print(f"   判定阈值：{config.THRESHOLD}")
        print("\n按 Ctrl+C 停止服务")
        print("-" * 60)
        
        # 初始化应用
        if init_app():
            # 启动Flask应用
            app.run(
                host=config.HOST,
                port=config.PORT,
                debug=config.DEBUG
            )
        else:
            print("❌ 应用初始化失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()