#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑肿瘤检测系统启动脚本
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'torch',
        'torchvision',
        'flask',
        'flask_cors',
        'pillow',
        'numpy',
        'ultralytics',
        'timm',
        'opencv-python',
    ]

    import_name_overrides = {
        'pillow': 'PIL',
        'opencv-python': 'cv2',
    }
    
    missing_packages = []
    
    for package in required_packages:
        try:
            import_name = import_name_overrides.get(package, package.replace('-', '_'))
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True

def check_model_files():
    """检查模型文件是否存在"""
    model_files = {
        'det_best.pt': 'models/det_best.pt',
        'seg_best.pt': 'models/seg_best.pt'
    }
    
    missing_files = []
    
    for model_name, model_path in model_files.items():
        if not Path(model_path).exists():
            missing_files.append(model_path)
    
    if missing_files:
        print("❌ 缺少以下模型文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n请确保模型文件存在于models目录中")
        return False
    
    print("✅ 模型文件检查通过")
    return True

def create_directories():
    """创建必要的目录"""
    directories = ['uploads', 'results', 'templates', 'static']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ 目录结构检查完成")

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ 需要Python 3.7或更高版本")
        print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True

def setup_environment():
    """设置环境变量"""
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # 确保模型路径在Python路径中
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def main():
    print("🧠 脑肿瘤检测系统启动检查")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查模型文件
    if not check_model_files():
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 设置环境
    setup_environment()
    
    print("=" * 50)
    print("🚀 启动脑肿瘤检测系统...")
    
    try:
        # 导入并启动Flask应用
        from app import app
        print("📡 正在启动Web服务器...")
        print("🌐 访问地址: http://localhost:5000")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
