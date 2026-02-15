@echo off
chcp 65001 >nul
echo.
echo 🧠 脑肿瘤检测系统启动脚本
echo ========================================

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到Python，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 检查pip是否可用
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip不可用，请检查Python安装
    pause
    exit /b 1
)

:: 检查虚拟环境
if not exist "venv" (
    echo 📦 创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ 创建虚拟环境失败
        pause
        exit /b 1
    )
)

:: 激活虚拟环境
echo 🔄 激活虚拟环境...
call venv\Scripts\activate.bat

:: 检查依赖
echo 🔍 检查依赖包...
pip show torch >nul 2>&1
if %errorlevel% neq 0 (
    echo 📥 安装依赖包...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)

:: 检查模型文件
if not exist "models\det_best.pt" (
    echo ⚠️  警告: 未找到检测模型 models\det_best.pt
)
if not exist "models\seg_best.pt" (
    echo ⚠️  警告: 未找到分割模型 models\seg_best.pt
)

:: 创建必要目录
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "templates" mkdir templates
if not exist "static" mkdir static

:: 启动系统
echo.
echo 🚀 启动脑肿瘤检测系统...
echo 🌐 服务地址: http://localhost:5000
echo ⏹️  按 Ctrl+C 停止服务
echo ========================================
echo.

python run.py

:: 如果程序正常退出，保持窗口打开
echo.
echo 服务已停止，按任意键退出...
pause >nul