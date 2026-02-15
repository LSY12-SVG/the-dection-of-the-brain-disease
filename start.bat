@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 🧠 脑肿瘤智能检测系统
echo ========================================
echo.

echo 📦 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python环境正常
echo.

echo 🧪 检查依赖...
python -c "import flask, torch, timm, PIL" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  检测到缺少依赖，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)

echo ✅ 依赖检查完成
echo.

echo 🧠 检查模型文件...
if not exist "weights_eff\best.pt" (
    echo ❌ 错误：模型文件不存在 weights_eff\best.pt
    echo 请确保模型文件在正确位置
    pause
    exit /b 1
)

echo ✅ 模型文件存在
echo.

echo 🚀 启动Web服务器...
echo.
echo 📱 请在浏览器中访问：http://localhost:5000
echo 🛑 按 Ctrl+C 停止服务
echo ========================================
echo.

python app.py

echo.
echo 👋 服务已停止
pause