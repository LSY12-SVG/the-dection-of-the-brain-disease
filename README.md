# 脑肿瘤智能检测系统

基于深度学习的脑肿瘤检测Web应用，使用EfficientNetV2-S模型实现高精度肿瘤分类。

## 🚀 快速开始

### 1. 环境准备

确保已安装Python 3.8+，然后安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 启动应用

#### 方法一：使用启动脚本（推荐）
```bash
python run.py
```

#### 方法二：直接运行主应用
```bash
python app.py
```

### 3. 访问应用

启动后，在浏览器中访问：
```
http://localhost:5000
```

## 📊 模型信息

- **模型架构**：EfficientNetV2-S
- **输入尺寸**：224x224
- **准确率**：75.49%
- **F1分数**：79.87%
- **判定阈值**：0.61
- **模型文件**：`weights_eff/best.pt`

## 🎯 功能特性

### 核心功能
- 📁 **智能图像上传**：支持拖拽、点击上传
- 🔍 **实时检测**：基于深度学习的快速分析
- 📊 **结果可视化**：直观的概率展示和置信度显示
- 💡 **医疗建议**：根据检测结果提供个性化建议
- 📱 **响应式设计**：适配各种设备屏幕

### 支持的图像格式
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

### 技术规格
- 最大文件大小：16MB
- 处理速度：< 2秒/张
- 并发支持：多用户同时使用
- 错误处理：完善的异常捕获机制

## 🛠 API接口

### 健康检查
```
GET /api/health
```

返回系统状态和模型加载情况。

### 图像检测
```
POST /api/predict
Content-Type: multipart/form-data
```

**参数：**
- `file`：图像文件

**返回示例：**
```json
{
  "success": true,
  "prediction": "肿瘤阴性",
  "confidence": 0.8542,
  "tumor_probability": 0.1245,
  "threshold": 0.61,
  "predicted_class": 0,
  "model_info": {
    "model_name": "efficientnetv2_s",
    "image_size": 224,
    "accuracy": "75.49%",
    "f1_score": "79.87%"
  },
  "recommendation": "检测结果为正常范围，但仍建议定期进行健康检查。"
}
```

## 📁 项目结构

```
脑肿瘤检测系统/
├── app.py                 # 主应用文件
├── run.py                 # 启动脚本
├── requirements.txt        # 依赖配置
├── templates/              # 前端模板
│   └── index.html        # 主界面
├── src/                  # 核心算法模块
├── weights_eff/          # 模型权重
│   ├── best.pt           # 最佳模型权重
│   └── metrics.json      # 模型性能指标
└── uploads/              # 临时上传目录
```

## ⚙️ 配置说明

在 `app.py` 中的 `Config` 类可以修改以下配置：

```python
class Config:
    MODEL_PATH = "weights_eff/best.pt"    # 模型文件路径
    MODEL_NAME = "efficientnetv2_s"       # 模型架构名称
    IMG_SIZE = 224                        # 输入图像尺寸
    THRESHOLD = 0.61                      # 判定阈值
    HOST = "0.0.0.0"                      # 服务器地址
    PORT = 5000                           # 服务器端口
```

## 🔧 故障排除

### 常见问题

1. **模型加载失败**
   - 检查 `weights_eff/best.pt` 文件是否存在
   - 确认PyTorch版本兼容性

2. **依赖安装失败**
   ```bash
   # 清理pip缓存
   pip cache purge
   # 使用国内镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

3. **端口被占用**
   ```bash
   # 查看端口占用
   netstat -ano | findstr :5000
   # 修改配置中的PORT值
   ```

4. **CUDA内存不足**
   - 系统会自动切换到CPU模式
   - 如需GPU加速，确保CUDA版本匹配

## 📈 性能优化

### 建议配置
- **CPU**：4核心以上
- **内存**：8GB以上
- **GPU**：NVIDIA GPU（可选，用于加速）

### 优化选项
- 批量处理：修改API支持批量检测
- 模型量化：减小模型大小，提高推理速度
- 缓存机制：对重复图像进行缓存

## 🔒 安全说明

- 所有上传文件仅在处理期间临时保存
- 自动清理机制防止磁盘空间占用
- 文件类型和大小验证
- 输入验证防止恶意攻击

## 📄 许可证

本项目仅用于研究和教育目的，不作为医疗诊断工具。

---

⚠️ **重要声明**：本系统仅为辅助检测工具，不能替代专业医疗诊断。任何检测结果都应由专业医疗人员进行复核。