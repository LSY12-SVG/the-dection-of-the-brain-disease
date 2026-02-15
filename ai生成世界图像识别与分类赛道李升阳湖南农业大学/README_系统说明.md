# 脑肿瘤检测系统

基于深度学习的医学影像智能诊断平台，支持脑肿瘤的检测和分割。

## 🌟 功能特性

- **智能检测**: 基于EfficientNetV2的脑肿瘤检测模型
- **精确分割**: 基于YOLOv8的肿瘤区域分割
- **可视化展示**: 热力图、掩码图、边界框等多种可视化方式
- **批量处理**: 支持大规模图像批量检测
- **Web界面**: 友好的Web用户界面，支持拖拽上传
- **结果导出**: 支持检测结果导出为JSON/CSV格式

## 🛠️ 系统要求

- Python 3.7+
- CUDA支持（可选，用于GPU加速）
- 4GB+ 内存
- 存储空间: 至少2GB

## 📦 安装步骤

### 1. 环境准备

```bash
# 克隆项目
cd "c:/Users/lsy/Desktop/AI生成世界/ai生成世界图像识别与分类赛道李升阳湖南农业大学"

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# 或
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型准备

确保以下模型文件存在于 `models/` 目录中：
- `det_best.pt` - 检测模型 (49.75MB)
- `seg_best.pt` - 分割模型 (6.46MB)

### 3. 启动系统

#### 方法一：使用启动脚本（推荐）

```bash
python run.py
```

#### 方法二：直接运行Flask应用

```bash
python app.py
```

启动成功后，访问 http://localhost:5000

## 🎯 使用指南

### Web界面使用

1. **打开浏览器**，访问 http://localhost:5000
2. **上传图像**：支持点击选择或拖拽上传
3. **查看结果**：
   - 原始影像
   - 注意力热力图
   - 肿瘤掩码
   - 边界框标注
4. **导出结果**：点击导出按钮下载JSON格式结果

### 批量处理

```bash
# 批量检测
python batch_process.py --input 测试数据集 --output 批量结果 --model detection --report

# 批量分割
python batch_process.py --input 测试数据集 --output 批量结果 --model segmentation --report

# 同时运行检测和分割
python batch_process.py --input 测试数据集 --output 批量结果 --model both --report
```

## 📊 API接口

### 单张图像检测

```http
POST /api/detect
Content-Type: multipart/form-data

参数:
- file: 图像文件
```

### 批量检测

```http
POST /api/batch_detect
Content-Type: application/json

{
    "image_paths": ["path1", "path2", ...]
}
```

### 模型信息

```http
GET /api/model_info
```

## 🖼️ 支持的图像格式

- PNG
- JPG/JPEG
- BMP
- TIF/TIFF

## 🧠 模型说明

### 检测模型 (EfficientNetV2-S)
- **用途**: 判断图像中是否存在脑肿瘤
- **输入尺寸**: 224x224
- **输出**: 二分类概率 (0-1)
- **置信度阈值**: 0.5

### 分割模型 (YOLOv8)
- **用途**: 精确分割肿瘤区域
- **输入尺寸**: 1024x1024
- **置信度阈值**: 0.35
- **IoU阈值**: 0.5

## 📈 输出结果说明

### 单张图像检测结果

```json
{
    "filename": "image.jpg",
    "prediction": "tumor_detected" | "no_tumor",
    "confidence": 0.95,
    "probability": 0.82,
    "original_image": "base64编码",
    "cam_image": "base64编码",  // 热力图
    "mask_image": "base64编码",  // 掩码图
    "bbox_image": "base64编码",  // 边界框图
    "bbox": [x1, y1, x2, y2]     // 边界框坐标
}
```

### 性能指标

- **准确率**: 模型分类准确程度
- **精确率**: 预测为正例中真正为正例的比例
- **召回率**: 真正正例中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

## 🔧 配置说明

主要配置在 `config.py` 文件中：

```python
# 检测配置
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
DETECTION_HIGH_CONF_THRESHOLD = 0.8   # 高置信度阈值

# 分割配置
SEGMENTATION_CONF_THRESHOLD = 0.35    # 分割置信度阈值
SEGMENTATION_IOU_THRESHOLD = 0.5      # IoU阈值

# Web服务配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
```

## 🚨 注意事项

1. **隐私保护**: 上传的医学影像可能包含敏感信息，请确保在安全环境中使用
2. **设备要求**: 建议使用GPU加速以提高处理速度
3. **文件大小**: 单张图像最大支持16MB
4. **结果解释**: 本系统仅作为辅助诊断工具，最终诊断请以医生意见为准

## 🐛 故障排除

### 常见问题

**Q: 启动时提示"缺少依赖包"**
A: 运行 `pip install -r requirements.txt` 安装所有依赖

**Q: 提示"模型文件不存在"**
A: 确保模型文件 `det_best.pt` 和 `seg_best.pt` 存在于 `models/` 目录

**Q: 检测结果不准确**
A: 
1. 检查输入图像质量
2. 确保图像格式正确
3. 尝试调整置信度阈值

**Q: Web界面无法访问**
A:
1. 检查防火墙设置
2. 确认端口5000未被占用
3. 检查Flask服务是否正常启动

### 日志查看

系统日志会输出到控制台，包含：
- 模型加载状态
- 请求处理信息
- 错误信息

## 🤝 技术支持

如遇到问题，请提供：
1. 错误信息截图
2. 系统环境信息
3. 使用的具体操作步骤

## 📄 许可证

本项目仅供学术研究使用，请勿用于商业用途。

---

**⚠️ 免责声明**: 本系统为研究原型，不应用于实际临床诊断。所有检测结果仅供参考，请以专业医师的诊断为准。