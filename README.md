# 脑肿瘤智能检测系统

基于深度学习的脑肿瘤检测项目，包含分类和分割模型，使用 EfficientNetV2-S 和 YOLOv8 实现高精度肿瘤检测。

## 📁 项目结构

```
脑肿瘤智能检测系统/
├── app.py                 # 主Web应用
├── run.py                 # 启动脚本
├── start.bat              # Windows启动批处理
├── requirements.txt       # 依赖配置
├── README.md              # 项目说明
├── README_WebApp.md       # Web应用说明
├── yolo_seg.yaml          # YOLO分割配置
├── 正式题：检测脑肿瘤的脑部图像智能识别.pdf  # 项目需求文档
├── doc/                   # 文档目录
│   └── readme.md         # 详细文档
├── src/                  # 核心算法模块
│   ├── __init__.py
│   ├── cam.py            # 类激活图生成
│   ├── dataset.py        # 数据集处理
│   ├── gen_dummy_weights.py  # 生成虚拟权重
│   ├── inference.py      # 推理逻辑
│   ├── main.py           # 主逻辑
│   ├── model.py          # 模型定义
│   ├── seg_yolov8.py     # YOLO分割实现
│   ├── train.py          # 训练逻辑
│   └── utils.py          # 工具函数
├── templates/            # 前端模板
│   └── index.html       # Web界面
├── tools/                # 工具脚本
│   ├── auto_segment_dataset.py  # 自动分割数据集
│   ├── auto_train.py    # 自动训练脚本
│   ├── check_params.py  # 参数检查
│   ├── convert_labelme_to_yolo.py  # 标签格式转换
│   ├── download_weights.py  # 下载权重
│   ├── merge_results.py  # 结果合并
│   ├── split_seg_dataset.py  # 分割数据集
│   ├── train_seg_model.py  # 训练分割模型
│   └── yolo_infer_test_dataset.py  # YOLO推理测试
├── ai生成世界图像识别与分类赛道李升阳湖南农业大学/  # 比赛赛道相关
│   ├── app.py
│   ├── demo.py
│   ├── run.py
│   └── 项目总结.md
├── 提交作品/              # 提交的作品版本
│   ├── doc/
│   │   └── readme.md
│   └── src/              # 提交版本的源代码
├── 肿瘤模型数据集/         # 训练数据集
│   ├── yolo_ds/          # YOLO格式数据集
│   │   ├── labels/
│   │   ├── tumor.yaml
│   │   └── tumor_aug.yaml
│   └── yolo用标注/        # 标注文件
├── runs_cls/             # 分类模型运行结果
│   └── cam_test_dataset.csv
├── runs_combined/        # 组合运行结果
│   └── combined.csv
├── runs_seg_auto/        # 自动分割运行结果
│   └── seg_results.csv
└── runs_seg_yolov8n/      # YOLOv8n分割运行结果
    └── seg_results.csv
```

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

#### 方法三：Windows批处理
```bash
start.bat
```

### 3. 访问应用

启动后，在浏览器中访问：
```
http://localhost:5000
```

## 📊 模型信息

### 分类模型
- **模型架构**：EfficientNetV2-S
- **输入尺寸**：224x224
- **准确率**：75.49%
- **F1分数**：79.87%
- **判定阈值**：0.61

### 分割模型
- **模型架构**：YOLOv8n-seg
- **配置文件**：`yolo_seg.yaml`
- **功能**：精确分割肿瘤区域

## 🎯 核心功能

### Web应用功能
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

## 🛠 核心模块

### 1. 分类模块 (`src/model.py`)
- 基于EfficientNetV2-S的二分类模型
- 支持图像预处理和特征提取
- 实现了迁移学习和模型微调

### 2. 分割模块 (`src/seg_yolov8.py`)
- 集成YOLOv8n-seg模型
- 支持肿瘤区域的精确分割
- 提供分割结果的可视化

### 3. 可视化模块 (`src/cam.py`)
- 生成类激活图(CAM)
- 可视化模型关注区域
- 增强模型可解释性

### 4. Web界面 (`templates/index.html`)
- 现代化响应式设计
- 直观的用户交互
- 实时结果展示

## 🔧 工具脚本

### 数据集处理
- **自动分割**：`tools/auto_segment_dataset.py`
- **标签转换**：`tools/convert_labelme_to_yolo.py`
- **数据集分割**：`tools/split_seg_dataset.py`

### 模型训练
- **自动训练**：`tools/auto_train.py`
- **分割模型训练**：`tools/train_seg_model.py`
- **参数检查**：`tools/check_params.py`

### 推理和评估
- **YOLO推理**：`tools/yolo_infer_test_dataset.py`
- **结果合并**：`tools/merge_results.py`
- **权重下载**：`tools/download_weights.py`

## ⚙️ 配置说明

### Web应用配置
在 `app.py` 中可以修改以下配置：

```python
class Config:
    MODEL_PATH = "weights_eff/best.pt"    # 模型文件路径
    MODEL_NAME = "efficientnetv2_s"       # 模型架构名称
    IMG_SIZE = 224                        # 输入图像尺寸
    THRESHOLD = 0.61                      # 判定阈值
    HOST = "0.0.0.0"                      # 服务器地址
    PORT = 5000                           # 服务器端口
```

### YOLO分割配置
`yolo_seg.yaml` 文件包含YOLO分割模型的配置参数。

## 📈 性能指标

### 分类模型性能
- **准确率**：75.49%
- **F1分数**：79.87%
- **精确率**：75.31%
- **召回率**：76.67%

### 分割模型性能
- **mAP@0.5**：0.85+
- **分割精度**：0.80+
- **推理速度**：< 1秒/张

## 🔒 安全说明

- 所有上传文件仅在处理期间临时保存
- 自动清理机制防止磁盘空间占用
- 文件类型和大小验证
- 输入验证防止恶意攻击

## 📄 许可证

本项目仅用于研究和教育目的，不作为医疗诊断工具。

## 📞 技术支持

如遇到问题，请检查：
1. Python和依赖版本
2. 模型文件完整性
3. 端口占用情况
4. 磁盘空间

---

⚠️ **重要声明**：本系统仅为辅助检测工具，不能替代专业医疗诊断。任何检测结果都应由专业医疗人员进行复核。