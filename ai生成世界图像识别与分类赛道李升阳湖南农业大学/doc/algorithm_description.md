# 算法原理与代码说明文档 (Algorithm Description)


使用说明：按照比赛提交作品的要求，请老师将检测用的图片放入测试数据集中，之后运行main文件，./cla_pre.csv和检测结果，分割结果的可视化图像都会保存到输出结果文件加重查看，0表示没有检测到肿瘤，1表示检测到肿瘤
检测模型 (YOLOv8m) : 约 25.86 M
分割模型 (YOLOv8n-seg) : 约 3.26 M
总计 : 29.12 M

- Python : 3.8+ (推荐 3.10 或 3.12)
- PyTorch : 深度学习框架 (本项目代码兼容 2.x 版本)
- Ultralytics : YOLOv8 官方库，用于加载和运行检测/分割模型。
- OpenCV (opencv-python) : 图像处理。
- Pillow (PIL) : 图像读取和保存。
- NumPy & Pandas : 数据处理和 CSV 生成。
- Timm : 用于 model.py 中的分类模型定义。


## 1. 项目概述 (Project Overview)
本项目旨在解决医疗图像中的肿瘤自动检测与分割问题。针对医疗影像数据标注困难、病灶形态多样的挑战，本项目采用了基于 **YOLOv8** 的多任务深度学习框架，结合了 **目标检测 (Object Detection)** 和 **实例分割 (Instance Segmentation)** 技术，以实现高灵敏度的病灶筛查。

## 2. 核心算法设计 (Core Algorithm Design)

本系统采用双流（Dual-Stream）并行推理架构，最终通过逻辑融合模块输出判读结果。

### 2.1 整体架构流程
1.  **图像输入**: 支持多种格式（JPG, PNG, TIFF等）的医疗影像输入。
2.  **并行推理**:
    *   **流 A (检测流)**: 使用 YOLOv8m 检测模型，专注于定位病灶的大致区域（Bounding Box）。
    *   **流 B (分割流)**: 使用 YOLOv8n-seg 分割模型，专注于提取病灶的精细轮廓（Pixel-wise Mask）。
3.  **结果融合**: 采用逻辑“或”（Logical OR）策略融合两路结果，最大化检出率（Recall）。
4.  **可视化与输出**: 生成带有标注的预测图像及 CSV 格式的分类报告。

### 2.2 目标检测模块 (Object Detection Module)
*   **模型选择**: YOLOv8m (Medium version)
*   **选择理由**: YOLOv8m 在精度（mAP）和推理速度之间取得了良好的平衡，适合在桌面端进行快速筛查。
*   **算法原理**:
    *   **Backbone**: 采用 CSPDarknet53 结构进行特征提取。
    *   **Neck**: 使用 PANet (Path Aggregation Network) 进行多尺度特征融合，增强对小目标的检测能力。
    *   **Head**: Decoupled Head 结构，将分类和回归任务解耦，提高收敛速度。
    *   **Loss Function**: 采用 CIoU Loss 进行边界框回归，DFL (Distribution Focal Loss) 处理类别不平衡。
*   **推理参数**: Confidence Threshold = 0.25 (保证较高的召回率)。

### 2.3 实例分割模块 (Instance Segmentation Module)
*   **模型选择**: YOLOv8n-seg (Nano version)
*   **选择理由**: 轻量级模型，计算开销极低，用于辅助验证检测结果并提供病灶形态信息。
*   **数据构建策略**:
    *   针对原始数据缺乏像素级标注的问题，本项目开发了 **半自动标注流程**。
    *   利用 **GrabCut** 算法，以人工标注的 Bounding Box 为初始化前景，自动提取病灶掩码（Mask）。
    *   将生成的掩码转换为 YOLO Segmentation 格式（Polygon coordinates）用于模型训练。
*   **推理参数**: Confidence Threshold = 0.35 (过滤低置信度噪点)。

### 2.4 结果融合策略 (Result Fusion)
为了降低漏诊率（False Negative Rate），系统采用以下判定逻辑：
*   **判定公式**: $Final\_Label = (Det\_Count > 0) \lor (Seg\_Count > 0)$
*   **逻辑解释**: 只要检测模型或分割模型中任意一个识别出病灶，该图像即被标记为“阳性（Tumor）”。
*   **优势**: 检测模型擅长捕捉纹理特征明显的病灶，而分割模型对形状不规则的病灶更敏感，两者互补可显著提升系统的鲁棒性。

## 3. 代码结构说明 (Code Structure)

项目代码组织结构如下，模块化设计便于维护和扩展。

```text
src/
├── main.py           # [核心入口] 负责加载模型、遍历数据、执行推理、结果融合及保存。
├── dataset.py        # [数据处理] 定义 PyTorch Dataset 类，处理图像读取、预处理和增强。
├── cam.py            # [可视化/辅助] 包含 Grad-CAM 实现及掩码处理工具（如 mask_to_bbox）。
├── model.py          # [模型定义] 定义 EfficientNet 等分类模型结构（主要用于对比实验或辅助分类）。
├── train.py          # [训练脚本] 包含 K-Fold 交叉验证的训练循环逻辑。
├── inference.py      # [推理工具] 封装单张图像的推理接口。
├── seg_yolov8.py     # [分割辅助] YOLOv8 分割结果的后处理工具。
└── utils.py          # [通用工具] 包含种子固定、路径处理、指标计算等通用函数。
```

## 4. 环境依赖 (Requirements)
项目运行依赖以下核心库（详见环境配置）：
*   `ultralytics`: YOLOv8 核心库
*   `torch`, `torchvision`: 深度学习框架
*   `numpy`, `pandas`: 数据处理
*   `Pillow`, `opencv-python`: 图像处理

## 5. 创新点 (Highlights)
1.  **数据工程**: 创新性地使用 GrabCut 算法扩充了分割数据集，解决了数据标注不足的问题。
2.  **模型集成**: 结合 Detection 和 Segmentation 两种范式，比单一分类网络具有更强的解释性（可定位、可量化面积）。
3.  **工程落地**: 完整的推理流水线（Pipeline），支持批量处理和结果可视化，具备实际应用价值。

## 6. 模型详细参数 (Detailed Model Parameters)

本系统使用的模型参数配置如下，这些参数经过验证集调优以达到最佳性能。

### 6.1 目标检测模型 (YOLOv8m)
| 参数项 (Parameter) | 值 (Value) | 说明 (Description) |
| :--- | :--- | :--- |
| **Model Architecture** | YOLOv8m | Medium 版本，平衡速度与精度 |
| **Input Resolution** | 640 x 640 | 标准输入分辨率 |
| **Confidence Threshold** | 0.25 | 推理时的置信度阈值，低于此值的框被过滤 |
| **NMS IoU Threshold** | 0.7 | 非极大值抑制的交并比阈值 (Default) |
| **Classes** | 1 | 类别：Tumor (肿瘤) |
| **Precision (FP16)** | Mixed | 自动混合精度推理 |

### 6.2 实例分割模型 (YOLOv8n-seg)
| 参数项 (Parameter) | 值 (Value) | 说明 (Description) |
| :--- | :--- | :--- |
| **Model Architecture** | YOLOv8n-seg | Nano 版本，轻量级分割模型 |
| **Input Resolution** | 640 x 640 | 标准输入分辨率 |
| **Confidence Threshold** | 0.35 | 分割任务的置信度阈值 |
| **Mask Threshold** | 0.5 | 像素分类阈值 (Default) |
| **Classes** | 1 | 类别：Tumor (肿瘤) |

### 6.3 训练超参数 (Training Hyperparameters - Reference)
虽然推理阶段固定了模型权重，但以下是训练阶段的关键参数设置（基于 YOLOv8 默认配置）：
*   **Optimizer**: SGD (lr0=0.01, momentum=0.937)
*   **Weight Decay**: 0.0005
*   **Warmup Epochs**: 3.0
*   **Augmentation**: Mosaic (1.0), Mixup (0.0), HSV-H (0.015), HSV-S (0.7), HSV-V (0.4)

### 6.4 模型参数大小
- 检测模型 (YOLOv8m) : 约 25.86 M
- 分割模型 (YOLOv8n-seg) : 约 3.26 M
- 总计 : 29.12 M