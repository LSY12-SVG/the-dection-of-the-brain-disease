# Tumor Detection and Segmentation Algorithm

## 1. Project Overview
This project implements an automated pipeline for tumor detection and segmentation in medical images. The system utilizes state-of-the-art YOLOv8 models for both object detection and instance segmentation, ensuring high sensitivity and precision.

## 2. Algorithm Description

### 2.1 Object Detection (YOLOv8m)
- **Model Architecture**: YOLOv8m (Medium)
- **Training Strategy**: Trained on the provided tumor dataset with optimized hyperparameters.
- **Inference**: Performs bounding box regression to identify potential tumor regions. A confidence threshold of 0.25 is used to balance precision and recall.

### 2.2 Instance Segmentation (YOLOv8n-seg)
- **Dataset Construction**: Due to the lack of pixel-level annotations in the original dataset, a semi-automated annotation pipeline was developed:
    1.  **Auto-Segmentation**: Used the **GrabCut** algorithm initialized with ground-truth bounding boxes to extract foreground masks.
    2.  **Dataset Generation**: Converted masks to YOLO segmentation format (Polygons).
- **Model Architecture**: YOLOv8n-seg (Nano) for efficient segmentation.
- **Inference**: Generates pixel-wise masks for tumor regions. A confidence threshold of 0.35 is applied.

### 2.3 Result Fusion
To maximize detection performance, a **Logical OR Fusion** strategy is employed:
- If **either** the detection model or the segmentation model identifies a tumor in an image, the image is classified as **Positive (Tumor)**.
- This approach leverages the complementary strengths of both models: detection handles general localization, while segmentation captures fine-grained features.

## 3. Code Structure (./src)

- **`main.py`**: The primary entry point for the submission. Orchestrates the loading of models, execution of inference, result fusion, and CSV generation.
- **`inference.py`**: Contains helper functions and legacy logic for EfficientNet-based classification and Grad-CAM visualization.
- **`seg_yolov8.py`**: Dedicated script for YOLOv8 segmentation inference with advanced mask validation logic (ROI checks, boundary checks).
- **`cam.py`**: Utilities for Class Activation Mapping (Grad-CAM) and image overlay visualization.
- **`utils.py`**: General utility functions for file handling and data processing.

## 4. Execution

To run the inference pipeline on the test dataset. Please replace `"path/to/images"` with the actual path to your image directory.

```bash
python src/main.py --data "path/to/images" --out ./cla_pre.csv
```

### Example:
```bash
python src/main.py --data "正式题数据集/no" --out ./cla_pre.csv
```

### Parameters:
- `--data`: Path to the directory containing test images.
- `--out`: Path where the output CSV file will be saved.
- `--det-weights`: (Optional) Path to detection model weights.
- `--seg-weights`: (Optional) Path to segmentation model weights.

## 5. Output Format
The output file (`cla_pre.csv`) follows the format:
```csv
filename,label
image_001.jpg,1
image_002.jpg,0
...
```
- **label**: `1` indicates Tumor, `0` indicates Normal.
