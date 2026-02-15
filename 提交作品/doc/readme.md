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

## 3. Directory Structure

- **`src/`**: Source code directory.
    - **`main.py`**: The primary entry point. Orchestrates model loading, inference, result fusion, and CSV generation.
    - **`cam.py`**: Utilities for visualization and mask processing.
    - **`seg_yolov8.py`**, **`inference.py`**, **`utils.py`**: Helper modules.
- **`models/`**: Contains the trained model weights.
    - **`det_best.pt`**: YOLOv8m detection weights.
    - **`seg_best.pt`**: YOLOv8n-seg segmentation weights.
- **`doc/`**: Documentation files.

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
- `--data`: Path to the directory containing test images. Defaults to `./测试数据集`.
- `--out`: Path where the output CSV file will be saved. Defaults to `./cla_pre.csv`.
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
