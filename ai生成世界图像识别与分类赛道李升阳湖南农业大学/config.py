# -*- coding: utf-8 -*-
"""
脑肿瘤检测系统配置文件
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "测试数据集"
OUTPUT_DIR = BASE_DIR / "输出结果"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# 模型配置
DETECTION_MODEL_PATH = MODELS_DIR / "det_best.pt"
SEGMENTATION_MODEL_PATH = MODELS_DIR / "seg_best.pt"

# 图像处理配置
IMAGE_SIZE = 224
MAX_IMAGE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# 检测配置
DETECTION_CONFIDENCE_THRESHOLD = 0.5
DETECTION_HIGH_CONF_THRESHOLD = 0.8
MASK_PERCENTILE = 80.0
ROI_MARGIN = 0.1
BORDER_RATIO = 0.02

# 分割配置
SEGMENTATION_CONF_THRESHOLD = 0.35
SEGMENTATION_IOU_THRESHOLD = 0.5
SEGMENTATION_IMG_SIZE = 1024
SEGMENTATION_MAX_DET = 3
SEGMENTATION_MIN_AREA = 0.0005
SEGMENTATION_MAX_AREA = 0.6

# Web服务配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# GPU配置
DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 数据库配置（可选，用于存储历史记录）
DATABASE_URL = "sqlite:///brain_tumor_detection.db"

# 缓存配置
CACHE_TYPE = "simple"
CACHE_DEFAULT_TIMEOUT = 300

# 安全配置
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
UPLOAD_FOLDER = str(UPLOAD_DIR)
MAX_CONTENT_LENGTH = MAX_IMAGE_SIZE

# 文件清理配置
CLEANUP_INTERVAL = 3600  # 1小时
MAX_FILE_AGE = 86400     # 24小时

def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        UPLOAD_DIR,
        RESULTS_DIR,
        TEMPLATES_DIR,
        STATIC_DIR,
        OUTPUT_DIR,
        MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories

# 创建目录
ensure_directories()