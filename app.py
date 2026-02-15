#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑肿瘤检测Web应用后端服务
基于EfficientNetV2-S模型实现肿瘤分类
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import timm
import torchvision.transforms as transforms

# 添加src目录到系统路径
sys.path.append(str(Path(__file__).parent / 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置参数
class Config:
    # 模型配置
    MODEL_PATH = "weights_eff/best.pt"
    MODEL_NAME = "efficientnetv2_s"
    IMG_SIZE = 224
    THRESHOLD = 0.61  # 从metrics.json中获取的最佳阈值
    NUM_CLASSES = 1   # 动态设置
    IS_BINARY = True  # 动态设置
    
    # 文件上传配置
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    # 应用配置
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = False

config = Config()

# 创建上传目录
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# 全局模型变量
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = None

def load_model():
    """加载训练好的模型"""
    global model, transform
    
    try:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {config.MODEL_PATH}")
        
        logger.info(f"正在加载模型: {config.MODEL_PATH}")
        
        # 加载模型检查点
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        
        # 检查分类器层维度
        if hasattr(checkpoint, 'get') and 'classifier.weight' in checkpoint:
            classifier_shape = checkpoint['classifier.weight'].shape
            num_classes = classifier_shape[0]
        elif isinstance(checkpoint, dict) and 'classifier.weight' in checkpoint:
            classifier_shape = checkpoint['classifier.weight'].shape
            num_classes = classifier_shape[0]
        else:
            num_classes = 1  # 默认二分类sigmoid输出
        
        logger.info(f"检测到模型输出类别数: {num_classes}")
        
        # 创建模型
        if num_classes == 1:
            model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=1)
        else:
            model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=num_classes)
        
        # 加载权重
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint.state_dict())
        
        model.to(device)
        model.eval()
        
        # 定义图像变换
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 更新全局配置
        config.NUM_CLASSES = num_classes
        config.IS_BINARY = (num_classes == 1)
        
        logger.info(f"模型加载成功，设备: {device}，输出类别: {num_classes}")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return Path(filename).suffix.lower() in config.ALLOWED_EXTENSIONS

def preprocess_image(image_path: str) -> torch.Tensor:
    """预处理图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        raise

def predict_tumor(image_path: str) -> Dict:
    """进行肿瘤检测预测"""
    try:
        # 预处理图像
        input_tensor = preprocess_image(image_path).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(input_tensor)
            
            if config.IS_BINARY:
                # 二分类sigmoid输出
                tumor_prob = torch.sigmoid(outputs).item()
                confidence_score = tumor_prob if tumor_prob >= 0.5 else (1 - tumor_prob)
                predicted_class = 1 if tumor_prob >= config.THRESHOLD else 0
            else:
                # 多分类softmax输出
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                tumor_prob = probabilities[0][1].item() if probabilities.shape[1] > 1 else 1.0 - confidence_score
        
        # 根据阈值判断
        has_tumor = tumor_prob >= config.THRESHOLD
        
        # 构建结果
        result = {
            "success": True,
            "prediction": "肿瘤阳性" if has_tumor else "肿瘤阴性",
            "confidence": round(confidence_score, 4),
            "tumor_probability": round(tumor_prob, 4),
            "threshold": config.THRESHOLD,
            "predicted_class": predicted_class,
            "model_info": {
                "model_name": config.MODEL_NAME,
                "image_size": config.IMG_SIZE,
                "accuracy": "75.49%",
                "f1_score": "79.87%",
                "num_classes": getattr(config, 'NUM_CLASSES', 1)
            },
            "recommendation": get_medical_recommendation(has_tumor, tumor_prob)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "预测过程中出现错误"
        }

def get_medical_recommendation(has_tumor: bool, tumor_prob: float) -> str:
    """生成医学建议"""
    if tumor_prob >= 0.8:
        return "检测结果显示高度疑似脑肿瘤，建议立即就医进行进一步检查和诊断。"
    elif tumor_prob >= config.THRESHOLD:
        return "检测结果显示存在脑肿瘤可能性，建议及时就医进行专业诊断。"
    elif tumor_prob >= 0.4:
        return "检测结果显示存在一定异常，建议进行定期复查或寻求专业医疗建议。"
    else:
        return "检测结果为正常范围，但仍建议定期进行健康检查。"

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_path": config.MODEL_PATH
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """肿瘤检测预测接口"""
    try:
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "没有找到文件"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "没有选择文件"
            }), 400
        
        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"不支持的文件格式。支持的格式: {', '.join(config.ALLOWED_EXTENSIONS)}"
            }), 400
        
        # 检查文件大小
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置文件指针
        
        if file_size > config.MAX_FILE_SIZE:
            return jsonify({
                "success": False,
                "error": f"文件大小超过限制 ({config.MAX_FILE_SIZE // (1024*1024)}MB)"
            }), 400
        
        # 保存文件
        filename = f"upload_{int(time.time())}_{file.filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 进行预测
        result = predict_tumor(filepath)
        result['filename'] = file.filename
        
        # 清理临时文件
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"预测接口错误: {e}")
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "message": str(e)
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(config.UPLOAD_FOLDER, filename)

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({
        "success": False,
        "error": "文件大小超过限制"
    }), 413

@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return jsonify({
        "success": False,
        "error": "请求的资源不存在"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """500错误处理"""
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500

def init_app():
    """初始化应用"""
    logger.info("正在初始化脑肿瘤检测应用...")
    
    # 加载模型
    if not load_model():
        logger.error("模型加载失败，应用无法正常工作")
        return False
    
    logger.info("应用初始化完成")
    return True

if __name__ == '__main__':
    # 导入时间模块
    import time
    
    if init_app():
        logger.info(f"启动脑肿瘤检测服务 - http://{config.HOST}:{config.PORT}")
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG
        )
    else:
        logger.error("应用初始化失败，退出程序")
        sys.exit(1)