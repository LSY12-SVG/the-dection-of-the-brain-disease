#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑肿瘤检测系统测试脚本
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试模块导入...")
    
    try:
        import flask
        import flask_cors
        from src.inference import run, base_transform, tta_variants
        from src.model import create_model
        from src.utils import list_images, infer_label_from_path
        from src.cam import GradCAM, overlay_cam, cam_to_mask, mask_to_bbox
        from src.yolov8_wrapper import predict_yolov8_seg
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_models():
    """测试模型加载"""
    print("\n🤖 测试模型加载...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")
    
    # 导入必要的模块
    from src.model import create_model
    
    # 测试检测模型
    try:
        det_weights_path = Path('models/det_best.pt')
        if not det_weights_path.exists():
            print("❌ 检测模型文件不存在")
            return False
        
        detection_model = create_model(model_name='efficientnetv2_s', pretrained=False)
        try:
            state = torch.load(det_weights_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            state = torch.load(det_weights_path, map_location=device)
        detection_model.load_state_dict(state)
        detection_model.to(device)
        detection_model.eval()
        print("✅ 检测模型加载成功")
        
    except Exception as e:
        print(f"❌ 检测模型加载失败: {e}")
        return False
    
    # 测试分割模型
    try:
        seg_weights_path = Path('models/seg_best.pt')
        if not seg_weights_path.exists():
            print("❌ 分割模型文件不存在")
            return False
        
        from src.yolov8_wrapper import predict_yolov8_seg
        seg_model = predict_yolov8_seg(seg_weights_path)
        print("✅ 分割模型加载成功")
        
    except Exception as e:
        print(f"❌ 分割模型加载失败: {e}")
        return False
    
    return True

def test_image_processing():
    """测试图像处理功能"""
    print("\n🖼️  测试图像处理...")
    
    # 查找测试图像
    test_data_dir = Path('测试数据集')
    if not test_data_dir.exists():
        print("❌ 测试数据目录不存在")
        return False
    
    images = list(test_data_dir.glob('*.jpg')) + list(test_data_dir.glob('*.jpeg')) + list(test_data_dir.glob('*.png'))
    
    if not images:
        print("❌ 未找到测试图像")
        return False
    
    test_image = images[0]
    print(f"   使用测试图像: {test_image.name}")
    
    try:
        # 导入必要的模块
        from src.inference import base_transform, tta_variants
        
        # 测试图像加载
        img = Image.open(test_image).convert('RGB')
        print(f"   图像尺寸: {img.size}")
        
        # 测试变换
        transform = base_transform(img_size=224)
        tensor = transform(img)
        print(f"   变换后张量形状: {tensor.shape}")
        
        # 测试TTA变体
        variants = tta_variants(img)
        print(f"   生成了 {len(variants)} 个TTA变体")
        
        print("✅ 图像处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return False

def test_detection_inference():
    """测试检测推理"""
    print("\n🔬 测试检测推理...")
    
    test_data_dir = Path('测试数据集')
    images = list(test_data_dir.glob('*.jpg')) + list(test_data_dir.glob('*.jpeg')) + list(test_data_dir.glob('*.png'))
    
    if not images:
        print("❌ 未找到测试图像")
        return False
    
    test_image = images[0]
    
    try:
        # 导入必要的模块
        from src.model import create_model
        from src.inference import base_transform, tta_variants
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        det_weights_path = Path('models/det_best.pt')
        detection_model = create_model(model_name='efficientnetv2_s', pretrained=False)
        try:
            state = torch.load(det_weights_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            state = torch.load(det_weights_path, map_location=device)
        detection_model.load_state_dict(state)
        detection_model.to(device)
        detection_model.eval()
        
        # 处理图像
        img = Image.open(test_image).convert('RGB')
        transform = base_transform(img_size=224)
        variants = tta_variants(img)
        batch = torch.stack([transform(v) for v in variants]).to(device)
        
        # 推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = detection_model(batch).squeeze(1)
                probs = torch.sigmoid(logits)
        
        prob = float(probs.mean().cpu().item())
        pred = 1 if prob >= 0.5 else 0
        
        print(f"   检测概率: {prob:.3f}")
        print(f"   预测结果: {'检测到肿瘤' if pred == 1 else '未检测到肿瘤'}")
        
        print("✅ 检测推理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 检测推理失败: {e}")
        return False

def test_cam_generation():
    """测试CAM生成"""
    print("\n🎯 测试CAM生成...")
    
    test_data_dir = Path('测试数据集')
    images = list(test_data_dir.glob('*.jpg')) + list(test_data_dir.glob('*.jpeg')) + list(test_data_dir.glob('*.png'))
    
    if not images:
        print("❌ 未找到测试图像")
        return False
    
    test_image = images[0]
    
    try:
        # 导入必要的模块
        from src.model import create_model
        from src.inference import base_transform
        from src.cam import GradCAM, overlay_cam, cam_to_mask
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        det_weights_path = Path('models/det_best.pt')
        detection_model = create_model(model_name='efficientnetv2_s', pretrained=False)
        try:
            state = torch.load(det_weights_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            state = torch.load(det_weights_path, map_location=device)
        detection_model.load_state_dict(state)
        detection_model.to(device)
        detection_model.eval()
        
        # 处理图像
        img = Image.open(test_image).convert('RGB')
        transform = base_transform(img_size=224)
        x = transform(img).unsqueeze(0).to(device)
        
        # 生成CAM
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            cam = GradCAM(detection_model).generate(x)
        
        # 生成可视化
        cam_vis = overlay_cam(img, cam)
        mask_small = cam_to_mask(cam, percentile=80.0)
        
        print(f"   CAM形状: {cam.shape}")
        print(f"   掩码形状: {mask_small.shape}")
        print(f"   可视化图像尺寸: {cam_vis.size}")
        
        print("✅ CAM生成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ CAM生成失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🧠 脑肿瘤检测系统测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("模型加载", test_models),
        ("图像处理", test_image_processing),
        ("检测推理", test_detection_inference),
        ("CAM生成", test_cam_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️  {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关配置。")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)