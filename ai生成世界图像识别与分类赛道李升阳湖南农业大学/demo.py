#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑肿瘤检测系统演示脚本
"""

import os
import sys
from pathlib import Path
from flask import Flask
import requests
import base64
from PIL import Image
import io

def test_image_upload(image_path):
    """测试图像上传和检测"""
    url = 'http://localhost:5000/api/detect'
    
    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 检测成功!")
            print(f"   文件名: {result.get('filename', 'N/A')}")
            print(f"   预测结果: {result.get('prediction', 'N/A')}")
            print(f"   置信度: {result.get('confidence', 0):.3f}")
            print(f"   概率: {result.get('probability', 0):.3f}")
            
            # 检查是否有可视化结果
            if result.get('bbox_image'):
                print("   ✅ 生成了边界框图像")
            if result.get('cam_image'):
                print("   ✅ 生成了热力图")
            if result.get('mask_image'):
                print("   ✅ 生成了掩码图")
            
            return True
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("   请确保服务器正在运行: python run.py")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_model_info():
    """测试模型信息接口"""
    url = 'http://localhost:5000/api/model_info'
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 模型信息获取成功!")
            print(f"   设备: {result.get('device', 'N/A')}")
            print(f"   检测模型已加载: {'是' if result.get('detection_model_loaded') else '否'}")
            print(f"   分割模型已加载: {'是' if result.get('segmentation_model_loaded') else '否'}")
            return True
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主演示函数"""
    print("🧠 脑肿瘤检测系统演示")
    print("=" * 50)
    
    # 检查测试数据
    test_dir = Path('测试数据集')
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')) + list(test_dir.glob('*.png'))
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("   请确保测试数据存在")
        return False
    
    print(f"📁 找到 {len(test_images)} 张测试图像")
    
    # 测试模型信息
    print("\n🤖 测试模型信息接口...")
    if not test_model_info():
        return False
    
    # 测试图像检测
    print(f"\n🔍 测试图像检测...")
    
    # 测试前3张图像
    for i, img_path in enumerate(test_images[:3]):
        print(f"\n测试图像 {i+1}: {img_path.name}")
        if test_image_upload(str(img_path)):
            print("   ✅ 检测成功")
        else:
            print("   ❌ 检测失败")
    
    print("\n" + "=" * 50)
    print("🎉 演示完成!")
    print("\n💡 提示:")
    print("   - 打开浏览器访问 http://localhost:5000 使用Web界面")
    print("   - 支持拖拽上传图像文件")
    print("   - 可以查看热力图、掩码图和边界框")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)