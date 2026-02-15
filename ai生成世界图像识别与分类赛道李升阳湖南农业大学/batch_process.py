#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量脑肿瘤检测脚本
"""

import os
import json
import csv
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd

from src.inference import run, evaluate
from src.seg_yolov8 import infer_seg
from src.utils import list_images

def batch_detection(input_dir: str, output_dir: str, model_type: str = 'detection'):
    """
    批量检测处理
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出结果目录
        model_type: 模型类型 ('detection' 或 'segmentation')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_type == 'detection':
        print("🔍 开始批量检测...")
        
        # 设置检测参数
        weights_path = Path('models/det_best.pt')
        output_csv = output_path / f'detection_results_{timestamp}.csv'
        
        # 运行检测
        run(
            data_dir=input_path,
            weights_path=weights_path,
            out_csv=output_csv,
            model_name='efficientnetv2_s',
            img_size=224,
            save_cam_dir=output_path / 'cam_outputs'
        )
        
        print(f"✅ 检测完成，结果保存至: {output_csv}")
        
        # 评估结果（如果有标签的话）
        try:
            print("📊 评估检测性能...")
            metrics = evaluate(input_path, weights_path)
            
            metrics_file = output_path / f'metrics_{timestamp}.json'
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            print(f"📈 评估结果:")
            print(f"   准确率: {metrics['accuracy']:.3f}")
            print(f"   精确率: {metrics['precision']:.3f}")
            print(f"   召回率: {metrics['recall']:.3f}")
            print(f"   F1分数: {metrics['f1']:.3f}")
            
        except Exception as e:
            print(f"⚠️ 评估失败: {e}")
    
    elif model_type == 'segmentation':
        print("✂️ 开始批量分割...")
        
        # 设置分割参数
        weights_path = Path('models/seg_best.pt')
        
        # 运行分割
        pos_count, neg_count = infer_seg(
            model_path=weights_path,
            data_dir=input_path,
            out_dir=output_path / 'seg_outputs',
            imgsz=1024,
            conf=0.35,
            iou=0.5
        )
        
        print(f"✅ 分割完成")
        print(f"   检测到肿瘤: {pos_count} 张")
        print(f"   未检测到肿瘤: {neg_count} 张")

def generate_report(input_dir: str, output_dir: str):
    """
    生成检测报告
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    
    # 查找最新的结果文件
    csv_files = list(output_path.glob('detection_results_*.csv'))
    if not csv_files:
        print("❌ 未找到检测结果文件")
        return
    
    latest_csv = max(csv_files, key=os.path.getctime)
    
    # 读取结果
    df = pd.read_csv(latest_csv)
    
    # 生成统计报告
    total_images = len(df)
    tumor_detected = len(df[df['pred'] == 1])
    no_tumor = len(df[df['pred'] == 0])
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'total_images': total_images,
        'tumor_detected': tumor_detected,
        'no_tumor': no_tumor,
        'detection_rate': tumor_detected / total_images if total_images > 0 else 0
    }
    
    # 保存报告
    report_file = output_path / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("📋 检测报告:")
    print(f"   总图像数: {total_images}")
    print(f"   检测到肿瘤: {tumor_detected} ({tumor_detected/total_images*100:.1f}%)")
    print(f"   无肿瘤: {no_tumor} ({no_tumor/total_images*100:.1f}%)")
    print(f"   报告保存至: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='批量脑肿瘤检测工具')
    parser.add_argument('--input', '-i', required=True, help='输入图像目录')
    parser.add_argument('--output', '-o', required=True, help='输出结果目录')
    parser.add_argument('--model', '-m', choices=['detection', 'segmentation', 'both'], 
                       default='detection', help='使用的模型类型')
    parser.add_argument('--report', '-r', action='store_true', help='生成检测报告')
    
    args = parser.parse_args()
    
    print("🧠 批量脑肿瘤检测系统")
    print("=" * 50)
    
    if args.model in ['detection', 'both']:
        batch_detection(args.input, args.output, 'detection')
    
    if args.model in ['segmentation', 'both']:
        batch_detection(args.input, args.output, 'segmentation')
    
    if args.report:
        generate_report(args.input, args.output)
    
    print("=" * 50)
    print("✅ 批量处理完成!")

if __name__ == '__main__':
    main()