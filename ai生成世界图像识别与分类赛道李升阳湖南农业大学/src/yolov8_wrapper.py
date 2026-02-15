import torch
from pathlib import Path
from PIL import Image
import numpy as np
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLOv8Segmentation:
    def __init__(self, model_path: Path):
        if YOLO is None:
            raise ImportError("ultralytics package is required for segmentation")
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(model_path))
        self.model_path = model_path
    
    def predict(self, image_path: str, save: bool = True, save_dir: str = None):
        """
        预测单个图像的分割结果
        
        Args:
            image_path: 图像路径
            save: 是否保存结果
            save_dir: 保存目录
        
        Returns:
            包含分割结果的列表
        """
        try:
            img = Image.open(image_path).convert('RGB')
            w, h = img.size
            
            # 使用YOLO进行预测
            results = self.model.predict(
                source=image_path,
                imgsz=1024,
                conf=0.35,
                iou=0.5,
                device=self.device,
                save=save,
                verbose=False,
                max_det=3
            )
            
            if save and save_dir:
                # 如果需要保存，使用项目提供的目录结构
                pass
            
            return results
            
        except Exception as e:
            print(f"Segmentation prediction error: {e}")
            return None
    
    def get_mask_and_bbox(self, image_path: str):
        """
        获取分割掩码和边界框
        
        Args:
            image_path: 图像路径
        
        Returns:
            tuple: (mask, bbox) 或 (None, None)
        """
        results = self.predict(image_path, save=False)
        
        if not results or len(results) == 0:
            return None, None
        
        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None or result.masks.data is None:
            return None, None
        
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        
        masks = result.masks.data.cpu().numpy()
        
        # 选择最大的掩码
        largest_mask = None
        largest_area = 0
        
        for mask_np in masks:
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
            mask_arr = np.array(mask_img)
            area = np.sum(mask_arr > 0)
            
            if area > largest_area:
                largest_area = area
                largest_mask = mask_arr
        
        if largest_mask is None:
            return None, None
        
        # 计算边界框
        from src.cam import mask_to_bbox
        bbox = mask_to_bbox(largest_mask)
        
        return largest_mask, bbox

def predict_yolov8_seg(model_path: Path):
    """
    创建分割模型实例的工厂函数
    
    Args:
        model_path: 模型路径
    
    Returns:
        YOLOv8Segmentation实例
    """
    return YOLOv8Segmentation(model_path)