from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DET_WEIGHTS = BASE_DIR / 'runs_yolo/tumor_yolov8m_30M_gridB_retrain5/weights/best.pt'
SEG_WEIGHTS = BASE_DIR / 'runs_seg_train/yolov8n_seg_auto/weights/best.pt'

def count_params(weights_path, model_name):
    if not weights_path.exists():
        print(f"Error: {model_name} weights not found at {weights_path}")
        return 0
    
    model = YOLO(str(weights_path))
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"{model_name}: {n_params / 1e6:.2f} M parameters")
    return n_params

def main():
    print("Checking model parameters...")
    det_params = count_params(DET_WEIGHTS, "Detection Model (YOLOv8m)")
    seg_params = count_params(SEG_WEIGHTS, "Segmentation Model (YOLOv8n-seg)")
    
    total_params = det_params + seg_params
    print(f"-" * 30)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    
    if total_params <= 30 * 1e6:
        print("Result: YES, total parameters are within 30 M.")
    else:
        print("Result: NO, total parameters exceed 30 M.")

if __name__ == "__main__":
    main()
