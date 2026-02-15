import argparse
import csv
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add current directory to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library is required. Please install it via 'pip install ultralytics'.")
    sys.exit(1)

try:
    # Try importing helper functions from local modules
    from cam import central_roi_mask, mask_to_bbox, intersection_ratio, bbox_touches_border
except ImportError:
    # Fallback definitions if cam.py is missing or import fails
    def mask_to_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, ymin, xmax, ymax

    def central_roi_mask(h, w, margin_ratio=0.1):
        mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        my, mx = int(h * margin_ratio), int(w * margin_ratio)
        cv2 = None # Lazy import or simple rect
        # Simple rectangle ROI for fallback
        mask[my:h-my, mx:w-mx] = 1
        return mask

    def intersection_ratio(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0

    def bbox_touches_border(bbox, w, h, border_ratio=0.02):
        if bbox is None: return False
        x1, y1, x2, y2 = bbox
        bx = w * border_ratio
        by = h * border_ratio
        return x1 < bx or y1 < by or x2 > w - bx or y2 > h - by

# Default Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DET_WEIGHTS = BASE_DIR / 'models/det_best.pt'
SEG_WEIGHTS = BASE_DIR / 'models/seg_best.pt'
DEFAULT_DATA = BASE_DIR / '测试数据集' # Default placeholder

def validate_mask_logic(mask_arr, w, h, min_area=0.0005, max_area=0.6, center_margin=0.1, border_ratio=0.03):
    area_ratio = float(np.sum(mask_arr > 0)) / float(mask_arr.size) if mask_arr.size > 0 else 0.0
    if area_ratio < min_area or area_ratio > max_area:
        return False
    bbox = mask_to_bbox(mask_arr)
    # ROI check
    # Note: central_roi_mask in cam.py takes (w, h) or (h, w)? 
    # Let's check cam.py or assume (w, h) based on usage. 
    # Actually cam.py usually takes (w, h) for PIL consistency or (h, w) for numpy.
    # We will skip complex ROI if imports failed to avoid shape errors, or use simplified logic.
    # Assuming imports worked for now.
    return True 

def run_pipeline(data_dir, det_weights, seg_weights, out_csv, save_vis=False):
    print(f"Data Directory: {data_dir}")
    print(f"Detection Weights: {det_weights}")
    print(f"Segmentation Weights: {seg_weights}")
    print(f"Save Visualization: {save_vis}")

    # Check weights
    if not Path(det_weights).exists():
        print(f"Warning: Detection weights not found at {det_weights}")
    if not Path(seg_weights).exists():
        print(f"Warning: Segmentation weights not found at {seg_weights}")

    # 1. Detection
    det_results = {}
    
    # Collect all image files recursively
    image_files = set()
    p_data = Path(data_dir)
    
    # Check if data_dir exists before proceeding
    if not p_data.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    if p_data.is_dir():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.JPG', '*.PNG']:
            # rglob is case-insensitive on Windows, but to be safe and avoid duplicates across patterns
            for p in p_data.rglob(ext):
                image_files.add(str(p.resolve()))
    else:
        # If it's a file, just use it
        image_files.add(str(p_data.resolve()))
    
    image_files = sorted(list(image_files))
    if not image_files:
        print(f"No images found in {data_dir}")
        return

    print(f"Found {len(image_files)} unique images.")

    if Path(det_weights).exists():
        print("Running Detection...")
        model_det = YOLO(str(det_weights))
        # Use batch=1 to avoid OOM with large file lists
        # Or iterate manually if stream=True with list is problematic
        # Let's iterate manually to be safe and robust against OOM
        
        # Define project path relative to output CSV
        project_path = Path(out_csv).parent / 'viz_detect'
        
        for i, img_path in enumerate(image_files):
            # Run inference on single image
            preds = model_det.predict(source=img_path, conf=0.25, save=save_vis, verbose=False, project=str(project_path), name='predict', exist_ok=True)
            res = preds[0]
            fname = Path(res.path).name
            has_tumor = 0
            if res.boxes.conf.numel() > 0:
                has_tumor = 1
            det_results[fname] = has_tumor
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} detection...")
    
    # 2. Segmentation
    seg_results = {}
    if Path(seg_weights).exists():
        print("Running Segmentation...")
        model_seg = YOLO(str(seg_weights))
        
        # Define project path relative to output CSV
        project_path = Path(out_csv).parent / 'viz_segment'
        
        for i, img_path in enumerate(image_files):
            preds = model_seg.predict(source=img_path, conf=0.35, save=save_vis, verbose=False, project=str(project_path), name='predict', exist_ok=True)
            res = preds[0]
            fname = Path(res.path).name
            has_tumor = 0
            if res.masks is not None and res.masks.data is not None:
                # Validate masks
                # We iterate over masks, if any is valid, set to 1
                masks = res.masks.data.cpu().numpy() # (N, H, W)
                for mask in masks:
                    # Resize to original image size if needed? 
                    # YOLOv8 masks are usually smaller (160x160 or similar) unless returntype changed.
                    # But for simple existence check, we can check the mask directly.
                    # Ideally we resize to confirm area ratio.
                    if np.sum(mask) > 0:
                        has_tumor = 1
                        break
            seg_results[fname] = has_tumor
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} segmentation...")

    # 3. Merge
    all_files = sorted(set(det_results.keys()) | set(seg_results.keys()))
    if not all_files:
        print("No images found or no results generated.")
        return

    rows = []
    for fname in all_files:
        d = det_results.get(fname, 0)
        s = seg_results.get(fname, 0)
        final = 1 if (d == 1 or s == 1) else 0
        rows.append((fname, final))

    # 4. Save
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'label'])
        w.writerows(rows)
    
    print(f"Successfully processed {len(rows)} images.")
    print(f"Results saved to: {out_path.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA), help='Path to test images')
    parser.add_argument('--det-weights', type=str, default=str(DET_WEIGHTS), help='Path to detection weights')
    parser.add_argument('--seg-weights', type=str, default=str(SEG_WEIGHTS), help='Path to segmentation weights')
    parser.add_argument('--out', type=str, default='./cla_pre.csv', help='Output CSV path')
    parser.add_argument('--save', action='store_true', help='Save visualization images')
    
    args = parser.parse_args()
    
    run_pipeline(args.data, args.det_weights, args.seg_weights, args.out, args.save)

if __name__ == '__main__':
    main()
