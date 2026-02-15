import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    from .utils import list_images
    from .cam import central_roi_mask, mask_to_bbox, intersection_ratio, bbox_touches_border
except ImportError:
    from utils import list_images
    from cam import central_roi_mask, mask_to_bbox, intersection_ratio, bbox_touches_border

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'}

def validate_mask(mask_arr: np.ndarray, w: int, h: int, min_area: float, max_area: float, center_margin: float, border_ratio: float) -> bool:
    area_ratio = float(np.sum(mask_arr > 0)) / float(mask_arr.size) if mask_arr.size > 0 else 0.0
    if area_ratio < min_area or area_ratio > max_area:
        return False
    bbox = mask_to_bbox(mask_arr)
    roi = central_roi_mask(w, h, margin_ratio=center_margin)
    inter = intersection_ratio(mask_arr, roi)
    if inter < 0.2:
        return False
    if bbox_touches_border(bbox, w, h, border_ratio=border_ratio):
        return False
    return True

def infer_seg(model_path: Path, data_dir: Path, out_dir: Path, imgsz: int = 1024, conf: float = 0.35, iou: float = 0.5, max_det: int = 3, min_area: float = 0.0005, max_area: float = 0.6, center_margin: float = 0.1, border_ratio: float = 0.03, augment: bool = False) -> Tuple[int, int]:
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(str(model_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'seg_masks').mkdir(parents=True, exist_ok=True)
    (out_dir / 'seg_overlay').mkdir(parents=True, exist_ok=True)
    
    # Custom list_images to avoid duplicates
    paths = []
    seen = set()
    for ext in IMG_EXTS:
        for p in data_dir.rglob(f"*{ext}"):
            abs_p = p.resolve()
            if abs_p not in seen:
                paths.append(p)
                seen.add(abs_p)
    images = sorted(paths)
    
    csv_path = out_dir / 'seg_results.csv'
    rows: List[Tuple[str, str]] = []
    for p in images:
        img = Image.open(p).convert('RGB')
        w, h = img.size
        res = model.predict(source=str(p), imgsz=imgsz, conf=conf, iou=iou, device=device, project=str(out_dir), name='pred', save=True, verbose=False, max_det=max_det, augment=augment)
        pos = False
        if res and hasattr(res[0], 'masks') and res[0].masks is not None and res[0].masks.data is not None:
            arr = res[0].masks.data.cpu().numpy()
            for mnp in arr:
                marr = (Image.fromarray((mnp * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST))
                m_arr = np.array(marr)
                if validate_mask(m_arr, w, h, min_area=min_area, max_area=max_area, center_margin=center_margin, border_ratio=border_ratio):
                    pos = True
                    mask_path = out_dir / 'seg_masks' / f"{p.stem}_mask.png"
                    marr.save(mask_path)
                    ov = np.array(img).astype(np.uint8)
                    mm = (m_arr > 0).astype(np.uint8)
                    red = ov.copy()
                    red[..., 0] = 255
                    alpha = (mm * 128).astype(np.uint8)
                    overlay = Image.fromarray(ov)
                    red_img = Image.fromarray(red)
                    overlay = Image.alpha_composite(overlay.convert('RGBA'), Image.merge('RGBA', (red_img.split()[0], red_img.split()[1], red_img.split()[2], Image.fromarray(alpha))))
                    overlay.save(out_dir / 'seg_overlay' / f"{p.stem}_overlay.png")
                    break
        rows.append((p.name, 'yes' if pos else 'no'))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.writer(f)
        wtr.writerow(['filename', 'tumor'])
        wtr.writerows(rows)
    return int(sum(1 for _, lab in rows if lab == 'yes')), int(sum(1 for _, lab in rows if lab == 'no'))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--max-det', type=int, default=3)
    ap.add_argument('--min-area', type=float, default=0.0005)
    ap.add_argument('--max-area', type=float, default=0.6)
    ap.add_argument('--center-margin', type=float, default=0.1)
    ap.add_argument('--border-ratio', type=float, default=0.03)
    ap.add_argument('--augment', action='store_true')
    return ap.parse_args()

def main():
    a = parse_args()
    model_path = Path(a.model)
    data_dir = Path(a.data)
    out_dir = Path(a.out)
    infer_seg(model_path, data_dir, out_dir, imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det, min_area=a.min_area, max_area=a.max_area, center_margin=a.center_margin, border_ratio=a.border_ratio, augment=a.augment)

if __name__ == '__main__':
    main()
