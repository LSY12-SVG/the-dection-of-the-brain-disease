import os
import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from ultralytics import YOLO

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'}

def list_images(root: Path) -> List[Path]:
    paths: List[Path] = []
    # Use set to avoid duplicates on case-insensitive filesystems (Windows)
    seen = set()
    for ext in IMG_EXTS:
        for p in root.rglob(f"*{ext}"):
            # Resolve to absolute path to ensure uniqueness check is robust
            abs_p = p.resolve()
            if abs_p not in seen:
                paths.append(p)
                seen.add(abs_p)
    return sorted(paths)

def central_roi_contains(cx: float, cy: float, w: int, h: int, margin_ratio: float = 0.1) -> bool:
    mx = w * margin_ratio
    my = h * margin_ratio
    return (mx <= cx <= (w - mx)) and (my <= cy <= (h - my))

def touches_border(x1: float, y1: float, x2: float, y2: float, w: int, h: int, border_ratio: float = 0.05) -> bool:
    bx = w * border_ratio
    by = h * border_ratio
    return (x1 <= bx) or (y1 <= by) or (x2 >= (w - bx)) or (y2 >= (h - by))

def area_ratio(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> float:
    a = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return a / float(w * h) if w > 0 and h > 0 else 0.0

def is_valid_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int,
                 min_area: float, max_area: float, border_ratio: float, center_margin: float) -> bool:
    if touches_border(x1, y1, x2, y2, w, h, border_ratio):
        return False
    ar = area_ratio(x1, y1, x2, y2, w, h)
    if ar < min_area or ar > max_area:
        return False
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    if not central_roi_contains(cx, cy, w, h, center_margin):
        return False
    return True


def infer_dir(model_path: Path, data_dir: Path, out_dir: Path, imgsz: int = 1024, conf: float = 0.35, iou: float = 0.5,
              min_area: float = 0.001, max_area: float = 0.5, border_ratio: float = 0.05, center_margin: float = 0.1,
              max_det: int = 1, augment: bool = False) -> Tuple[int, int]:
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(str(model_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(data_dir)
    csv_path = out_dir / 'results.csv'
    rows: List[Tuple[str, str]] = []
    for p in images:
        img = Image.open(p).convert('RGB')
        w, h = img.size
        results = model.predict(source=str(p), imgsz=imgsz, conf=conf, iou=iou, device=device, project=str(out_dir), name='pred', save=True, verbose=False, max_det=max_det, augment=augment)
        pos = False
        if results:
            r = results[0]
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                c = float(b.conf.item())
                x1, y1, x2, y2 = xyxy
                if not is_valid_box(x1, y1, x2, y2, w, h, min_area, max_area, border_ratio, center_margin):
                    continue
                if c >= conf:
                    pos = True
                    break
        rows.append((p.name, 'yes' if pos else 'no'))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.writer(f)
        wtr.writerow(['filename', 'tumor'])
        wtr.writerows(rows)
    return int(sum(1 for _, lab in rows if lab == 'yes')), int(sum(1 for _, lab in rows if lab == 'no'))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=False)
    ap.add_argument('--data', type=str, required=False)
    ap.add_argument('--out', type=str, required=False)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--min-area', type=float, default=0.001)
    ap.add_argument('--max-area', type=float, default=0.5)
    ap.add_argument('--border-ratio', type=float, default=0.05)
    ap.add_argument('--center-margin', type=float, default=0.1)
    ap.add_argument('--max-det', type=int, default=1)
    ap.add_argument('--augment', action='store_true')
    args = ap.parse_args()
    base = Path.cwd()
    model_path = Path(args.model) if args.model else (base / 'runs_yolo' / 'tumor_yolov8m_30M_full5' / 'weights' / 'best.pt')
    data_dir = Path(args.data) if args.data else (base / '测试数据集')
    out_dir = Path(args.out) if args.out else (base / 'runs_yolo' / 'infer_test_dataset')
    yes_count, no_count = infer_dir(model_path, data_dir, out_dir, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                    min_area=args.min_area, max_area=args.max_area, border_ratio=args.border_ratio,
                                    center_margin=args.center_margin, max_det=args.max_det, augment=args.augment)
    print(f"Done. Positives: {yes_count}, Negatives: {no_count}. CSV: {out_dir / 'results.csv'}")
