import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
import tqdm

def find_image_path(images_dir, stem):
    """Find image with various extensions."""
    for ext in ['.jpg', '.jpeg', '.JPG', '.png', '.bmp']:
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None

def yolo_to_bbox(x_c, y_c, w, h, img_w, img_h):
    """Convert YOLO format to x, y, w, h (absolute)."""
    x = int((x_c - w / 2) * img_w)
    y = int((y_c - h / 2) * img_h)
    bw = int(w * img_w)
    bh = int(h * img_h)
    # Clip to image boundaries
    x = max(0, x)
    y = max(0, y)
    bw = min(img_w - x, bw)
    bh = min(img_h - y, bh)
    return x, y, bw, bh

def grabcut_segment(img, bbox, iter_count=5):
    """
    Apply GrabCut segmentation.
    bbox: (x, y, w, h)
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Initialize with rect
    try:
        cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        # If rect is invalid (too small), return empty mask
        return np.zeros(img.shape[:2], np.uint8)

    # Mask: 0=BG, 1=FG, 2=Prob_BG, 3=Prob_FG
    # We want 1 and 3
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

def mask_to_polygon(mask):
    """Convert binary mask to polygon (normalized)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    h, w = mask.shape
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # Filter small noise
            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) < 3:
                continue
                
            # Normalize
            poly = approx.flatten().astype(float)
            poly[0::2] /= w
            poly[1::2] /= h
            
            # Clip
            poly = np.clip(poly, 0.0, 1.0)
            polygons.append(poly)
            
    return polygons

def process_dataset(img_dir, label_dir, out_dir, viz_dir=None):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    out_labels = out_dir / 'labels' / 'train'
    out_images = out_dir / 'images' / 'train'
    
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)
    
    if viz_dir:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)

    labels = list(label_dir.glob('*.txt'))
    
    print(f"Found {len(labels)} label files.")
    
    for label_path in tqdm.tqdm(labels):
        stem = label_path.stem
        img_path = find_image_path(img_dir, stem)
        
        if img_path is None:
            # print(f"Warning: No image found for {label_path.name}")
            continue
            
        # Copy image to new dataset
        shutil.copy2(img_path, out_images / img_path.name)
        
        # Read image (handle unicode paths)
        # img = cv2.imread(str(img_path))
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        h, w = img.shape[:2]
        
        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        
        # Visualization image copy
        viz_img = img.copy() if viz_dir else None
        
        for line in lines:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            x_c, y_c, bw, bh = parts[1:]
            
            bbox = yolo_to_bbox(x_c, y_c, bw, bh, w, h)
            bx, by, b_w, b_h = bbox
            
            # Skip if bbox is too small
            if b_w < 2 or b_h < 2:
                continue
                
            # Apply GrabCut
            mask = grabcut_segment(img, bbox)
            
            # Get polygons
            polys = mask_to_polygon(mask)
            
            if not polys:
                # Fallback: if GrabCut returns empty (or fails), use the bbox as a polygon (rectangle)
                # x1, y1, x2, y2, x2, y2, x1, y2
                # But actually, GrabCut usually works if there is contrast. 
                # If it fails completely, maybe just use the box? 
                # Let's use the box as a fallback polygon.
                x1 = max(0, x_c - bw/2)
                y1 = max(0, y_c - bh/2)
                x2 = min(1, x_c + bw/2)
                y2 = min(1, y_c + bh/2)
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                new_lines.append(f"{cls_id} {' '.join(map(str, poly))}")
                
                if viz_img is not None:
                     cv2.rectangle(viz_img, (bx, by), (bx+b_w, by+b_h), (0, 0, 255), 2)
            else:
                for poly in polys:
                    new_lines.append(f"{cls_id} {' '.join(map(str, poly))}")
                    
                    if viz_img is not None:
                        pts = poly.reshape(-1, 2)
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                        pts = pts.astype(np.int32)
                        cv2.polylines(viz_img, [pts], True, (0, 255, 0), 2)

        # Save new label file
        with open(out_labels / label_path.name, 'w') as f:
            for nl in new_lines:
                f.write(nl + '\n')
                
        if viz_dir and viz_img is not None:
            # cv2.imwrite(str(viz_dir / f"{stem}_seg.jpg"), viz_img)
            success, buffer = cv2.imencode(".jpg", viz_img)
            if success:
                with open(str(viz_dir / f"{stem}_seg.jpg"), "wb") as f:
                    f.write(buffer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--viz', type=str, default=None)
    args = parser.parse_args()
    
    process_dataset(args.images, args.labels, args.out, args.viz)

if __name__ == '__main__':
    main()
