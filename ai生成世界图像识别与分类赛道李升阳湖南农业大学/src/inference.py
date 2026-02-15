import json
from pathlib import Path
import csv
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import pickle
try:
    from .model import create_model
    from .utils import list_images, infer_label_from_path, find_threshold_max_precision
    from .cam import GradCAM, overlay_cam, cam_to_mask, mask_to_bbox, central_roi_mask, intersection_ratio, bbox_touches_border
except ImportError:
    from model import create_model
    from utils import list_images, infer_label_from_path, find_threshold_max_precision
    from cam import GradCAM, overlay_cam, cam_to_mask, mask_to_bbox, central_roi_mask, intersection_ratio, bbox_touches_border

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _try_load_efficientnet(weights_path: Path, device, model_name: str):
    model = create_model(model_name=model_name, pretrained=False)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception as e:
        msg = str(e)
        if isinstance(e, pickle.UnpicklingError) or 'Weights only load failed' in msg or 'Unsupported global' in msg:
            return None
        return None
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    if isinstance(state, dict) and 'model' in state and hasattr(state['model'], 'state_dict'):
        state = state['model'].state_dict()
    try:
        model.load_state_dict(state)
    except Exception:
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            return None
    model.to(device)
    model.eval()
    return model

def _try_load_yolo(weights_path: Path):
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        return YOLO(str(weights_path))
    except Exception:
        return None

def base_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def tta_variants(img: Image.Image):
    return [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.rotate(10, expand=False),
        img.rotate(-10, expand=False),
    ]

def load_meta(weights_dir: Path):
    meta_path = weights_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def run(
    data_dir: Path,
    weights_path: Path,
    out_csv: Path,
    model_name: str = 'efficientnetv2_s',
    img_size: int = 224,
    save_cam_dir: Path = None,
    cam_limit: int = 50,
    tune_threshold: bool = False,
    min_recall: float = 0.9,
    mask_percentile: float = 80.0,
    roi_margin: float = 0.1,
    border_ratio: float = 0.02,
    high_conf_thresh: float = 0.8,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    yolo = None
    if weights_path.exists():
        model = _try_load_efficientnet(weights_path, device, model_name=model_name)
        if model is None:
            yolo = _try_load_yolo(weights_path)
    if model is None and yolo is None:
        raise FileNotFoundError(f"无法加载模型权重: {weights_path}")
    tfm = base_transform(img_size)
    images = list_images(Path(data_dir))
    rows = []
    threshold = 0.5
    if weights_path.exists():
        if tune_threshold:
            _res = tune_threshold_precision(data_dir=data_dir, weights_path=weights_path, model_name=model_name, img_size=img_size, min_recall=min_recall)
        meta = load_meta(weights_path.parent)
        if 'threshold' in meta:
            threshold = float(meta['threshold'])
    cammer = None
    if save_cam_dir is not None:
        save_cam_dir.mkdir(parents=True, exist_ok=True)
        if model is not None:
            cammer = GradCAM(model)
    count_cam = 0
    for p in images:
        img = Image.open(p).convert('RGB')
        prob = None
        if model is not None:
            variants = tta_variants(img)
            batch = torch.stack([tfm(v) for v in variants]).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(batch).squeeze(1)
                    probs = torch.sigmoid(logits)
            prob = float(probs.mean().cpu().item())
            pred = 1 if prob >= threshold else 0
        else:
            device_arg = 0 if torch.cuda.is_available() else 'cpu'
            res = yolo.predict(source=str(p), conf=0.25, iou=0.7, imgsz=640, device=device_arg, verbose=False, max_det=10)
            pred = 1 if (res and hasattr(res[0], 'boxes') and res[0].boxes is not None and len(res[0].boxes) > 0) else 0
        rows.append((p.name, pred))
        if model is not None and pred == 1:
            x = tfm(img).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                cam = cammer.generate(x) if cammer is not None else GradCAM(model).generate(x)
            m_small = cam_to_mask(cam, percentile=mask_percentile)
            w, h = img.size
            m_img = Image.fromarray((m_small * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
            m_arr = np.array(m_img)
            bbox = mask_to_bbox(m_arr)
            high_conf = (prob is not None) and (prob >= high_conf_thresh)
            if not high_conf:
                roi = central_roi_mask(w, h, margin_ratio=roi_margin)
                ratio = intersection_ratio(m_arr, roi)
                if ratio < 0.3 and bbox_touches_border(bbox, w, h, border_ratio=border_ratio):
                    pred = 0
                    rows[-1] = (p.name, pred)
        if cammer is not None and pred == 1 and count_cam < cam_limit:
            vis = overlay_cam(img, cam)
            vis.save(save_cam_dir / f"{p.stem}_cam.png")
            m_img.save(save_cam_dir / f"{p.stem}_mask.png")
            if bbox is not None:
                with open(save_cam_dir / f"{p.stem}_bbox.json", 'w', encoding='utf-8') as f:
                    json.dump(bbox, f, ensure_ascii=False, indent=2)
            count_cam += 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'pred'])
        w.writerows(rows)

def evaluate(data_dir: Path, weights_path: Path, model_name: str = 'efficientnetv2_s', img_size: int = 224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _try_load_efficientnet(weights_path, device, model_name=model_name)
    yolo = None
    if model is None:
        yolo = _try_load_yolo(weights_path)
        if yolo is None:
            raise RuntimeError(f"无法加载评估模型: {weights_path}")
    tfm = base_transform(img_size)
    threshold = 0.5
    meta = load_meta(weights_path.parent)
    if 'threshold' in meta:
        threshold = float(meta['threshold'])
    images = list_images(Path(data_dir))
    preds = []
    gts = []
    for p in images:
        gt = infer_label_from_path(p)
        if gt is None:
            continue
        img = Image.open(p).convert('RGB')
        if model is not None:
            variants = tta_variants(img)
            batch = torch.stack([tfm(v) for v in variants]).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(batch).squeeze(1)
                    probs = torch.sigmoid(logits)
            prob = float(probs.mean().cpu().item())
            pred = 1 if prob >= threshold else 0
        else:
            device_arg = 0 if torch.cuda.is_available() else 'cpu'
            res = yolo.predict(source=str(p), conf=0.25, iou=0.7, imgsz=640, device=device_arg, verbose=False, max_det=10)
            pred = 1 if (res and hasattr(res[0], 'boxes') and res[0].boxes is not None and len(res[0].boxes) > 0) else 0
        preds.append(pred)
        gts.append(int(gt))
    if len(gts) == 0:
        return {
            'count': 0,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'threshold': threshold,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
        }
    preds = np.array(preds, dtype=np.int32)
    gts = np.array(gts, dtype=np.int32)
    tp = int(np.sum((preds == 1) & (gts == 1)))
    tn = int(np.sum((preds == 0) & (gts == 0)))
    fp = int(np.sum((preds == 1) & (gts == 0)))
    fn = int(np.sum((preds == 0) & (gts == 1)))
    acc = float((preds == gts).mean())
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    result = {
        'count': int(len(gts)),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': threshold,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }
    metrics_path = weights_path.parent / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def collect_probs_labels(data_dir: Path, weights_path: Path, model_name: str = 'efficientnetv2_s', img_size: int = 224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name=model_name, pretrained=False)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tfm = base_transform(img_size)
    images = list_images(Path(data_dir))
    probs = []
    labels = []
    for p in images:
        gt = infer_label_from_path(p)
        if gt is None:
            continue
        img = Image.open(p).convert('RGB')
        variants = tta_variants(img)
        batch = torch.stack([tfm(v) for v in variants]).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(batch).squeeze(1)
                ps = torch.sigmoid(logits)
        prob = float(ps.mean().cpu().item())
        probs.append(prob)
        labels.append(int(gt))
    return np.array(probs, dtype=np.float32), np.array(labels, dtype=np.int32)

def tune_threshold_precision(data_dir: Path, weights_path: Path, model_name: str = 'efficientnetv2_s', img_size: int = 224, min_recall: float = 0.9):
    probs, labels = collect_probs_labels(data_dir, weights_path, model_name=model_name, img_size=img_size)
    t, prec, rec = find_threshold_max_precision(probs, labels, grid=None, min_recall=min_recall)
    meta_path = weights_path.parent / 'meta.json'
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    meta['threshold'] = float(t)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {'threshold': float(t), 'precision': float(prec), 'recall': float(rec)}
