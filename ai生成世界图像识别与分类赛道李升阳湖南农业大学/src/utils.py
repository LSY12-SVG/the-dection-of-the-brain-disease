import re
import random
from pathlib import Path
import numpy as np
import torch

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_label_from_name(name: str):
    n = name.lower()
    if n.startswith('y') or '/y/' in n:
        return 1
    if n.startswith('n') or '/n/' in n:
        return 0
    if any(k in n for k in ["tumor", "positive", "_1", "-1", "pos", " yes", "yes", "/yes/"]):
        return 1
    if any(k in n for k in ["no_tumor", "negative", "_0", "-0", "neg", " normal", "normal", " no", "no", "/no/"]):
        return 0
    m = re.search(r"(^|[_\-])([01])([_\-]|$)", n)
    return int(m.group(2)) if m else None

def infer_label_from_path(p: Path):
    s = str(p).lower()
    parts = [part.lower() for part in p.parts]
    parent = p.parent.name.lower() if p.parent is not None else ''
    if 'yes' in parts or parent == 'yes':
        return 1
    if 'no' in parts or parent == 'no':
        return 0
    return infer_label_from_name(p.name)

def list_images(root: Path):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(list(root.rglob(f"*{ext}")))
    return sorted(paths)

def best_threshold(probs: np.ndarray, labels: np.ndarray, grid=None):
    if grid is None:
        grid = np.linspace(0.3, 0.7, 41)
    accs = []
    for t in grid:
        preds = (probs >= t).astype(np.int32)
        acc = (preds == labels).mean()
        accs.append(acc)
    i = int(np.argmax(accs))
    return float(grid[i]), float(accs[i])

def accuracy_from_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(np.int32)
    return float((preds == labels).mean())

def count_pos_neg(labels):
    pos = int(np.sum(np.array(labels) == 1))
    neg = int(np.sum(np.array(labels) == 0))
    return pos, neg

def precision_recall_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return prec, rec

def find_threshold_max_precision(probs: np.ndarray, labels: np.ndarray, grid=None, min_recall: float = 0.9):
    if grid is None:
        grid = np.linspace(0.3, 0.8, 51)
    best_t = None
    best_prec = -1.0
    best_rec = 0.0
    for t in grid:
        prec, rec = precision_recall_from_probs(probs, labels, t)
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_rec = rec
            best_t = float(t)
    if best_t is None:
        return float(0.5), float(0.0), float(0.0)
    return best_t, best_prec, best_rec
