import os
import csv
import time
import subprocess
from pathlib import Path


BASE = Path.cwd()
DATA_YAML = BASE / '肿瘤模型数据集' / 'yolo_ds' / 'tumor.yaml'
PROJECT = BASE / 'runs_yolo'
INIT_MODEL = PROJECT / 'tumor_yolov8m_30M_ab960_auto4' / 'weights' / 'best.pt'


def build_train_code(name: str,
                    imgsz: int = 960,
                    epochs: int = 120,
                    batch: int = 4,
                    lr0: float = 0.002,
                    mosaic: float = 0.2,
                    mixup: float = 0.1,
                    auto_augment: str = 'randaugment',
                    erasing: float = 0.2,
                    rect: bool = True,
                    cos_lr: bool = True,
                    patience: int = 150,
                    single_cls: bool = True,
                    box: float = 9.0,
                    dfl: float = 2.0,
                    save_period: int = 0,
                    workers: int = 2,
                    plots: bool = False) -> str:
    py = (
        "from ultralytics import YOLO; import torch; "
        f"m=YOLO(r'{INIT_MODEL if INIT_MODEL.exists() else 'yolov8m.pt'}'); "
        f"m.train(data=r'{DATA_YAML}', imgsz={imgsz}, epochs={epochs}, batch={batch}, "
        "device=0 if torch.cuda.is_available() else 'cpu', "
        f"project=r'{PROJECT}', name=r'{name}', workers={workers}, cache=True, amp=True, "
        f"optimizer='AdamW', lr0={lr0}, single_cls={str(single_cls)}, mosaic={mosaic}, mixup={mixup}, "
        f"auto_augment='{auto_augment}', erasing={erasing}, rect={str(rect)}, cos_lr={str(cos_lr)}, "
        f"patience={patience}, verbose=True, save_period={save_period}, box={box}, dfl={dfl}, plots={str(plots)})"
    )
    return py


def prepare_augmented_dataset(n_per_image: int = 2) -> Path:
    from PIL import Image
    import shutil
    import albumentations as A
    ds_root = BASE / '肿瘤模型数据集' / 'yolo_ds'
    no_root = BASE / '肿瘤模型数据集' / 'no'
    train_img_dir = ds_root / 'images' / 'train'
    train_lbl_dir = ds_root / 'labels' / 'train'
    val_img_dir = ds_root / 'images' / 'val'
    out_img_dir = ds_root / 'images' / 'train_mix'
    out_lbl_dir = ds_root / 'labels' / 'train_mix'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    # copy originals to mix
    for p in sorted(train_img_dir.glob('*')):
        if p.is_file():
            dst = out_img_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            lbl = train_lbl_dir / (p.stem + '.txt')
            dst_lbl = out_lbl_dir / (p.stem + '.txt')
            if lbl.exists():
                if not dst_lbl.exists():
                    shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.touch(exist_ok=True)
    # add hard negatives from no_root
    for p in sorted(no_root.glob('*')):
        if p.is_file():
            prefixed = f"no_{p.name}"
            dst = out_img_dir / prefixed
            if not dst.exists():
                shutil.copy2(p, dst)
            dst_lbl = out_lbl_dir / (Path(prefixed).stem + '.txt')
            if not dst_lbl.exists():
                dst_lbl.touch(exist_ok=True)
    # augmentation pipeline
    aug = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(p=0.2),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.ToGray(p=0.05),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))
    # generate augmented samples
    for p in sorted(train_img_dir.glob('*')):
        if not p.is_file():
            continue
        lbl = train_lbl_dir / (p.stem + '.txt')
        bboxes = []
        class_labels = []
        if lbl.exists():
            try:
                lines = [ln.strip() for ln in lbl.read_text(encoding='utf-8').splitlines() if ln.strip()]
                for ln in lines:
                    parts = ln.split()
                    if len(parts) >= 5:
                        # YOLO: cls cx cy w h (normalized)
                        cls = int(float(parts[0]))
                        cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(cls)
            except Exception:
                bboxes = []
                class_labels = []
        im = Image.open(p).convert('RGB')
        w, h = im.size
        for i in range(n_per_image):
            try:
                transformed = aug(image=np.array(im), bboxes=bboxes, class_labels=class_labels)
                out_im = transformed['image']
                out_boxes = transformed.get('bboxes', [])
                out_cls = transformed.get('class_labels', [])
            except Exception:
                out_im = np.array(im)
                out_boxes = bboxes
                out_cls = class_labels
            aug_name = f"{p.stem}_aug{i+1}{p.suffix}"
            (out_img_dir / aug_name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out_im).save(out_img_dir / aug_name)
            # write label
            with open(out_lbl_dir / (Path(aug_name).stem + '.txt'), 'w', encoding='utf-8') as f:
                for bb, cls in zip(out_boxes, out_cls):
                    cx, cy, bw, bh = bb
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    # augment negatives as well
    for p in sorted(no_root.glob('*')):
        if not p.is_file():
            continue
        im = Image.open(p).convert('RGB')
        bboxes = []
        class_labels = []
        for i in range(n_per_image):
            try:
                transformed = aug(image=np.array(im), bboxes=bboxes, class_labels=class_labels)
                out_im = transformed['image']
            except Exception:
                out_im = np.array(im)
            aug_name = f"no_{Path(p).stem}_aug{i+1}{Path(p).suffix}"
            (out_img_dir / aug_name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out_im).save(out_img_dir / aug_name)
            with open(out_lbl_dir / (Path(aug_name).stem + '.txt'), 'w', encoding='utf-8') as f:
                pass
    # write new yaml
    yaml_path = ds_root / 'tumor_aug.yaml'
    content = (
        f"path: {ds_root.resolve()}\n"
        f"train: {out_img_dir.resolve()}\n"
        f"val: {val_img_dir.resolve()}\n"
        f"names: [tumor]\n"
    )
    yaml_path.write_text(content, encoding='utf-8')
    return yaml_path

def read_last_metrics(results_csv: Path):
    if not results_csv.exists():
        return None
    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        header = rows[0]
        last = rows[-1]
        hmap = {h: i for i, h in enumerate(header)}
        def get(key, default=None):
            try:
                return float(last[hmap[key]])
            except Exception:
                return default
        epoch = int(get('epoch', 0)) if 'epoch' in header else None
        prec = get('metrics/precision(B)', None)
        rec = get('metrics/recall(B)', None)
        map50 = get('metrics/mAP50(B)', None)
        map95 = get('metrics/mAP50-95(B)', None)
        return {
            'epoch': epoch,
            'precision': prec,
            'recall': rec,
            'map50': map50,
            'map95': map95,
        }
    except Exception:
        return None


def monitor_and_adjust(run_name: str,
                       initial_cfg: dict,
                       alt_cfg: dict,
                       check_interval_s: int = 20,
                       first_check_epoch: int = 30,
                       min_prec_target: float = 0.80,
                       min_rec_target: float = 0.70,
                       min_map50_target: float = 0.75):
    code = build_train_code(name=run_name, **initial_cfg)
    env = os.environ.copy()
    env['WANDB_DISABLED'] = '1'
    env['ULTRALYTICS_WANDB'] = '0'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    proc = subprocess.Popen(['python', '-c', code], env=env)
    results_csv = PROJECT / run_name / 'results.csv'
    best_ok = False
    while True:
        time.sleep(check_interval_s)
        if proc.poll() is not None:
            break
        m = read_last_metrics(results_csv)
        if not m or m['epoch'] is None:
            continue
        if m['epoch'] >= first_check_epoch:
            prec = m['precision'] or 0.0
            rec = m['recall'] or 0.0
            mp = m['map50'] or 0.0
            if prec >= min_prec_target and rec >= min_rec_target and mp >= min_map50_target:
                best_ok = True
                continue
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            run_name2 = f"{run_name}_adjust"
            code2 = build_train_code(name=run_name2, **alt_cfg)
            proc = subprocess.Popen(['python', '-c', code2], env=env)
            results_csv = PROJECT / run_name2 / 'results.csv'
            # tighten targets for second phase
            min_prec_target = max(min_prec_target, 0.85)
            min_rec_target = max(min_rec_target, 0.72)
            min_map50_target = max(min_map50_target, 0.76)
            first_check_epoch = 50
            run_name = run_name2
    return best_ok, run_name


if __name__ == '__main__':
    import numpy as np
    # prepare offline augmented dataset and switch yaml
    DATA_YAML = prepare_augmented_dataset(n_per_image=2)
    initial = {
        'imgsz': 1280,
        'epochs': 160,
        'batch': 4,
        'lr0': 0.002,
        'mosaic': 0.2,
        'mixup': 0.1,
        'auto_augment': 'randaugment',
        'erasing': 0.2,
        'rect': True,
        'cos_lr': True,
        'patience': 180,
        'single_cls': True,
        'box': 9.0,
        'dfl': 2.0,
        'save_period': 0,
        'workers': 2,
        'plots': False,
    }
    alt = {
        'imgsz': 1280,
        'epochs': 200,
        'batch': 4,
        'lr0': 0.0018,
        'mosaic': 0.1,
        'mixup': 0.05,
        'auto_augment': 'randaugment',
        'erasing': 0.15,
        'rect': True,
        'cos_lr': True,
        'patience': 200,
        'single_cls': True,
        'box': 9.5,
        'dfl': 2.0,
        'save_period': 0,
        'workers': 2,
        'plots': False,
    }
    ok, final_run = monitor_and_adjust('tumor_yolov8m_30M_gridB_retrain', initial, alt)
    print(f"Training finished. Success={ok}. Final run: {final_run}")

