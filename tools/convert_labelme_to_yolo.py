import json
import random
import shutil
from pathlib import Path


def is_image(p: Path):
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}


def list_images(root: Path):
    return sorted([p for p in root.rglob('*') if p.is_file() and is_image(p)])


def convert_labelme_bbox(jp: Path):
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    W = int(data.get('imageWidth', 0))
    H = int(data.get('imageHeight', 0))
    lines = []
    for sh in data.get('shapes', []):
        pts = sh.get('points', [])
        if not pts:
            continue
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        cx = ((x1 + x2) / 2.0) / W if W > 0 else 0.0
        cy = ((y1 + y2) / 2.0) / H if H > 0 else 0.0
        w = (x2 - x1) / W if W > 0 else 0.0
        h = (y2 - y1) / H if H > 0 else 0.0
        lines.append(f"0 {cx} {cy} {w} {h}")
    return lines


def write_yaml(path: Path, train_dir: Path, val_dir: Path):
    content = (
        f"path: {path.resolve()}\n"
        f"train: {train_dir.resolve()}\n"
        f"val: {val_dir.resolve()}\n"
        f"names: [tumor]\n"
    )
    with open(path / 'tumor.yaml', 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='dataset root that contains yolo用数据集 and yolo用标注')
    ap.add_argument('--out', type=str, default=None, help='output yolo dataset dir, default: <root>/yolo_ds')
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--extra-neg', type=str, nargs='*', default=None, help='additional directories of negative images (no tumor)')
    args = ap.parse_args()

    root = Path(args.root)
    imgs = root / 'yolo用数据集'
    ann = root / 'yolo用标注'
    out = Path(args.out) if args.out is not None else (root / 'yolo_ds')

    random.seed(args.seed)
    (out / 'images/train').mkdir(parents=True, exist_ok=True)
    (out / 'images/val').mkdir(parents=True, exist_ok=True)
    (out / 'labels/train').mkdir(parents=True, exist_ok=True)
    (out / 'labels/val').mkdir(parents=True, exist_ok=True)

    files = list_images(imgs)
    if args.extra_neg:
        for d in args.extra_neg:
            neg_root = Path(d)
            if neg_root.exists():
                files.extend(list_images(neg_root))
    random.shuffle(files)
    split = int(len(files) * (1.0 - args.val_ratio))
    train_files = files[:split]
    val_files = files[split:]

    def convert_one(p: Path, subset: str):
        jp = ann / (p.stem + '.json')
        lbl_path = out / 'labels' / subset / (p.stem + '.txt')
        img_out = out / 'images' / subset / p.name
        shutil.copy2(p, img_out)
        if jp.exists():
            lines = convert_labelme_bbox(jp)
            with open(lbl_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        else:
            open(lbl_path, 'w').close()

    for p in train_files:
        convert_one(p, 'train')
    for p in val_files:
        convert_one(p, 'val')

    write_yaml(out, out / 'images/train', out / 'images/val')
    print('yolo dataset ready:', out, 'train:', len(train_files), 'val:', len(val_files))


if __name__ == '__main__':
    main()

