import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
try:
    from .dataset import ImageBinaryDataset, get_transforms, build_paths_and_labels
    from .model import create_model, freeze_backbone, head_parameters, count_parameters
    from .utils import set_seed, best_threshold, accuracy_from_threshold, count_pos_neg
except ImportError:
    from dataset import ImageBinaryDataset, get_transforms, build_paths_and_labels
    from model import create_model, freeze_backbone, head_parameters, count_parameters
    from utils import set_seed, best_threshold, accuracy_from_threshold, count_pos_neg

def train_one_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total = 0
    bce = None
    if pos_weight is not None:
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x).squeeze(1)
            loss = bce(logits, y) if bce is not None else F.binary_cross_entropy_with_logits(logits, y)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        loss.backward()
        optimizer.step()
        total += x.size(0)
    return total

def eval_model(model, loader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x).squeeze(1)
                p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return probs, labels

def run_fold(paths, labels, fold_idx, k, data_root, out_dir, model_name, img_size, seed, epochs_head=10, epochs_ft=20, batch_size=32, num_workers=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        if i == fold_idx:
            break
    train_subset = Subset(ImageBinaryDataset(data_root, paths=paths, labels=labels, transform=get_transforms(img_size, train=True)), train_idx)
    val_subset = Subset(ImageBinaryDataset(data_root, paths=paths, labels=labels, transform=get_transforms(img_size, train=False)), val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset, batch_size=max(64, batch_size), shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    y_train = np.array(labels)[np.array(train_idx)]
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    pos_weight = None
    if pos > 0:
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    model = create_model(model_name=model_name, pretrained=True)
    freeze_backbone(model)
    model.to(device)
    opt1 = AdamW(head_parameters(model), lr=1e-3, weight_decay=1e-4)
    best_acc = -1.0
    best_state = None
    best_t = 0.5
    for _ in range(epochs_head):
        train_one_epoch(model, train_loader, opt1, device, pos_weight=pos_weight)
        probs, yval = eval_model(model, val_loader, device)
        t, acc = best_threshold(probs, yval)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_t = t
    for p in model.parameters():
        p.requires_grad = True
    head_names = set()
    for n, _ in model.named_parameters():
        if any(k in n for k in ['classifier', 'head', 'fc']):
            head_names.add(n)
    pg = []
    for n, p in model.named_parameters():
        if n in head_names:
            pg.append({'params': [p], 'lr': 5e-4})
        else:
            pg.append({'params': [p], 'lr': 1e-4})
    opt2 = AdamW(pg, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_ft)
    for _ in range(epochs_ft):
        train_one_epoch(model, train_loader, opt2, device, pos_weight=pos_weight)
        scheduler.step()
        probs, yval = eval_model(model, val_loader, device)
        t, acc = best_threshold(probs, yval)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_t = t
    weights_dir = Path(out_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    fold_path = weights_dir / f"best_{model_name}_fold{fold_idx}.pt"
    torch.save(best_state, fold_path)
    return best_acc, best_t, fold_path

def train_kfold(data_root: Path, out_dir: Path, model_name: str = 'efficientnetv2_s', img_size: int = 224, seed: int = 42, k: int = 5, epochs_head: int = 10, epochs_ft: int = 20, batch_size: int = 32, num_workers: int = 0):
    set_seed(seed)
    paths, labels = build_paths_and_labels(data_root)
    best_overall_acc = -1.0
    best_overall_t = 0.5
    best_overall_path = None
    for fold in range(k):
        acc, t, path = run_fold(paths, labels, fold, k, data_root, out_dir, model_name, img_size, seed, epochs_head=epochs_head, epochs_ft=epochs_ft, batch_size=batch_size, num_workers=num_workers)
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_t = t
            best_overall_path = path
    final_path = Path(out_dir) / 'best.pt'
    state = torch.load(best_overall_path, map_location='cpu')
    torch.save(state, final_path)
    meta = {
        'model_name': model_name,
        'threshold': best_overall_t,
    }
    with open(Path(out_dir) / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return final_path

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--out', type=str, default='./weights')
    ap.add_argument('--model', type=str, default='efficientnetv2_s')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs-head', type=int, default=10)
    ap.add_argument('--epochs-ft', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=0)
    a = ap.parse_args()
    train_kfold(Path(a.data), Path(a.out), model_name=a.model, img_size=a.img_size, seed=a.seed, k=a.folds, epochs_head=a.epochs_head, epochs_ft=a.epochs_ft, batch_size=a.batch_size, num_workers=a.num_workers)

if __name__ == '__main__':
    main()
