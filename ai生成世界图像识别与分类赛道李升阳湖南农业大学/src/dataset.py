from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
try:
    from .utils import list_images, infer_label_from_path
except ImportError:
    from utils import list_images, infer_label_from_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_paths_and_labels(root: Path) -> Tuple[List[Path], List[int]]:
    items = []
    for p in list_images(root):
        y = infer_label_from_path(p)
        if y is None:
            continue
        items.append((p, int(y)))
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    return paths, labels

def get_transforms(img_size: int = 224, train: bool = False):
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class ImageBinaryDataset(Dataset):
    def __init__(self, root: Path, paths: Optional[List[Path]] = None, labels: Optional[List[int]] = None, transform=None, return_filename: bool = False):
        self.root = Path(root)
        self.transform = transform if transform is not None else get_transforms(train=labels is not None)
        if paths is None:
            ps = list_images(self.root)
        else:
            ps = paths
        self.items = []
        if labels is None:
            for p in ps:
                self.items.append((p, None))
        else:
            for p, y in zip(ps, labels):
                self.items.append((p, int(y)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        if y is None:
            if self.return_filename:
                return x, p.name
            return x
        else:
            if self.return_filename:
                return x, int(y), p.name
            return x, int(y)
