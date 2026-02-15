from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

def find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else find_last_conv(model)
        self.activations = None
        self.gradients = None
        self.hook_a = self.target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor):
        self.model.zero_grad()
        out = self.model(x)
        score = out.squeeze()
        score.backward(torch.ones_like(score))
        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0)
        cam = cam.cpu().numpy()
        cam = cam - cam.min() if cam.max() > cam.min() else cam
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam

def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    w, h = img.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0
    img_arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    heat = np.zeros_like(img_arr)
    heat[:, :, 0] = cam_arr
    out = (1 - alpha) * img_arr + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))

def cam_to_mask(cam: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    pos = cam[cam > 0]
    if pos.size == 0:
        return np.zeros_like(cam, dtype=np.uint8)
    t = np.percentile(pos, percentile)
    if t <= 1e-6:
        t = np.percentile(pos, 95.0)
    m = (cam >= t).astype(np.uint8)
    r = float(m.sum()) / float(m.size)
    if r > 0.3:
        t = np.percentile(pos, 95.0)
        m = (cam >= t).astype(np.uint8)
    elif r < 0.005:
        t = np.percentile(pos, 70.0)
        m = (cam >= t).astype(np.uint8)
    return m

def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y1 = int(ys.min())
    y2 = int(ys.max())
    x1 = int(xs.min())
    x2 = int(xs.max())
    return {
        'x': x1,
        'y': y1,
        'w': int(x2 - x1 + 1),
        'h': int(y2 - y1 + 1),
    }

def central_roi_mask(w: int, h: int, margin_ratio: float = 0.1):
    cx = w * 0.5
    cy = h * 0.5
    rx = w * (0.5 - margin_ratio)
    ry = h * (0.5 - margin_ratio)
    xs = np.arange(w)
    ys = np.arange(h)
    X, Y = np.meshgrid(xs, ys)
    val = ((X - cx) ** 2) / (rx ** 2) + ((Y - cy) ** 2) / (ry ** 2)
    m = (val <= 1.0).astype(np.uint8)
    return m

def intersection_ratio(mask: np.ndarray, roi: np.ndarray):
    if mask.size == 0:
        return 0.0
    inside = int(np.sum((mask > 0) & (roi > 0)))
    total = int(np.sum(mask > 0))
    if total == 0:
        return 0.0
    return float(inside) / float(total)

def bbox_touches_border(bbox: dict, w: int, h: int, border_ratio: float = 0.05):
    if bbox is None:
        return True
    x = bbox['x']
    y = bbox['y']
    x2 = x + bbox['w']
    y2 = y + bbox['h']
    bx = int(w * border_ratio)
    by = int(h * border_ratio)
    if x <= bx or y <= by or x2 >= (w - bx) or y2 >= (h - by):
        return True
    return False
