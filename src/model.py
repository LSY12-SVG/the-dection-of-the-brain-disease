import torch
import torch.nn as nn

def create_model(model_name: str = 'efficientnetv2_s', pretrained: bool = True):
    import timm
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    except Exception:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)
    return model

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def freeze_backbone(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if any(k in n for k in ['classifier', 'head', 'fc']):
            p.requires_grad = True

def head_parameters(model: nn.Module):
    params = []
    for n, p in model.named_parameters():
        if any(k in n for k in ['classifier', 'head', 'fc']):
            params.append(p)
    return params
