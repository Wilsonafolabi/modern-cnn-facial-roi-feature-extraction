"""Input preparation, activation extraction, and descriptive statistics."""
from __future__ import annotations

import time
import numpy as np
import torch
from torchvision import transforms

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_input(crop: np.ndarray, device: torch.device):
    return PREPROCESS(crop).unsqueeze(0).to(device)


def activation_stats(feature_map: torch.Tensor):
    arr = feature_map.detach().float().cpu().numpy()
    return {
        "feature_map_shape": list(arr.shape),
        "channels": int(arr.shape[1]),
        "height": int(arr.shape[2]),
        "width": int(arr.shape[3]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "zero_fraction": float(np.mean(arr == 0)),
        "finite_fraction": float(np.isfinite(arr).mean()),
        "l2_norm": float(np.linalg.norm(arr)),
        "numerically_valid": bool(np.isfinite(arr).all()),
    }


@torch.inference_mode()
def extract_feature_map(model_name, model, crop, device, forward_fn):
    tensor = prepare_input(crop, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    fmap = forward_fn(model_name, model, tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if fmap.ndim != 4 or not torch.isfinite(fmap).all():
        raise ValueError(f"Invalid activation output for {model_name}: shape={tuple(fmap.shape)}")
    fmap = fmap.detach().float().cpu()
    stats = activation_stats(fmap)
    stats["runtime_seconds"] = float(elapsed)
    return fmap, stats
