"""Model construction and loading for the three notebook configurations."""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models as tv_models


class CustomCNN(nn.Module):
    """Small randomly initialized CNN; intentionally untrained baseline."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


def load_models(device: torch.device | None = None, checkpoint_path: str | Path | None = None):
    """Load Custom CNN, ImageNet ResNet-50, and the specified VGGFace2 checkpoint.

    The VGGFace2 checkpoint is downloaded only if no local checkpoint is provided.
    """
    import gdown

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    custom = CustomCNN().to(device).eval()

    imagenet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
    imagenet.fc = nn.Identity()
    imagenet = imagenet.to(device).eval()

    repo_dir = Path(os.environ.get("VGGFACE2_REPO_DIR", Path.cwd() / "third_party" / "VGGFace2-pytorch"))
    checkpoint = Path(checkpoint_path or os.environ.get("VGGFACE2_CHECKPOINT", Path.cwd() / "models" / "resnet50_ft_weight.pkl"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if not repo_dir.is_dir():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/cydonia999/VGGFace2-pytorch.git", str(repo_dir)
        ], check=True)

    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        gdown.download(id="1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU", output=str(checkpoint), quiet=False)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("VGGFace2 checkpoint download failed; do not substitute another checkpoint.")

    sys.path.insert(0, str(repo_dir))
    from models.resnet import resnet50 as vggface_resnet50  # supplied by the checkpoint repository

    with checkpoint.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    if isinstance(data, dict):
        if "state_dict" in data:
            data = data["state_dict"]
        elif "model" in data:
            data = data["model"]
    state = {}
    for key, value in data.items():
        key = key.replace("module.", "")
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if torch.is_tensor(value):
            state[key] = value
    vggface = vggface_resnet50(num_classes=8631, include_top=True)
    vggface.load_state_dict(state, strict=True)
    vggface = vggface.to(device).eval()

    return {
        "Custom CNN": custom,
        "ResNet-50 ImageNet": imagenet,
        "ResNet-50 VGGFace2": vggface,
    }


def forward_feature_map(model_name: str, model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Return the convolutional feature map captured in the original notebook."""
    if model_name == "Custom CNN":
        return model(tensor)
    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    return model.layer4(x)


def extract_progressive_stages(model_name: str, model: nn.Module, tensor: torch.Tensor):
    """Capture intermediate stages for visual inspection, matching notebook cells."""
    stages = {}
    with torch.inference_mode():
        if model_name == "Custom CNN":
            x = model.features[0](tensor); x = model.features[1](x); stages["Conv1 + ReLU"] = x[0].cpu().numpy()
            x = model.features[2](x); x = model.features[3](x); x = model.features[4](x); stages["Conv2 + ReLU"] = x[0].cpu().numpy()
            x = model.features[5](x); x = model.features[6](x); x = model.features[7](x); stages["Conv3 + ReLU"] = x[0].cpu().numpy()
        else:
            x = model.conv1(tensor); x = model.bn1(x); x = model.relu(x); stages["conv1"] = x[0].cpu().numpy()
            x = model.maxpool(x); x = model.layer1(x); stages["layer1"] = x[0].cpu().numpy()
            x = model.layer2(x); stages["layer2"] = x[0].cpu().numpy()
            x = model.layer3(x); stages["layer3"] = x[0].cpu().numpy()
            x = model.layer4(x); stages["layer4"] = x[0].cpu().numpy()
    return stages
