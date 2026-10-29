"""Plots for ROI inspection and CNN activation maps."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_rois(regions, output_path=None):
    fig, axes = plt.subplots(1, len(regions), figsize=(3 * len(regions), 3))
    axes = np.atleast_1d(axes)
    for ax, (name, crop) in zip(axes, regions.items()):
        ax.imshow(crop); ax.set_title(name); ax.axis("off")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def normalize_channel(channel):
    channel = np.asarray(channel, dtype=np.float32)
    lo, hi = np.percentile(channel, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(channel)
    return np.clip((channel - lo) / (hi - lo), 0, 1)


def plot_feature_maps(all_maps, model_names, region_name="Full Face", channels=8, output_path=None):
    fig, axes = plt.subplots(len(model_names), channels, figsize=(2.0 * channels, 2.4 * len(model_names)), squeeze=False)
    for row, model_name in enumerate(model_names):
        fmap = all_maps[(model_name, region_name)][0].numpy()
        selected = np.argsort(fmap.std(axis=(1, 2)))[-channels:][::-1]
        for col, idx in enumerate(selected):
            axes[row, col].imshow(normalize_channel(fmap[idx]), cmap="viridis", vmin=0, vmax=1)
            axes[row, col].set_title(f"Ch {idx}", fontsize=8)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(model_name, fontsize=9)
    fig.suptitle(f"{region_name}: channels selected by spatial standard deviation")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_progressive_stages(stages, title, channels=4, output_path=None):
    fig, axes = plt.subplots(len(stages), channels, figsize=(3 * channels, 2.3 * len(stages)), squeeze=False)
    for row, (stage_name, fmap) in enumerate(stages.items()):
        selected = np.argsort(fmap.std(axis=(1, 2)))[-channels:][::-1]
        for col, idx in enumerate(selected):
            axes[row, col].imshow(normalize_channel(fmap[idx]), cmap="viridis")
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{stage_name}\nCh {idx}" if col == 0 else f"Ch {idx}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
