"""Cosine similarity utilities for ROI feature maps."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def cosine_similarity_matrix(vectors):
    """Compute cosine similarities between vectors; accepts 1D arrays or tensors."""
    rows = []
    for value in vectors:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        rows.append(np.asarray(value).reshape(-1))
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError("All vectors must have equal lengths to form a similarity matrix.")
    return cosine_similarity(np.stack(rows))


def plot_similarity(matrix, labels, title, output_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title); fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
