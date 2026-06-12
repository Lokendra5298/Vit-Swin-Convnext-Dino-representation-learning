from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import STL10_MEAN, STL10_STD


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(STL10_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(STL10_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = denormalize_tensor(tensor.detach().cpu()).clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def plot_image_grid(
    images: torch.Tensor,
    labels: Optional[Sequence[int]],
    class_names: Sequence[str],
    path: str,
    title: str = "STL-10 samples",
    max_images: int = 36,
) -> None:
    _ensure_parent(path)
    images = images[:max_images]
    n = len(images)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx, ax in enumerate(axes[:n]):
        ax.imshow(tensor_to_image(images[idx]))
        if labels is not None:
            label = int(labels[idx])
            if 0 <= label < len(class_names):
                ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_multicrop_grid(crops: List[torch.Tensor], path: str, title: str = "DINO multi-crop views") -> None:
    _ensure_parent(path)
    n = len(crops)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx, ax in enumerate(axes[:n]):
        ax.imshow(tensor_to_image(crops[idx]))
        kind = "global" if idx < 2 else "local"
        ax.set_title(f"{kind} crop {idx + 1}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pretraining_curves(history: List[Dict[str, float]], path: str) -> None:
    _ensure_parent(path)
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, [row["loss"] for row in history], label="DINO loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["lr"] for row in history], linestyle="--", label="Learning rate")
    ax2.set_ylabel("Learning rate")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    fig.suptitle("Mini-DINO pretraining curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: List[Dict[str, float]], path: str, title: str = "Training curves") -> None:
    _ensure_parent(path)
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    ax1.plot(epochs, [row["test_loss"] for row in history], label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["train_top1"] for row in history], linestyle="--", label="Train top-1")
    ax2.plot(epochs, [row["test_top1"] for row in history], linestyle="--", label="Test top-1")
    ax2.plot(epochs, [row["test_top5"] for row in history], linestyle=":", label="Test top-5")
    ax2.set_ylabel("Accuracy (%)")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(confusion: torch.Tensor, class_names: Sequence[str], path: str, normalize: bool = True) -> None:
    _ensure_parent(path)
    cm = confusion.detach().cpu().numpy().astype(np.float64)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pca_2d(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    features = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(features, full_matrices=False)
    return features @ vt[:2].T


def plot_pca_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    path: str,
    max_points: int = 5000,
    title: str = "PCA visualization of learned features",
) -> None:
    _ensure_parent(path)
    mask = labels >= 0
    features = features[mask]
    labels = labels[mask]
    if len(features) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(features), size=max_points, replace=False)
        features = features[indices]
        labels = labels[indices]
    coords = pca_2d(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=12, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(class_names)))
    cbar.ax.set_yticklabels(class_names)
    cbar.set_label("Class")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
