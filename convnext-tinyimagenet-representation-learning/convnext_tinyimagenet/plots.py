from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = denormalize_tensor(tensor.detach().cpu()).clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def plot_image_grid(
    images: torch.Tensor,
    labels: Sequence[int],
    class_names: Sequence[str],
    path: str,
    title: str = "Tiny ImageNet samples",
    max_images: int = 36,
) -> None:
    _ensure_parent(path)
    images = images[:max_images]
    labels = labels[:max_images]
    n = len(images)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx, ax in enumerate(axes[:n]):
        ax.imshow(tensor_to_image(images[idx]))
        label = int(labels[idx])
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    labels: Sequence[int],
    class_names: Sequence[str],
    path: str,
    title: str = "Tiny ImageNet class distribution",
) -> None:
    _ensure_parent(path)
    counts = np.bincount(np.asarray(labels), minlength=len(class_names))
    indices = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(indices, counts)
    ax.set_title(title)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Number of images")
    ax.set_xticks(indices[::10])
    ax.set_xticklabels([str(i) for i in indices[::10]])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: List[Dict[str, float]], path: str, task: str) -> None:
    _ensure_parent(path)
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    if "val_loss" in history[0]:
        ax1.plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    if task == "classification" and "val_top1" in history[0]:
        ax2 = ax1.twinx()
        ax2.plot(epochs, [row["train_top1"] for row in history], linestyle="--", label="Train top-1")
        ax2.plot(epochs, [row["val_top1"] for row in history], linestyle="--", label="Val top-1")
        ax2.plot(epochs, [row["val_top5"] for row in history], linestyle=":", label="Val top-5")
        ax2.set_ylabel("Accuracy (%)")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    else:
        ax1.legend(loc="upper right")

    fig.suptitle(f"Training curves ({task})")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion: torch.Tensor,
    class_names: Sequence[str],
    path: str,
    normalize: bool = True,
) -> None:
    _ensure_parent(path)
    cm = confusion.detach().cpu().numpy().astype(np.float64)

    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(24, 24))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title("Confusion matrix" + (" normalized by true class" if normalize else ""))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    tick_step = 10
    ticks = np.arange(len(class_names))[::tick_step]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([class_names[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([class_names[i] for i in ticks], fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(per_class: List[Dict[str, object]], path: str, top_n: Optional[int] = 60) -> None:
    _ensure_parent(path)
    rows = sorted(per_class, key=lambda x: float(x["accuracy"]))
    if top_n is not None:
        rows = rows[:top_n]

    names = [str(row["class_name"]) for row in rows]
    values = [float(row["accuracy"]) * 100 for row in rows]

    fig_height = max(8, len(rows) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(names, values)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Hardest classes by per-class accuracy")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_most_confused_pairs(
    confusion: torch.Tensor,
    class_names: Sequence[str],
    path: str,
    top_n: int = 30,
) -> None:
    _ensure_parent(path)
    cm = confusion.detach().cpu().numpy().copy()
    np.fill_diagonal(cm, 0)

    pairs = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            count = cm[true_idx, pred_idx]
            if count > 0:
                pairs.append((count, true_idx, pred_idx))
    pairs.sort(reverse=True)
    pairs = pairs[:top_n]

    labels = [f"{class_names[t]} → {class_names[p]}" for count, t, p in pairs]
    counts = [int(count) for count, t, p in pairs]

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.32)))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel("Number of mistakes")
    ax.set_title(f"Top {len(pairs)} most-confused class pairs")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_topk_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    class_names: Sequence[str],
    path: str,
    max_images: int = 16,
    top_k: int = 3,
) -> None:
    _ensure_parent(path)

    images = images[:max_images].detach().cpu()
    targets = targets[:max_images].detach().cpu()
    probs = logits.softmax(dim=1).detach().cpu()
    top_probs, top_indices = probs.topk(top_k, dim=1)

    n = len(images)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.8))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx, ax in enumerate(axes[:n]):
        ax.imshow(tensor_to_image(images[idx]))
        true_name = class_names[int(targets[idx])]
        lines = [f"True: {true_name}"]
        for rank in range(top_k):
            pred_idx = int(top_indices[idx, rank])
            prob = float(top_probs[idx, rank]) * 100
            lines.append(f"{rank + 1}. {class_names[pred_idx]}: {prob:.1f}%")
        ax.set_title("\n".join(lines), fontsize=8)
        ax.axis("off")

    fig.suptitle("Top-k predictions")
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
    title: str = "PCA visualization of ConvNeXt features",
) -> None:
    _ensure_parent(path)

    if len(features) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(features), size=max_points, replace=False)
        features = features[indices]
        labels = labels[indices]

    coords = pca_2d(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class index")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
