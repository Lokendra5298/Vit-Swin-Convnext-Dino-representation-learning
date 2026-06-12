from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CIFAR100_MEAN, CIFAR100_STD


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    return tensor.cpu() * std + mean


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return denormalize_tensor(tensor).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_image_grid(images: torch.Tensor, labels: Sequence[int], class_names: Sequence[str], path, title='CIFAR-100 samples', max_images=36):
    _ensure_parent(path)
    images = images[:max_images]
    labels = labels[:max_images]
    n = len(images)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for idx, ax in enumerate(axes[:n]):
        ax.imshow(tensor_to_image(images[idx]))
        ax.set_title(class_names[int(labels[idx])], fontsize=8)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_class_distribution(targets: Sequence[int], class_names: Sequence[str], path):
    _ensure_parent(path)
    counts = np.bincount(np.asarray(targets), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(np.arange(len(class_names)), counts)
    ax.set_title('CIFAR-100 class distribution')
    ax.set_xlabel('Class index')
    ax.set_ylabel('Images')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history: List[Dict[str, float]], path):
    _ensure_parent(path)
    if not history:
        return
    epochs = [row['epoch'] for row in history]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, [row['train_loss'] for row in history], label='Train loss')
    if 'val_loss' in history[0]:
        ax1.plot(epochs, [row['val_loss'] for row in history], label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    if 'train_top1' in history[0]:
        ax2.plot(epochs, [row['train_top1'] for row in history], linestyle='--', label='Train top-1')
        ax2.plot(epochs, [row['val_top1'] for row in history], linestyle='--', label='Val top-1')
        ax2.plot(epochs, [row['val_top5'] for row in history], linestyle=':', label='Val top-5')
        ax2.set_ylabel('Accuracy (%)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    fig.suptitle('Training curves')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(confusion: torch.Tensor, class_names: Sequence[str], path, normalize=True):
    _ensure_parent(path)
    cm = confusion.cpu().numpy().astype(np.float64)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title('Normalized confusion matrix' if normalize else 'Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ticks = np.arange(len(class_names))[::5]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([class_names[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([class_names[i] for i in ticks], fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_accuracy(rows: List[Dict[str, object]], path, bottom_n=None):
    _ensure_parent(path)
    rows = sorted(rows, key=lambda x: float(x['accuracy']))
    if bottom_n is not None:
        rows = rows[:bottom_n]
    names = [row['class_name'] for row in rows]
    values = [float(row['accuracy']) * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(8, len(rows) * 0.22)))
    ax.barh(names, values)
    ax.set_title('Per-class accuracy')
    ax.set_xlabel('Accuracy (%)')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_pca_embeddings(features: np.ndarray, labels: np.ndarray, class_names: Sequence[str], path, max_points=3000):
    _ensure_parent(path)
    if len(features) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(features), size=max_points, replace=False)
        features = features[idx]
        labels = labels[idx]
    x = features.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.65)
    ax.set_title('PCA visualization of Swin image embeddings')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Class index')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
