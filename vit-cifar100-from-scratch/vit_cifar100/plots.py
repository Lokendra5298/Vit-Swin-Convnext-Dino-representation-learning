from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CIFAR100_MEAN, CIFAR100_STD


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def denormalize_tensor(tensor: torch.Tensor, mean: Tuple[float, float, float] = CIFAR100_MEAN, std: Tuple[float, float, float] = CIFAR100_STD) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    return tensor * std_t + mean_t


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = denormalize_tensor(tensor.detach().cpu()).clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def plot_image_grid(images: torch.Tensor, labels: Sequence[int], class_names: Sequence[str], path: str, title: str = 'CIFAR-100 sample images', max_images: int = 36) -> None:
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
        label = int(labels[idx])
        ax.set_title(class_names[label], fontsize=8)
        ax.axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_class_distribution(targets: Sequence[int], class_names: Sequence[str], path: str, title: str = 'CIFAR-100 class distribution') -> None:
    _ensure_parent(path)
    counts = np.bincount(np.asarray(targets), minlength=len(class_names))
    indices = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(indices, counts)
    ax.set_title(title)
    ax.set_xlabel('Class index')
    ax.set_ylabel('Number of images')
    ax.set_xticks(indices[::5])
    ax.set_xticklabels([str(i) for i in indices[::5]], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_augmentation_grid(original_image, transform, path: str, title: str = 'Random augmentations of one CIFAR-100 image', n: int = 12) -> None:
    _ensure_parent(path)
    cols = 6
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for idx in range(n):
        augmented = transform(original_image)
        axes[idx].imshow(tensor_to_image(augmented))
        axes[idx].set_title(f'Aug {idx + 1}', fontsize=8)
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history: List[Dict[str, float]], path: str) -> None:
    _ensure_parent(path)
    if not history:
        return

    epochs = [row['epoch'] for row in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, [row['train_loss'] for row in history], label='Train loss')
    ax1.plot(epochs, [row['val_loss'] for row in history], label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, [row['train_top1'] for row in history], linestyle='--', label='Train top-1')
    ax2.plot(epochs, [row['val_top1'] for row in history], linestyle='--', label='Val top-1')
    ax2.plot(epochs, [row['val_top5'] for row in history], linestyle=':', label='Val top-5')
    ax2.set_ylabel('Accuracy (%)')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    fig.suptitle('Training curves')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(confusion: torch.Tensor, class_names: Sequence[str], path: str, normalize: bool = True) -> None:
    _ensure_parent(path)
    cm = confusion.detach().cpu().numpy().astype(np.float64)

    if normalize:
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm = cm / row_sum

    size = max(12, min(32, len(class_names) * 0.28))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title('Confusion matrix' + (' normalized by true class' if normalize else ''))
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    tick_step = 5 if len(class_names) > 40 else 1
    ticks = np.arange(len(class_names))[::tick_step]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([class_names[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([class_names[i] for i in ticks], fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_accuracy(per_class: List[Dict[str, object]], path: str, top_n: Optional[int] = None) -> None:
    _ensure_parent(path)
    rows = sorted(per_class, key=lambda x: float(x['accuracy']))
    if top_n is not None:
        rows = rows[:top_n]

    names = [str(row['class_name']) for row in rows]
    values = [float(row['accuracy']) * 100.0 for row in rows]

    fig_height = max(8, len(rows) * 0.20)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(names, values)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Per-class accuracy sorted from hardest to easiest')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_most_confused_pairs(confusion: torch.Tensor, class_names: Sequence[str], path: str, top_n: int = 25) -> None:
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

    labels = [f'{class_names[t]} -> {class_names[p]}' for count, t, p in pairs]
    counts = [int(count) for count, t, p in pairs]

    fig_height = max(6, top_n * 0.32)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel('Number of mistakes')
    ax.set_title(f'Top {len(pairs)} most-confused class pairs')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_topk_predictions(images: torch.Tensor, targets: torch.Tensor, logits: torch.Tensor, class_names: Sequence[str], path: str, max_images: int = 16, top_k: int = 3) -> None:
    _ensure_parent(path)

    images = images[:max_images].detach().cpu()
    targets = targets[:max_images].detach().cpu()
    probs = logits.softmax(dim=1).detach().cpu()
    top_probs, top_indices = probs.topk(top_k, dim=1)

    n = len(images)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for idx, ax in enumerate(axes[:n]):
        ax.imshow(tensor_to_image(images[idx]))
        true_name = class_names[int(targets[idx])]
        lines = [f'True: {true_name}']
        for rank in range(top_k):
            pred_idx = int(top_indices[idx, rank])
            pred_name = class_names[pred_idx]
            prob = float(top_probs[idx, rank]) * 100
            lines.append(f'{rank + 1}. {pred_name}: {prob:.1f}%')
        ax.set_title('\n'.join(lines), fontsize=8)
        ax.axis('off')

    fig.suptitle('Top-k predictions')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
