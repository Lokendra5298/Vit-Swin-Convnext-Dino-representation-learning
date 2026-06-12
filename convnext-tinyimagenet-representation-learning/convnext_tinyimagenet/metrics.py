import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)):
    max_k = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def update_confusion_matrix(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
    num_classes = confusion.shape[0]
    with torch.no_grad():
        indices = targets * num_classes + preds
        counts = torch.bincount(indices, minlength=num_classes * num_classes)
        confusion += counts.reshape(num_classes, num_classes).to(confusion.device)
    return confusion


def compute_metrics_from_confusion(confusion: torch.Tensor, class_names: Sequence[str]) -> Dict[str, object]:
    cm = confusion.cpu().numpy().astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)
    eps = 1e-12

    precision = correct / np.maximum(predicted, eps)
    recall = correct / np.maximum(support, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)

    total = cm.sum()
    accuracy = correct.sum() / max(total, eps)

    weights = support / max(support.sum(), eps)

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_index": idx,
                "class_name": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "accuracy": float(recall[idx]),
                "support": int(support[idx]),
            }
        )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "per_class": per_class,
    }


def save_history_csv(history: List[Dict[str, float]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_per_class_csv(per_class: List[Dict[str, object]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not per_class:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)
