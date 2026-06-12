import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
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
    out = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out.append(correct_k.mul_(100.0 / batch_size))
    return out


def update_confusion_matrix(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
    num_classes = confusion.shape[0]
    idx = targets * num_classes + preds
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).to(confusion.device)
    return confusion


def per_class_accuracy_from_confusion(confusion: torch.Tensor, class_names: Sequence[str]):
    cm = confusion.cpu().numpy().astype(np.float64)
    correct = np.diag(cm)
    totals = np.maximum(cm.sum(axis=1), 1)
    acc = correct / totals
    rows = []
    for i, name in enumerate(class_names):
        rows.append({'class_index': i, 'class_name': name, 'accuracy': float(acc[i]), 'support': int(totals[i])})
    return rows


def save_history_csv(history: List[Dict[str, float]], path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_rows_csv(rows: List[Dict[str, object]], path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
