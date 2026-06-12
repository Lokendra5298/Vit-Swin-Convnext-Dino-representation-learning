import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from convnext_tinyimagenet.data import build_dataloaders
from convnext_tinyimagenet.metrics import (
    compute_metrics_from_confusion,
    save_per_class_csv,
    topk_accuracy,
    update_confusion_matrix,
)
from convnext_tinyimagenet.model import create_convnext
from convnext_tinyimagenet.plots import (
    plot_confusion_matrix,
    plot_most_confused_pairs,
    plot_per_class_accuracy,
    plot_topk_predictions,
)
from convnext_tinyimagenet.utils import get_device, load_checkpoint, mkdir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt classification checkpoint on Tiny ImageNet val set")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint):
    args = checkpoint.get("args", {})
    return create_convnext(
        model_size=args.get("model_size", "nano"),
        num_classes=200,
        drop_path_rate=float(args.get("drop_path_rate", 0.0)),
        layer_scale_init_value=float(args.get("layer_scale_init_value", 1e-6)),
        head_dropout=float(args.get("head_dropout", 0.0)),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    top1_sum = 0.0
    top5_sum = 0.0
    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    saved_batch = None

    for images, targets in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
        preds = logits.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        top1_sum += top1.item() * batch_size
        top5_sum += top5.item() * batch_size
        total_samples += batch_size

        update_confusion_matrix(confusion, preds.cpu(), targets.cpu())

        if saved_batch is None:
            saved_batch = (images.detach().cpu(), targets.detach().cpu(), logits.detach().cpu())

    return {
        "loss": total_loss / max(1, total_samples),
        "top1": top1_sum / max(1, total_samples),
        "top5": top5_sum / max(1, total_samples),
        "confusion": confusion,
        "saved_batch": saved_batch,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    metric_dir = mkdir(output_dir / "metrics")
    plot_dir = mkdir(output_dir / "plots")

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    if checkpoint.get("task") == "supcon":
        raise ValueError("This is a SupCon checkpoint. Use scripts/linear_probe.py for evaluation.")

    class_names = checkpoint["class_names"]
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_size = int(checkpoint.get("args", {}).get("image_size", 64))
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    amp_enabled = args.amp and device.type == "cuda"
    criterion = nn.CrossEntropyLoss()
    stats = evaluate(model, loaders["val"], criterion, device, class_names, amp_enabled)
    cm_metrics = compute_metrics_from_confusion(stats["confusion"], class_names)

    summary = {
        "checkpoint": str(args.checkpoint),
        "val_loss": stats["loss"],
        "val_top1_percent": stats["top1"],
        "val_top5_percent": stats["top5"],
        "accuracy_from_confusion": cm_metrics["accuracy"],
        "macro_precision": cm_metrics["macro_precision"],
        "macro_recall": cm_metrics["macro_recall"],
        "macro_f1": cm_metrics["macro_f1"],
        "weighted_precision": cm_metrics["weighted_precision"],
        "weighted_recall": cm_metrics["weighted_recall"],
        "weighted_f1": cm_metrics["weighted_f1"],
    }

    save_json(summary, str(metric_dir / "val_metrics.json"))
    save_per_class_csv(cm_metrics["per_class"], str(metric_dir / "per_class_metrics.csv"))

    plot_confusion_matrix(stats["confusion"], class_names, str(plot_dir / "confusion_matrix.png"), normalize=True)
    plot_per_class_accuracy(cm_metrics["per_class"], str(plot_dir / "per_class_accuracy.png"), top_n=80)
    plot_most_confused_pairs(stats["confusion"], class_names, str(plot_dir / "most_confused_pairs.png"), top_n=30)

    if stats["saved_batch"] is not None:
        images, targets, logits = stats["saved_batch"]
        plot_topk_predictions(images, targets, logits, class_names, str(plot_dir / "topk_predictions.png"))

    print("Evaluation complete.")
    print(f"Validation top-1: {summary['val_top1_percent']:.2f}%")
    print(f"Validation top-5: {summary['val_top5_percent']:.2f}%")
    print(f"Metrics saved to: {metric_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
