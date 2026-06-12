import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minidino_stl10.data import STL10_CLASSES, get_labeled_loaders
from minidino_stl10.metrics import (
    AverageMeter,
    compute_metrics_from_confusion,
    save_history_csv,
    save_per_class_csv,
    topk_accuracy,
    update_confusion_matrix,
)
from minidino_stl10.model import create_dino_model
from minidino_stl10.plots import plot_confusion_matrix, plot_training_curves
from minidino_stl10.utils import build_grad_scaler, get_device, load_checkpoint, mkdir, save_checkpoint, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-backbone linear probe for Mini-DINO STL-10 features")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_linear_probe")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-teacher", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--quick-debug", action="store_true")
    return parser.parse_args()


def build_backbone_from_checkpoint(checkpoint, use_teacher: bool):
    args = checkpoint.get("args", {})
    model = create_dino_model(
        model_size=args.get("model_size", "small"),
        image_size=int(args.get("image_size", 96)),
        patch_size=int(args.get("patch_size", 8)),
        num_register_tokens=int(args.get("num_register_tokens", 4)),
        drop_path_rate=0.0,
        out_dim=int(args.get("out_dim", 4096)),
        hidden_dim=int(args.get("hidden_dim", 2048)),
        bottleneck_dim=int(args.get("bottleneck_dim", 256)),
    )
    state_key = "teacher" if use_teacher and "teacher" in checkpoint else "student"
    model.load_state_dict(checkpoint[state_key])
    return model.backbone


def autocast_context(device, enabled):
    if enabled:
        return torch.autocast(device_type=device.type)
    return nullcontext()


def run_epoch(backbone, classifier, loader, criterion, optimizer, device, train, amp_enabled, scaler=None):
    backbone.eval()
    classifier.train(train)
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    confusion = torch.zeros(len(STL10_CLASSES), len(STL10_CLASSES), dtype=torch.long)

    desc = "train_probe" if train else "test_probe"
    progress = tqdm(loader, desc=desc, leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with autocast_context(device, amp_enabled):
                features = backbone.forward_features(images)
                features = torch.nn.functional.normalize(features, dim=1)

        with torch.set_grad_enabled(train):
            with autocast_context(device, amp_enabled):
                logits = classifier(features)
                loss = criterion(logits, targets)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
        preds = logits.argmax(dim=1)
        update_confusion_matrix(confusion, preds.cpu(), targets.cpu())

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
        "confusion": confusion,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = mkdir(output_dir / "checkpoints")
    log_dir = mkdir(output_dir / "logs")
    metric_dir = mkdir(output_dir / "metrics")
    plot_dir = mkdir(output_dir / "plots")

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    backbone = build_backbone_from_checkpoint(checkpoint, use_teacher=args.use_teacher).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    classifier = nn.Linear(backbone.embed_dim, len(STL10_CLASSES)).to(device)

    image_size = int(checkpoint.get("args", {}).get("image_size", 96))
    max_train = 512 if args.quick_debug else None
    max_test = 512 if args.quick_debug else None
    loaders = get_labeled_loaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_aug=True,
        max_train_samples=max_train,
        max_test_samples=max_test,
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = build_grad_scaler(device.type, amp_enabled)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    history: List[Dict[str, float]] = []
    best_top1 = 0.0
    epochs = 2 if args.quick_debug else args.epochs

    for epoch in range(1, epochs + 1):
        print(f"\nLinear probe epoch {epoch}/{epochs}")
        train_stats = run_epoch(backbone, classifier, loaders["train"], criterion, optimizer, device, True, amp_enabled, scaler)
        test_stats = run_epoch(backbone, classifier, loaders["test"], criterion, None, device, False, amp_enabled)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_top1": train_stats["top1"],
            "train_top5": train_stats["top5"],
            "test_loss": test_stats["loss"],
            "test_top1": test_stats["top1"],
            "test_top5": test_stats["top5"],
        }
        history.append(row)
        print(f"train_top1={row['train_top1']:.2f}, test_top1={row['test_top1']:.2f}, test_top5={row['test_top5']:.2f}")

        state = {
            "epoch": epoch,
            "classifier": classifier.state_dict(),
            "backbone_checkpoint": str(args.checkpoint),
            "args": vars(args),
            "history": history,
            "class_names": STL10_CLASSES,
        }
        save_checkpoint(state, str(checkpoint_dir / "last_probe.pt"))
        if row["test_top1"] > best_top1:
            best_top1 = row["test_top1"]
            save_checkpoint(state, str(checkpoint_dir / "best_probe.pt"))
            print(f"Saved best linear probe with test top-1={best_top1:.2f}%")

        save_history_csv(history, str(log_dir / "linear_probe_history.csv"))
        save_json({"history": history}, str(log_dir / "linear_probe_history.json"))
        plot_training_curves(history, str(plot_dir / "linear_probe_curves.png"), title="Linear probe curves")

    final_stats = run_epoch(backbone, classifier, loaders["test"], criterion, None, device, False, amp_enabled)
    cm_metrics = compute_metrics_from_confusion(final_stats["confusion"], STL10_CLASSES)
    summary = {
        "test_loss": final_stats["loss"],
        "test_top1_percent": final_stats["top1"],
        "test_top5_percent": final_stats["top5"],
        "accuracy_from_confusion": cm_metrics["accuracy"],
        "macro_f1": cm_metrics["macro_f1"],
        "weighted_f1": cm_metrics["weighted_f1"],
    }
    save_json(summary, str(metric_dir / "test_metrics.json"))
    save_per_class_csv(cm_metrics["per_class"], str(metric_dir / "per_class_metrics.csv"))
    plot_confusion_matrix(final_stats["confusion"], STL10_CLASSES, str(plot_dir / "confusion_matrix.png"))

    print(f"\nBest linear probe test top-1: {best_top1:.2f}%")
    print(f"Final metrics saved to: {metric_dir}")


if __name__ == "__main__":
    main()
