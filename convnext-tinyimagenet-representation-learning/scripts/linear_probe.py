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

from convnext_tinyimagenet.data import build_dataloaders
from convnext_tinyimagenet.metrics import AverageMeter, save_history_csv, topk_accuracy
from convnext_tinyimagenet.model import create_convnext
from convnext_tinyimagenet.plots import plot_training_curves
from convnext_tinyimagenet.utils import build_grad_scaler, get_device, load_checkpoint, mkdir, save_checkpoint, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train a frozen-feature linear probe")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_linear_probe")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
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

    desc = "train_probe" if train else "val_probe"
    progress = tqdm(loader, desc=desc, leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with autocast_context(device, amp_enabled):
                features = backbone.forward_features(images)

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
        batch_size = images.size(0)

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = mkdir(output_dir / "checkpoints")
    log_dir = mkdir(output_dir / "logs")
    plot_dir = mkdir(output_dir / "plots")

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    class_names = checkpoint["class_names"]

    backbone = build_model_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    classifier = nn.Linear(backbone.feature_dim, len(class_names)).to(device)

    image_size = int(checkpoint.get("args", {}).get("image_size", 64))
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rand_augment=False,
        random_erasing=0.0,
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = build_grad_scaler(device.type, amp_enabled)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    history: List[Dict[str, float]] = []
    best_val_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nLinear probe epoch {epoch}/{args.epochs}")

        train_stats = run_epoch(
            backbone, classifier, loaders["train"], criterion, optimizer, device, True, amp_enabled, scaler
        )
        val_stats = run_epoch(
            backbone, classifier, loaders["val"], criterion, None, device, False, amp_enabled
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_top1": train_stats["top1"],
            "train_top5": train_stats["top5"],
            "val_loss": val_stats["loss"],
            "val_top1": val_stats["top1"],
            "val_top5": val_stats["top5"],
        }
        history.append(row)

        print(
            f"train_top1={row['train_top1']:.2f}, val_top1={row['val_top1']:.2f}, val_top5={row['val_top5']:.2f}"
        )

        state = {
            "epoch": epoch,
            "backbone_checkpoint": str(args.checkpoint),
            "classifier": classifier.state_dict(),
            "history": history,
            "class_names": class_names,
            "args": vars(args),
        }
        save_checkpoint(state, str(checkpoint_dir / "last_probe.pt"))
        if row["val_top1"] > best_val_top1:
            best_val_top1 = row["val_top1"]
            save_checkpoint(state, str(checkpoint_dir / "best_probe.pt"))
            print(f"Saved best linear probe: {best_val_top1:.2f}%")

        save_history_csv(history, str(log_dir / "linear_probe_history.csv"))
        save_json({"history": history}, str(log_dir / "linear_probe_history.json"))
        plot_training_curves(history, str(plot_dir / "linear_probe_curves.png"), task="classification")

    print(f"Best linear probe val top-1: {best_val_top1:.2f}%")


if __name__ == "__main__":
    main()
