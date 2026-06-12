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
from convnext_tinyimagenet.losses import SupConLoss
from convnext_tinyimagenet.metrics import AverageMeter, save_history_csv, topk_accuracy
from convnext_tinyimagenet.model import ProjectionHead, create_convnext
from convnext_tinyimagenet.plots import plot_image_grid, plot_training_curves
from convnext_tinyimagenet.utils import (
    build_grad_scaler,
    count_parameters,
    get_device,
    load_checkpoint,
    maybe_compile,
    mkdir,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt from scratch on Tiny ImageNet")
    parser.add_argument("--task", type=str, choices=["classification", "supcon"], default="classification")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--model-size", type=str, choices=["nano", "tiny", "small"], default="nano")
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--layer-scale-init-value", type=float, default=1e-6)
    parser.add_argument("--head-dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--rand-augment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--quick-debug", action="store_true")

    return parser.parse_args()


def build_model(args):
    return create_convnext(
        model_size=args.model_size,
        num_classes=200,
        drop_path_rate=args.drop_path_rate,
        layer_scale_init_value=args.layer_scale_init_value,
        head_dropout=args.head_dropout,
    )


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type)
    return nullcontext()


def train_classification_epoch(model, loader, criterion, optimizer, device, scaler, amp_enabled, grad_clip):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    progress = tqdm(loader, desc="train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{top1_meter.avg:.2f}")

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device, amp_enabled):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    progress = tqdm(loader, desc="val", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1.item(), batch_size)
        top5_meter.update(top5.item(), batch_size)

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


def train_supcon_epoch(model, projector, loader, criterion, optimizer, device, scaler, amp_enabled, grad_clip):
    model.train()
    projector.train()
    loss_meter = AverageMeter()

    progress = tqdm(loader, desc="train_supcon", leave=False)
    for views, targets in progress:
        view1, view2 = views
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images = torch.cat([view1, view2], dim=0)
        batch_size = targets.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            features = model.forward_features(images)
            projections = projector(features)
            projections = projections.view(2, batch_size, -1).permute(1, 0, 2)
            loss = criterion(projections, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()), grad_clip)
            optimizer.step()

        loss_meter.update(loss.item(), batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return {"loss": loss_meter.avg}


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = mkdir(output_dir / "checkpoints")
    log_dir = mkdir(output_dir / "logs")
    plot_dir = mkdir(output_dir / "plots")

    batch_size = 32 if args.quick_debug else args.batch_size

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        task=args.task,
        batch_size=batch_size,
        num_workers=args.num_workers,
        rand_augment=args.rand_augment,
        random_erasing=args.random_erasing,
    )
    class_names = loaders["class_names"]

    # Save a visual sample grid when possible.
    if args.task == "classification":
        sample_images, sample_labels = next(iter(loaders["train"]))
        plot_image_grid(
            sample_images,
            sample_labels.tolist(),
            class_names,
            str(plot_dir / "training_batch_samples.png"),
            title="Tiny ImageNet training batch",
        )

    device = get_device()
    amp_enabled = args.amp and device.type == "cuda"

    model = build_model(args).to(device)
    projector = None
    if args.task == "supcon":
        projector = ProjectionHead(model.feature_dim, hidden_dim=model.feature_dim, out_dim=args.projection_dim).to(device)

    model = maybe_compile(model, args.compile)

    if args.task == "classification":
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        criterion = SupConLoss(temperature=args.temperature)
        optimizer = AdamW(
            list(model.parameters()) + list(projector.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = build_grad_scaler(device.type, amp_enabled)

    start_epoch = 1
    history: List[Dict[str, float]] = []
    best_score = -1e9 if args.task == "classification" else 1e9

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=str(device))
        model.load_state_dict(checkpoint["model"])
        if projector is not None and "projector" in checkpoint:
            projector.load_state_dict(checkpoint["projector"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", [])
        best_score = float(checkpoint.get("best_score", best_score))

    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Task: {args.task}")
    print(f"Trainable model parameters: {count_parameters(model):,}")
    if projector is not None:
        print(f"Projection head parameters: {count_parameters(projector):,}")

    save_json(vars(args), str(log_dir / "args.json"))

    max_epochs = 2 if args.quick_debug else args.epochs

    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        if args.task == "classification":
            train_stats = train_classification_epoch(
                model, loaders["train"], criterion, optimizer, device, scaler, amp_enabled, args.grad_clip
            )
            val_stats = evaluate_classification(model, loaders["val"], criterion, device, amp_enabled)
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
            current_score = row["val_top1"]
            is_best = current_score > best_score
            if is_best:
                best_score = current_score

            print(
                f"train_loss={row['train_loss']:.4f}, train_top1={row['train_top1']:.2f}, "
                f"val_loss={row['val_loss']:.4f}, val_top1={row['val_top1']:.2f}, val_top5={row['val_top5']:.2f}"
            )

        else:
            train_stats = train_supcon_epoch(
                model, projector, loaders["train"], criterion, optimizer, device, scaler, amp_enabled, args.grad_clip
            )
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_stats["loss"],
            }
            current_score = row["train_loss"]
            is_best = current_score < best_score
            if is_best:
                best_score = current_score

            print(f"train_supcon_loss={row['train_loss']:.4f}")

        scheduler.step()
        history.append(row)

        state = {
            "epoch": epoch,
            "task": args.task,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "class_names": class_names,
            "history": history,
            "best_score": best_score,
        }
        if projector is not None:
            state["projector"] = projector.state_dict()

        save_checkpoint(state, str(checkpoint_dir / "last.pt"))
        if is_best:
            save_checkpoint(state, str(checkpoint_dir / "best.pt"))
            print(f"Saved best checkpoint. Best score: {best_score:.4f}")

        save_history_csv(history, str(log_dir / "history.csv"))
        save_json({"history": history}, str(log_dir / "history.json"))
        plot_training_curves(history, str(plot_dir / "training_curves.png"), task=args.task)

    print("\nTraining complete.")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
