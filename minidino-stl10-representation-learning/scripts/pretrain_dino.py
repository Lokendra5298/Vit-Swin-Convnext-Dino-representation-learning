import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minidino_stl10.data import STL10_CLASSES, get_ssl_loader
from minidino_stl10.losses import DINOLoss
from minidino_stl10.metrics import AverageMeter, save_history_csv
from minidino_stl10.model import create_dino_model
from minidino_stl10.plots import plot_pretraining_curves
from minidino_stl10.utils import (
    build_grad_scaler,
    copy_student_to_teacher,
    cosine_scheduler,
    count_parameters,
    get_device,
    get_params_groups,
    load_checkpoint,
    mkdir,
    save_checkpoint,
    save_json,
    seed_everything,
    set_optimizer_lr_wd,
    update_teacher,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-DINOv2-style self-supervised pretraining on STL-10 unlabeled")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs_dino")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-size", type=str, choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--num-register-tokens", type=int, default=4)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)

    parser.add_argument("--global-size", type=int, default=96)
    parser.add_argument("--local-size", type=int, default=48)
    parser.add_argument("--local-crops-number", type=int, default=6)

    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--weight-decay-end", type=float, default=0.4)
    parser.add_argument("--grad-clip", type=float, default=3.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--quick-debug", action="store_true")
    return parser.parse_args()


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type)
    return nullcontext()


def build_models(args, device):
    student = create_dino_model(
        model_size=args.model_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_register_tokens=args.num_register_tokens,
        drop_path_rate=args.drop_path_rate,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    teacher = create_dino_model(
        model_size=args.model_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_register_tokens=args.num_register_tokens,
        drop_path_rate=0.0,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    copy_student_to_teacher(student, teacher)
    return student, teacher


def train_one_epoch(
    student,
    teacher,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled: bool,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch: int,
    global_step_start: int,
    args,
):
    student.train()
    teacher.eval()
    loss_meter = AverageMeter()
    progress = tqdm(loader, desc=f"pretrain epoch {epoch}", leave=False)
    global_step = global_step_start
    num_student_crops = 2 + args.local_crops_number

    for crops, _ in progress:
        crops = [crop.to(device, non_blocking=True) for crop in crops]
        lr = float(lr_schedule[global_step])
        wd = float(wd_schedule[global_step])
        momentum = float(momentum_schedule[global_step])
        set_optimizer_lr_wd(optimizer, lr=lr, weight_decay=wd)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            student_output = torch.cat([student(crop) for crop in crops], dim=0)
            with torch.no_grad():
                teacher_output = torch.cat([teacher(crop) for crop in crops[:2]], dim=0)
            loss = criterion(student_output, teacher_output, num_student_crops=num_student_crops)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()

        update_teacher(student, teacher, momentum=momentum)
        batch_size = crops[0].size(0)
        loss_meter.update(loss.item(), batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{lr:.2e}", m=f"{momentum:.4f}")
        global_step += 1

    return {"loss": loss_meter.avg, "last_lr": lr, "last_wd": wd, "last_momentum": momentum}, global_step


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_dir = mkdir(output_dir / "checkpoints")
    log_dir = mkdir(output_dir / "logs")
    plot_dir = mkdir(output_dir / "plots")

    max_samples = 2048 if args.quick_debug else None
    batch_size = 16 if args.quick_debug else args.batch_size
    epochs = 2 if args.quick_debug else args.epochs

    loader = get_ssl_loader(
        data_dir=args.data_dir,
        global_size=args.global_size,
        local_size=args.local_size,
        local_crops_number=args.local_crops_number,
        batch_size=batch_size,
        num_workers=args.num_workers,
        max_samples=max_samples,
    )

    device = get_device()
    amp_enabled = args.amp and device.type == "cuda"
    student, teacher = build_models(args, device)

    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Student trainable parameters: {count_parameters(student):,}")

    optimizer = AdamW(get_params_groups(student), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device.type, amp_enabled)

    criterion = DINOLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        center_momentum=args.center_momentum,
        num_teacher_crops=2,
    ).to(device)

    steps_per_epoch = len(loader)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, epochs, steps_per_epoch, args.warmup_epochs, 0.0)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, epochs, steps_per_epoch)
    momentum_schedule = cosine_scheduler(args.teacher_momentum, 1.0, epochs, steps_per_epoch)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    history: List[Dict[str, float]] = []

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=str(device))
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        criterion.load_state_dict(checkpoint["criterion"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        history = checkpoint.get("history", [])
        best_loss = float(checkpoint.get("best_loss", best_loss))

    save_json(vars(args), str(log_dir / "args.json"))

    for epoch in range(start_epoch, epochs + 1):
        stats, global_step = train_one_epoch(
            student,
            teacher,
            loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            global_step,
            args,
        )

        row = {
            "epoch": epoch,
            "loss": stats["loss"],
            "lr": stats["last_lr"],
            "weight_decay": stats["last_wd"],
            "teacher_momentum": stats["last_momentum"],
        }
        history.append(row)
        print(
            f"epoch={epoch}, loss={row['loss']:.4f}, lr={row['lr']:.2e}, "
            f"wd={row['weight_decay']:.4f}, teacher_momentum={row['teacher_momentum']:.5f}"
        )

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
            "args": vars(args),
            "class_names": STL10_CLASSES,
            "history": history,
            "best_loss": best_loss,
        }
        save_checkpoint(state, str(checkpoint_dir / "last.pt"))
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            state["best_loss"] = best_loss
            save_checkpoint(state, str(checkpoint_dir / "best.pt"))
            print(f"Saved best checkpoint with loss={best_loss:.4f}")

        save_history_csv(history, str(log_dir / "history.csv"))
        save_json({"history": history}, str(log_dir / "history.json"))
        plot_pretraining_curves(history, str(plot_dir / "pretraining_curves.png"))

    print("\nPretraining complete.")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
