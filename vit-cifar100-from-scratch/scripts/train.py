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

from vit_cifar100.data import get_dataloaders
from vit_cifar100.metrics import AverageMeter, save_history_csv, topk_accuracy
from vit_cifar100.model import VisionTransformer
from vit_cifar100.plots import plot_image_grid, plot_training_curves
from vit_cifar100.utils import build_grad_scaler, count_parameters, get_device, load_checkpoint, maybe_compile, mkdir, save_checkpoint, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer from scratch on CIFAR-100')

    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--embed-dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--mlp-ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--drop-path', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    parser.add_argument('--rand-augment', action='store_true')
    parser.add_argument('--random-erasing', type=float, default=0.0)
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision on CUDA')
    parser.add_argument('--compile', action='store_true', help='Try torch.compile for PyTorch 2.x')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--quick-debug', action='store_true', help='Use a tiny subset to verify the pipeline')

    return parser.parse_args()


def build_model(args) -> VisionTransformer:
    return VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=100,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        drop_path=args.drop_path,
    )


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type)
    return nullcontext()


def run_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device, scaler, train: bool, amp_enabled: bool, grad_clip: float) -> Dict[str, float]:
    model.train(train)

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    desc = 'train' if train else 'val'
    progress = tqdm(loader, desc=desc, leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast_context(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if train:
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

        progress.set_postfix(loss=f'{loss_meter.avg:.4f}', top1=f'{top1_meter.avg:.2f}', top5=f'{top5_meter.avg:.2f}')

    return {'loss': loss_meter.avg, 'top1': top1_meter.avg, 'top5': top5_meter.avg}


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = mkdir(output_dir / 'checkpoints')
    log_dir = mkdir(output_dir / 'logs')
    plot_dir = mkdir(output_dir / 'plots')

    max_train_samples = 4096 if args.quick_debug else None
    max_val_samples = 1024 if args.quick_debug else None
    max_test_samples = 1024 if args.quick_debug else None

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        rand_augment=args.rand_augment,
        random_erasing=args.random_erasing,
        seed=args.seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    class_names = loaders['class_names']

    sample_images, sample_labels = next(iter(loaders['train']))
    plot_image_grid(sample_images, sample_labels.tolist(), class_names, str(plot_dir / 'training_batch_samples.png'), title='Training batch samples')

    device = get_device()
    amp_enabled = args.amp and device.type == 'cuda'

    model = build_model(args).to(device)
    model = maybe_compile(model, args.compile)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = build_grad_scaler(device.type, amp_enabled)

    start_epoch = 1
    best_val_top1 = 0.0
    history: List[Dict[str, float]] = []

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=str(device))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_val_top1 = float(checkpoint.get('best_val_top1', 0.0))
        history = checkpoint.get('history', [])

    print(f'Device: {device}')
    print(f'AMP enabled: {amp_enabled}')
    print(f'Trainable parameters: {count_parameters(model):,}')

    save_json(vars(args), str(log_dir / 'args.json'))

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        train_stats = run_one_epoch(model=model, loader=loaders['train'], criterion=criterion, optimizer=optimizer, device=device, scaler=scaler, train=True, amp_enabled=amp_enabled, grad_clip=args.grad_clip)
        val_stats = run_one_epoch(model=model, loader=loaders['val'], criterion=criterion, optimizer=None, device=device, scaler=None, train=False, amp_enabled=amp_enabled, grad_clip=args.grad_clip)

        scheduler.step()

        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_stats['loss'],
            'train_top1': train_stats['top1'],
            'train_top5': train_stats['top5'],
            'val_loss': val_stats['loss'],
            'val_top1': val_stats['top1'],
            'val_top5': val_stats['top5'],
        }
        history.append(row)

        print(f"train loss={row['train_loss']:.4f}, train top1={row['train_top1']:.2f}, val loss={row['val_loss']:.4f}, val top1={row['val_top1']:.2f}, val top5={row['val_top5']:.2f}")

        checkpoint_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_top1': best_val_top1,
            'args': vars(args),
            'class_names': class_names,
            'history': history,
        }

        save_checkpoint(checkpoint_state, str(checkpoint_dir / 'last.pt'))

        if row['val_top1'] > best_val_top1:
            best_val_top1 = row['val_top1']
            checkpoint_state['best_val_top1'] = best_val_top1
            save_checkpoint(checkpoint_state, str(checkpoint_dir / 'best.pt'))
            print(f'Saved new best checkpoint with val top-1={best_val_top1:.2f}%')

        save_history_csv(history, str(log_dir / 'history.csv'))
        save_json({'history': history}, str(log_dir / 'history.json'))
        plot_training_curves(history, str(plot_dir / 'training_curves.png'))

    print('\nTraining complete.')
    print(f'Best validation top-1 accuracy: {best_val_top1:.2f}%')
    print(f'Checkpoints saved in: {checkpoint_dir}')


if __name__ == '__main__':
    main()
