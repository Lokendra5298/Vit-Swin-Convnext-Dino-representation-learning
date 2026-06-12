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

from swin_cifar100_rep.data import get_dataloaders
from swin_cifar100_rep.losses import SupConLoss
from swin_cifar100_rep.metrics import AverageMeter, save_history_csv, topk_accuracy
from swin_cifar100_rep.model import SwinTransformer
from swin_cifar100_rep.plots import plot_image_grid, plot_training_curves
from swin_cifar100_rep.utils import build_grad_scaler, count_parameters, get_device, load_checkpoint, mkdir, save_checkpoint, save_json, seed_everything


def parse_tuple(text: str):
    return tuple(int(x.strip()) for x in text.split(',') if x.strip())


def parse_args():
    p = argparse.ArgumentParser(description='Train Swin Transformer from scratch on CIFAR-100')
    p.add_argument('--task', choices=['classification', 'supcon'], default='classification')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--output-dir', type=str, default='outputs')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--patch-size', type=int, default=2)
    p.add_argument('--window-size', type=int, default=4)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--depths', type=str, default='2,2,6,2')
    p.add_argument('--num-heads', type=str, default='2,4,8,16')
    p.add_argument('--mlp-ratio', type=float, default=4.0)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--attn-dropout', type=float, default=0.0)
    p.add_argument('--drop-path', type=float, default=0.1)
    p.add_argument('--projection-dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--quick-debug', action='store_true')
    return p.parse_args()


def build_model(args):
    return SwinTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=100,
        embed_dim=args.embed_dim,
        depths=parse_tuple(args.depths),
        num_heads=parse_tuple(args.num_heads),
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path=args.drop_path,
        projection_dim=args.projection_dim,
    )


def autocast_context(device, enabled):
    return torch.autocast(device_type=device.type) if enabled else nullcontext()


def train_classification(model, loader, criterion, optimizer, device, scaler, amp_enabled, grad_clip):
    model.train()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for images, targets in tqdm(loader, desc='train', leave=False):
        images, targets = images.to(device), targets.to(device)
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
        top1, top5 = topk_accuracy(logits, targets)
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1.item(), n)
        top5_meter.update(top5.item(), n)
    return {'loss': loss_meter.avg, 'top1': top1_meter.avg, 'top5': top5_meter.avg}


def train_supcon(model, loader, criterion, optimizer, device, scaler, amp_enabled, grad_clip):
    model.train()
    loss_meter = AverageMeter()
    for views, targets in tqdm(loader, desc='train_supcon', leave=False):
        images1, images2 = views
        images = torch.cat([images1, images2], dim=0).to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            z = model.forward_projection(images)
            z1, z2 = torch.split(z, [images1.size(0), images2.size(0)], dim=0)
            features = torch.stack([z1, z2], dim=1)
            loss = criterion(features, targets)
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
        loss_meter.update(loss.item(), images1.size(0))
    return {'loss': loss_meter.avg}


@torch.no_grad()
def validate_classification(model, loader, criterion, device, amp_enabled):
    model.eval()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for images, targets in tqdm(loader, desc='val', leave=False):
        images, targets = images.to(device), targets.to(device)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        top1, top5 = topk_accuracy(logits, targets)
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1.item(), n)
        top5_meter.update(top5.item(), n)
    return {'loss': loss_meter.avg, 'top1': top1_meter.avg, 'top5': top5_meter.avg}


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_dir = mkdir(output_dir / 'checkpoints')
    log_dir = mkdir(output_dir / 'logs')
    plot_dir = mkdir(output_dir / 'plots')

    max_train = 4096 if args.quick_debug else None
    max_val = 1024 if args.quick_debug else None
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        task=args.task,
        seed=args.seed,
        max_train_samples=max_train,
        max_val_samples=max_val,
    )
    class_names = loaders['class_names']

    # Make a grid with validation images because SupCon training batches contain two views.
    sample_images, sample_labels = next(iter(loaders['val']))
    plot_image_grid(sample_images, sample_labels.tolist(), class_names, plot_dir / 'sample_batch.png')

    device = get_device()
    amp_enabled = args.amp and device.type == 'cuda'
    model = build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = build_grad_scaler(device.type, amp_enabled)

    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = SupConLoss(temperature=args.temperature)

    start_epoch = 1
    best_score = float('inf') if args.task == 'supcon' else 0.0
    history: List[Dict[str, float]] = []

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=str(device))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = int(ckpt['epoch']) + 1
        best_score = ckpt.get('best_score', best_score)
        history = ckpt.get('history', [])

    print(f'Device: {device}')
    print(f'Task: {args.task}')
    print(f'AMP enabled: {amp_enabled}')
    print(f'Trainable parameters: {count_parameters(model):,}')
    save_json(vars(args), log_dir / 'args.json')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        if args.task == 'classification':
            train_stats = train_classification(model, loaders['train'], criterion, optimizer, device, scaler, amp_enabled, args.grad_clip)
            val_stats = validate_classification(model, loaders['val'], criterion, device, amp_enabled)
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
            improved = row['val_top1'] > best_score
            if improved:
                best_score = row['val_top1']
            print(f"train_loss={row['train_loss']:.4f}, val_top1={row['val_top1']:.2f}, val_top5={row['val_top5']:.2f}")
        else:
            train_stats = train_supcon(model, loaders['train'], criterion, optimizer, device, scaler, amp_enabled, args.grad_clip)
            row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'train_loss': train_stats['loss']}
            improved = row['train_loss'] < best_score
            if improved:
                best_score = row['train_loss']
            print(f"supcon_loss={row['train_loss']:.4f}")

        scheduler.step()
        history.append(row)
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_score': best_score,
            'args': vars(args),
            'class_names': class_names,
            'history': history,
        }
        save_checkpoint(state, checkpoint_dir / 'last.pt')
        if improved:
            save_checkpoint(state, checkpoint_dir / 'best.pt')
            print('Saved new best checkpoint.')
        save_history_csv(history, log_dir / 'history.csv')
        save_json({'history': history}, log_dir / 'history.json')
        plot_training_curves(history, plot_dir / 'training_curves.png')

    print('Training complete.')


if __name__ == '__main__':
    main()
