import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from swin_cifar100_rep.data import get_dataloaders
from swin_cifar100_rep.metrics import AverageMeter, save_history_csv, topk_accuracy
from swin_cifar100_rep.model import SwinTransformer
from swin_cifar100_rep.plots import plot_training_curves
from swin_cifar100_rep.utils import get_device, load_checkpoint, mkdir, save_checkpoint, save_json, seed_everything


def parse_tuple(text: str):
    return tuple(int(x.strip()) for x in text.split(',') if x.strip())


def build_model_from_args(args_dict):
    return SwinTransformer(
        image_size=int(args_dict.get('image_size', 32)),
        patch_size=int(args_dict.get('patch_size', 2)),
        num_classes=100,
        embed_dim=int(args_dict.get('embed_dim', 64)),
        depths=parse_tuple(str(args_dict.get('depths', '2,2,6,2'))),
        num_heads=parse_tuple(str(args_dict.get('num_heads', '2,4,8,16'))),
        window_size=int(args_dict.get('window_size', 4)),
        mlp_ratio=float(args_dict.get('mlp_ratio', 4.0)),
        dropout=float(args_dict.get('dropout', 0.0)),
        attn_dropout=float(args_dict.get('attn_dropout', 0.0)),
        drop_path=float(args_dict.get('drop_path', 0.0)),
        projection_dim=int(args_dict.get('projection_dim', 128)),
    )


def parse_args():
    p = argparse.ArgumentParser(description='Train linear probe on frozen Swin features')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='outputs')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def validate(backbone, head, loader, device):
    backbone.eval(); head.eval()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    criterion = nn.CrossEntropyLoss()
    for images, targets in tqdm(loader, desc='val', leave=False):
        images, targets = images.to(device), targets.to(device)
        features = backbone.forward_features(images)
        logits = head(features)
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
    checkpoint_dir = mkdir(output_dir / 'linear_probe')
    log_dir = mkdir(output_dir / 'logs')
    plot_dir = mkdir(output_dir / 'plots')
    device = get_device()

    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))
    model_args = ckpt.get('args', {})
    class_names = ckpt['class_names']
    backbone = build_model_from_args(model_args).to(device)
    backbone.load_state_dict(ckpt['model'])
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    head = nn.Linear(backbone.num_features, 100).to(device)
    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=int(model_args.get('image_size', 32)),
        batch_size=args.batch_size,
        val_ratio=float(model_args.get('val_ratio', 0.1)),
        num_workers=args.num_workers,
        task='classification',
        seed=int(model_args.get('seed', 42)),
    )

    best_top1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
        for images, targets in tqdm(loaders['train'], desc=f'probe epoch {epoch}', leave=False):
            images, targets = images.to(device), targets.to(device)
            with torch.no_grad():
                features = backbone.forward_features(images)
            logits = head(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            top1, top5 = topk_accuracy(logits, targets)
            n = images.size(0)
            loss_meter.update(loss.item(), n)
            top1_meter.update(top1.item(), n)
            top5_meter.update(top5.item(), n)
        val = validate(backbone, head, loaders['val'], device)
        scheduler.step()
        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': loss_meter.avg,
            'train_top1': top1_meter.avg,
            'train_top5': top5_meter.avg,
            'val_loss': val['loss'],
            'val_top1': val['top1'],
            'val_top5': val['top5'],
        }
        history.append(row)
        print(f"epoch={epoch} val_top1={val['top1']:.2f} val_top5={val['top5']:.2f}")
        if val['top1'] > best_top1:
            best_top1 = val['top1']
            save_checkpoint({'head': head.state_dict(), 'backbone_checkpoint': args.checkpoint, 'class_names': class_names, 'epoch': epoch}, checkpoint_dir / 'best_linear_probe.pt')
        save_history_csv(history, log_dir / 'linear_probe_history.csv')
        save_json({'history': history}, log_dir / 'linear_probe_history.json')
        plot_training_curves(history, plot_dir / 'linear_probe_curves.png')
    print(f'Best linear-probe val top-1: {best_top1:.2f}%')


if __name__ == '__main__':
    main()
