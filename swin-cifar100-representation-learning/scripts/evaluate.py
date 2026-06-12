import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from swin_cifar100_rep.data import get_dataloaders
from swin_cifar100_rep.metrics import AverageMeter, per_class_accuracy_from_confusion, save_rows_csv, topk_accuracy, update_confusion_matrix
from swin_cifar100_rep.model import SwinTransformer
from swin_cifar100_rep.plots import plot_confusion_matrix, plot_per_class_accuracy
from swin_cifar100_rep.utils import get_device, load_checkpoint, mkdir, save_json


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
    p = argparse.ArgumentParser(description='Evaluate supervised Swin checkpoint')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='outputs')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    return p.parse_args()


def autocast_context(device, enabled):
    return torch.autocast(device_type=device.type) if enabled else nullcontext()


@torch.no_grad()
def main():
    args = parse_args()
    metric_dir = mkdir(Path(args.output_dir) / 'metrics')
    plot_dir = mkdir(Path(args.output_dir) / 'plots')
    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))
    model_args = ckpt.get('args', {})
    class_names = ckpt['class_names']
    model = build_model_from_args(model_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=int(model_args.get('image_size', 32)),
        batch_size=args.batch_size,
        val_ratio=float(model_args.get('val_ratio', 0.1)),
        num_workers=args.num_workers,
        task='classification',
        seed=int(model_args.get('seed', 42)),
    )

    criterion = nn.CrossEntropyLoss()
    amp_enabled = args.amp and device.type == 'cuda'
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.long)

    for images, targets in tqdm(loaders['test'], desc='test'):
        images, targets = images.to(device), targets.to(device)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        top1, top5 = topk_accuracy(logits, targets)
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1.item(), n)
        top5_meter.update(top5.item(), n)
        update_confusion_matrix(confusion, preds.cpu(), targets.cpu())

    rows = per_class_accuracy_from_confusion(confusion, class_names)
    summary = {
        'test_loss': loss_meter.avg,
        'test_top1_percent': top1_meter.avg,
        'test_top5_percent': top5_meter.avg,
    }
    save_json(summary, metric_dir / 'test_metrics.json')
    save_rows_csv(rows, metric_dir / 'per_class_accuracy.csv')
    plot_confusion_matrix(confusion, class_names, plot_dir / 'confusion_matrix.png')
    plot_per_class_accuracy(rows, plot_dir / 'per_class_accuracy.png')
    print(f"Test top-1: {top1_meter.avg:.2f}%")
    print(f"Test top-5: {top5_meter.avg:.2f}%")


if __name__ == '__main__':
    main()
