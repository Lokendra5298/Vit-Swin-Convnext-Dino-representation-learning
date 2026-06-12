import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from vit_cifar100.data import get_dataloaders
from vit_cifar100.metrics import compute_metrics_from_confusion, save_per_class_csv, topk_accuracy, update_confusion_matrix
from vit_cifar100.model import VisionTransformer
from vit_cifar100.plots import plot_confusion_matrix, plot_most_confused_pairs, plot_per_class_accuracy, plot_topk_predictions
from vit_cifar100.utils import get_device, load_checkpoint, mkdir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ViT checkpoint on CIFAR-100')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint) -> VisionTransformer:
    args = checkpoint.get('args', {})
    return VisionTransformer(
        image_size=int(args.get('image_size', 32)),
        patch_size=int(args.get('patch_size', 4)),
        in_channels=3,
        num_classes=100,
        embed_dim=int(args.get('embed_dim', 192)),
        depth=int(args.get('depth', 6)),
        num_heads=int(args.get('num_heads', 6)),
        mlp_ratio=float(args.get('mlp_ratio', 4.0)),
        dropout=float(args.get('dropout', 0.0)),
        attention_dropout=float(args.get('attention_dropout', 0.0)),
        drop_path=float(args.get('drop_path', 0.0)),
    )


@torch.no_grad()
def evaluate(model, loader, device, class_names, amp_enabled: bool):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    top1_sum = 0.0
    top5_sum = 0.0
    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.long)

    saved_batch = None

    for images, targets in tqdm(loader, desc='test'):
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

    avg_loss = total_loss / max(1, total_samples)
    top1_acc = top1_sum / max(1, total_samples)
    top5_acc = top5_sum / max(1, total_samples)

    return {'loss': avg_loss, 'top1': top1_acc, 'top5': top5_acc, 'confusion': confusion, 'saved_batch': saved_batch}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    metric_dir = mkdir(output_dir / 'metrics')
    plot_dir = mkdir(output_dir / 'plots')

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    class_names = checkpoint.get('class_names')
    if class_names is None:
        raise ValueError('Checkpoint does not contain class_names.')

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    image_size = int(checkpoint.get('args', {}).get('image_size', 32))
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        val_ratio=float(checkpoint.get('args', {}).get('val_ratio', 0.1)),
        num_workers=args.num_workers,
        rand_augment=False,
        random_erasing=0.0,
        seed=int(checkpoint.get('args', {}).get('seed', 42)),
    )

    amp_enabled = args.amp and device.type == 'cuda'
    stats = evaluate(model, loaders['test'], device, class_names, amp_enabled)

    metrics_from_cm = compute_metrics_from_confusion(stats['confusion'], class_names)

    summary = {
        'checkpoint': str(args.checkpoint),
        'test_loss': stats['loss'],
        'test_top1_percent': stats['top1'],
        'test_top5_percent': stats['top5'],
        'accuracy_from_confusion': metrics_from_cm['accuracy'],
        'macro_precision': metrics_from_cm['macro_precision'],
        'macro_recall': metrics_from_cm['macro_recall'],
        'macro_f1': metrics_from_cm['macro_f1'],
        'weighted_precision': metrics_from_cm['weighted_precision'],
        'weighted_recall': metrics_from_cm['weighted_recall'],
        'weighted_f1': metrics_from_cm['weighted_f1'],
    }

    save_json(summary, str(metric_dir / 'test_metrics.json'))
    save_per_class_csv(metrics_from_cm['per_class'], str(metric_dir / 'per_class_metrics.csv'))

    plot_confusion_matrix(stats['confusion'], class_names, str(plot_dir / 'confusion_matrix.png'), normalize=True)
    plot_per_class_accuracy(metrics_from_cm['per_class'], str(plot_dir / 'per_class_accuracy.png'))
    plot_most_confused_pairs(stats['confusion'], class_names, str(plot_dir / 'most_confused_pairs.png'), top_n=25)

    if stats['saved_batch'] is not None:
        images, targets, logits = stats['saved_batch']
        plot_topk_predictions(images, targets, logits, class_names, str(plot_dir / 'topk_predictions.png'), max_images=16, top_k=3)

    print('Evaluation complete.')
    print(f"Test top-1 accuracy: {summary['test_top1_percent']:.2f}%")
    print(f"Test top-5 accuracy: {summary['test_top5_percent']:.2f}%")
    print(f'Metrics saved to: {metric_dir}')
    print(f'Plots saved to: {plot_dir}')


if __name__ == '__main__':
    main()
