import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from vit_cifar100.utils import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a blog-style Markdown report from saved outputs')
    parser.add_argument('--output-dir', type=str, default='outputs')
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open('r') as f:
        return json.load(f)


def read_history(path: Path):
    if not path.exists():
        return []
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = mkdir(output_dir / 'reports')

    history = read_history(output_dir / 'logs' / 'history.csv')
    metrics = read_json(output_dir / 'metrics' / 'test_metrics.json')

    best_epoch = None
    if history:
        best_epoch = max(history, key=lambda row: float(row['val_top1']))

    report_path = report_dir / 'vit_cifar100_report.md'

    lines = []
    lines.append('# ViT from Scratch on CIFAR-100: Experiment Report\n')
    lines.append('This report is generated automatically from the training logs, metrics, and plots saved by the project scripts.\n')

    lines.append('## 1. Dataset visuals\n')
    lines.append('![Sample grid](../plots/sample_grid.png)\n')
    lines.append('![Augmentation grid](../plots/augmentation_grid.png)\n')
    lines.append('![Class distribution](../plots/class_distribution.png)\n')

    lines.append('## 2. Model\n')
    lines.append('The model is a Vision Transformer implemented from scratch in `vit_cifar100/model.py`. It uses patch embeddings, a learnable class token, positional embeddings, transformer encoder blocks, multi-head self-attention, MLP blocks, and a classification head.\n')

    lines.append('## 3. Training curves\n')
    lines.append('![Training curves](../plots/training_curves.png)\n')

    if best_epoch is not None:
        lines.append('### Best validation epoch\n')
        lines.append(
            f"- Epoch: `{best_epoch['epoch']}`\n"
            f"- Validation top-1: `{float(best_epoch['val_top1']):.2f}%`\n"
            f"- Validation top-5: `{float(best_epoch['val_top5']):.2f}%`\n"
            f"- Validation loss: `{float(best_epoch['val_loss']):.4f}`\n"
        )

    lines.append('## 4. Test results\n')
    if metrics is not None:
        lines.append(
            f"- Test loss: `{metrics['test_loss']:.4f}`\n"
            f"- Test top-1 accuracy: `{metrics['test_top1_percent']:.2f}%`\n"
            f"- Test top-5 accuracy: `{metrics['test_top5_percent']:.2f}%`\n"
            f"- Macro F1: `{metrics['macro_f1']:.4f}`\n"
            f"- Weighted F1: `{metrics['weighted_f1']:.4f}`\n"
        )
    else:
        lines.append('Run `scripts/evaluate.py` to add test metrics here.\n')

    lines.append('## 5. Evaluation plots\n')
    lines.append('![Confusion matrix](../plots/confusion_matrix.png)\n')
    lines.append('![Per-class accuracy](../plots/per_class_accuracy.png)\n')
    lines.append('![Most confused pairs](../plots/most_confused_pairs.png)\n')
    lines.append('![Top-k predictions](../plots/topk_predictions.png)\n')

    lines.append('## 6. Things to improve next\n')
    lines.append('- Train longer, for example 200-300 epochs.\n')
    lines.append('- Tune learning rate, weight decay, dropout, and stochastic depth.\n')
    lines.append('- Try larger ViT settings such as `embed_dim=384`, `depth=8`, `num_heads=8`.\n')
    lines.append('- Add Mixup/CutMix for stronger regularization.\n')
    lines.append('- Compare against a small CNN baseline.\n')

    report_path.write_text('\n'.join(lines))
    print(f'Report saved to: {report_path}')


if __name__ == '__main__':
    main()
