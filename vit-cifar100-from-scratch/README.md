# Vision Transformer from Scratch on CIFAR-100

A GitHub-ready PyTorch project for training a **Vision Transformer (ViT) from scratch** on the **CIFAR-100** image classification dataset.

This project is designed to feel like a small experiment/vlog/blog workflow:
- visualize CIFAR-100 samples
- visualize augmentations
- train a ViT implemented from scratch
- log top-1 and top-5 accuracy
- save checkpoints
- plot training curves
- evaluate on the test set
- create confusion matrix, per-class accuracy, most-confused pairs, and prediction grids
- generate a Markdown report using saved metrics and plots

## Project structure

```text
vit-cifar100-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── vit_cifar100/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualize_dataset.py
│   └── generate_blog_report.py
└── outputs/
    └── .gitkeep
```

## Setup

```bash
git clone <your-repo-url>
cd vit-cifar100-from-scratch

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 1. Visualize CIFAR-100 like a mini vlog/blog

```bash
python scripts/visualize_dataset.py --data-dir data --output-dir outputs/plots
```

This creates:
- `sample_grid.png`
- `augmentation_grid.png`
- `class_distribution.png`

## 2. Train ViT from scratch

A good starter command for a single GPU:

```bash
python scripts/train.py \
  --data-dir data \
  --output-dir outputs \
  --epochs 100 \
  --batch-size 128 \
  --image-size 32 \
  --patch-size 4 \
  --embed-dim 192 \
  --depth 6 \
  --num-heads 6 \
  --mlp-ratio 4 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --rand-augment \
  --random-erasing 0.1 \
  --amp
```

For a quick sanity check:

```bash
python scripts/train.py --epochs 2 --batch-size 64 --quick-debug
```

Training outputs:
- `outputs/checkpoints/best.pt`
- `outputs/checkpoints/last.pt`
- `outputs/logs/history.csv`
- `outputs/logs/history.json`
- `outputs/plots/training_curves.png`

## 3. Evaluate the best checkpoint

```bash
python scripts/evaluate.py \
  --data-dir data \
  --checkpoint outputs/checkpoints/best.pt \
  --output-dir outputs
```

Evaluation outputs:
- `outputs/metrics/test_metrics.json`
- `outputs/metrics/per_class_metrics.csv`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/per_class_accuracy.png`
- `outputs/plots/most_confused_pairs.png`
- `outputs/plots/topk_predictions.png`

## 4. Predict one image

```bash
python scripts/predict.py \
  --checkpoint outputs/checkpoints/best.pt \
  --image path/to/image.png \
  --top-k 5
```

## 5. Generate a blog-style Markdown report

```bash
python scripts/generate_blog_report.py --output-dir outputs
```

This creates:

```text
outputs/reports/vit_cifar100_report.md
```

## Notes

- This is a **from-scratch ViT** implementation: patch embedding, class token, positional embedding, multi-head self-attention, MLP block, transformer encoder blocks, and classifier head are implemented in `vit_cifar100/model.py`.
- CIFAR-100 images are small, so `patch_size=4` gives an 8x8 patch grid for 32x32 images.
- ViTs often need enough epochs and regularization. Try 100-300 epochs for stronger results.
- For better accuracy, tune `embed_dim`, `depth`, `num_heads`, `weight_decay`, augmentations, and learning rate schedule.

## Suggested GitHub repo description

> Vision Transformer from scratch in PyTorch on CIFAR-100 with full training, evaluation, plots, metrics, and blog-style experiment report.
