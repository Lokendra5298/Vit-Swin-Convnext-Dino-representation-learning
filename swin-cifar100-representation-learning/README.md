# Swin Transformer from Scratch for CIFAR-100 Representation Learning

This is a GitHub-ready PyTorch project that implements a **Swin Transformer from scratch** and trains it on the public **CIFAR-100** dataset.

The project focuses on two goals:

1. **Feature extraction**: train a Swin backbone and export image embeddings.
2. **Representation learning**: train the backbone either with supervised classification or supervised contrastive learning, then evaluate the frozen representation with a linear probe.

No torchvision Swin model is used. The model code implements patch embedding, window partitioning, shifted-window attention, relative position bias, patch merging, Swin blocks, and a classification/projection head in `swin_cifar100_rep/model.py`.

## Project structure

```text
swin-cifar100-representation-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── swin_cifar100_rep/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_features.py
│   ├── linear_probe.py
│   ├── visualize_dataset.py
│   └── visualize_embeddings.py
└── outputs/
    └── .gitkeep
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Visualize CIFAR-100

```bash
python scripts/visualize_dataset.py --data-dir data --output-dir outputs/plots
```

Creates:

```text
outputs/plots/sample_grid.png
outputs/plots/augmentation_grid.png
outputs/plots/class_distribution.png
```

## 2. Train Swin with supervised classification

```bash
python scripts/train.py   --task classification   --data-dir data   --output-dir outputs   --epochs 100   --batch-size 128   --image-size 32   --patch-size 2   --window-size 4   --embed-dim 64   --depths 2,2,6,2   --num-heads 2,4,8,16   --lr 3e-4   --weight-decay 0.05   --amp
```

Quick debug run:

```bash
python scripts/train.py --task classification --epochs 2 --batch-size 64 --quick-debug
```

## 3. Train Swin with supervised contrastive learning

This trains the backbone to produce better representations using two augmented views per image and a supervised contrastive loss.

```bash
python scripts/train.py   --task supcon   --data-dir data   --output-dir outputs_supcon   --epochs 100   --batch-size 128   --temperature 0.1   --amp
```

After SupCon training, evaluate the frozen representation with a linear probe:

```bash
python scripts/linear_probe.py   --data-dir data   --checkpoint outputs_supcon/checkpoints/best.pt   --output-dir outputs_supcon   --epochs 50
```

## 4. Evaluate a classification checkpoint

```bash
python scripts/evaluate.py   --data-dir data   --checkpoint outputs/checkpoints/best.pt   --output-dir outputs
```

Creates:

```text
outputs/metrics/test_metrics.json
outputs/metrics/per_class_accuracy.csv
outputs/plots/confusion_matrix.png
outputs/plots/per_class_accuracy.png
```

## 5. Extract features

```bash
python scripts/extract_features.py   --data-dir data   --checkpoint outputs/checkpoints/best.pt   --output-dir outputs/features   --splits train val test
```

Creates `.npz` files with:

- `features`: Swin backbone embeddings
- `labels`: CIFAR-100 labels
- `class_names`: label names

## 6. Visualize learned representations with PCA

```bash
python scripts/visualize_embeddings.py   --features outputs/features/test_features.npz   --output outputs/plots/pca_embeddings.png   --max-points 3000
```

## Suggested GitHub topics

```text
pytorch, swin-transformer, cifar100, representation-learning, feature-extraction, computer-vision, from-scratch
```

## Notes

- CIFAR-100 images are 32x32. This project uses `patch_size=2`, which creates a 16x16 initial patch grid.
- `window_size=4` works well for small images.
- For serious accuracy, train longer and tune augmentations, model size, weight decay, learning rate, and stochastic depth.
- SupCon is included because it is a clean and understandable representation-learning objective before moving to a DINO/DINOv2-style self-supervised project.
