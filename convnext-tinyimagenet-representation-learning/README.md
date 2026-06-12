# ConvNeXt from Scratch on Tiny ImageNet

A GitHub-ready PyTorch project for **ConvNeXt from scratch** on **Tiny ImageNet-200**, focused on:

- feature extraction
- representation learning
- supervised classification
- supervised contrastive learning
- frozen-backbone linear probing
- PCA visualization of embeddings
- training curves, confusion matrix, per-class accuracy, and top-k predictions

This repository intentionally implements ConvNeXt components in `convnext_tinyimagenet/model.py` rather than importing a ready-made ConvNeXt model from torchvision or timm.

## Project structure

```text
convnext-tinyimagenet-representation-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── convnext_tinyimagenet/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── prepare_tinyimagenet.py
│   ├── visualize_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_features.py
│   ├── linear_probe.py
│   └── visualize_embeddings.py
└── outputs/
    └── .gitkeep
```

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download Tiny ImageNet

```bash
python scripts/prepare_tinyimagenet.py --data-dir data
```

Expected layout after download/extraction:

```text
data/
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    ├── test/
    ├── wnids.txt
    └── words.txt
```

The code reads the original Tiny ImageNet validation annotation file directly, so you do not need to manually rearrange validation images into class folders.

## 3. Visualize the dataset

```bash
python scripts/visualize_dataset.py \
  --data-dir data \
  --output-dir outputs/plots
```

This creates:
- `sample_grid.png`
- `augmentation_grid.png`
- `class_distribution.png`

## 4. Train ConvNeXt for classification

```bash
python scripts/train.py \
  --task classification \
  --data-dir data \
  --output-dir outputs_classification \
  --epochs 100 \
  --batch-size 128 \
  --model-size nano \
  --amp
```

For a stronger but heavier model:

```bash
python scripts/train.py \
  --task classification \
  --data-dir data \
  --output-dir outputs_classification_tiny \
  --epochs 100 \
  --batch-size 96 \
  --model-size tiny \
  --amp
```

## 5. Train ConvNeXt for representation learning with SupCon

```bash
python scripts/train.py \
  --task supcon \
  --data-dir data \
  --output-dir outputs_supcon \
  --epochs 100 \
  --batch-size 128 \
  --model-size nano \
  --temperature 0.1 \
  --amp
```

SupCon creates two augmented views of each image and trains the encoder to pull same-class samples together in representation space.

## 6. Evaluate a classification checkpoint

```bash
python scripts/evaluate.py \
  --data-dir data \
  --checkpoint outputs_classification/checkpoints/best.pt \
  --output-dir outputs_classification
```

Outputs:
- `outputs_classification/metrics/test_metrics.json`
- `outputs_classification/metrics/per_class_metrics.csv`
- `outputs_classification/plots/confusion_matrix.png`
- `outputs_classification/plots/per_class_accuracy.png`
- `outputs_classification/plots/most_confused_pairs.png`
- `outputs_classification/plots/topk_predictions.png`

## 7. Extract features from any checkpoint

```bash
python scripts/extract_features.py \
  --data-dir data \
  --checkpoint outputs_supcon/checkpoints/best.pt \
  --output-dir outputs_supcon/features \
  --splits train val
```

This saves `.npz` files with:
- `features`: learned image embeddings
- `labels`: integer class labels
- `class_names`: class names

## 8. Visualize learned embeddings

```bash
python scripts/visualize_embeddings.py \
  --features outputs_supcon/features/val_features.npz \
  --output outputs_supcon/plots/pca_embeddings.png \
  --max-points 5000
```

## 9. Linear probe frozen ConvNeXt features

After SupCon training, freeze the ConvNeXt backbone and train only a linear classifier:

```bash
python scripts/linear_probe.py \
  --data-dir data \
  --checkpoint outputs_supcon/checkpoints/best.pt \
  --output-dir outputs_supcon_linear_probe \
  --epochs 50 \
  --batch-size 256 \
  --amp
```

## Suggested GitHub repo description

> ConvNeXt from scratch in PyTorch on Tiny ImageNet with classification, supervised contrastive representation learning, feature extraction, PCA embedding plots, and linear probing.

## Notes

- Tiny ImageNet images are 64x64, so the default image size is 64.
- The default `nano` ConvNeXt is smaller than official ConvNeXt-Tiny and is easier to train on a single GPU.
- Use `--model-size tiny` for a closer ConvNeXt-Tiny-style model.
- For representation learning, train with `--task supcon`, then run `extract_features.py`, `visualize_embeddings.py`, and `linear_probe.py`.
