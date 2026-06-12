# Mini-DINOv2-Style Representation Learning on STL-10

A GitHub-ready PyTorch project for an educational **DINOv2-style self-supervised model** trained on the **STL-10 unlabeled split**, with downstream evaluation on the labeled STL-10 train/test splits.

This is intentionally a small, readable, from-scratch implementation. It does **not** claim to reproduce Meta AI's full DINOv2 training recipe or scale. Instead, it implements the core ideas that are useful for learning and GitHub projects:

- Vision Transformer backbone from scratch
- optional register tokens inspired by DINOv2-style ViTs
- multi-crop augmentation
- student and teacher networks
- EMA teacher update
- centering and sharpening self-distillation loss
- feature extraction
- k-NN evaluation
- frozen-backbone linear probing
- PCA visualization of learned representations

## Project structure

```text
minidino-stl10-representation-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── minidino_stl10/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── prepare_stl10.py
│   ├── visualize_dataset.py
│   ├── pretrain_dino.py
│   ├── extract_features.py
│   ├── knn_eval.py
│   ├── linear_probe.py
│   ├── visualize_embeddings.py
│   └── generate_report.py
└── outputs/
    └── .gitkeep
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download STL-10

```bash
python scripts/prepare_stl10.py --data-dir data
```

This downloads:
- `unlabeled`: used for self-supervised pretraining
- `train`: used for linear probe training
- `test`: used for evaluation

## 3. Visualize STL-10 and multi-crop views

```bash
python scripts/visualize_dataset.py --data-dir data --output-dir outputs/plots
```

Outputs:
- `stl10_unlabeled_samples.png`
- `stl10_labeled_samples.png`
- `multicrop_views.png`

## 4. Pretrain Mini-DINOv2-style model on STL-10 unlabeled

```bash
python scripts/pretrain_dino.py \
  --data-dir data \
  --output-dir outputs_dino \
  --epochs 100 \
  --batch-size 128 \
  --model-size small \
  --global-size 96 \
  --local-size 48 \
  --local-crops-number 6 \
  --amp
```

Quick debug run:

```bash
python scripts/pretrain_dino.py \
  --data-dir data \
  --output-dir outputs_debug \
  --epochs 2 \
  --batch-size 16 \
  --quick-debug
```

## 5. Extract frozen features

```bash
python scripts/extract_features.py \
  --data-dir data \
  --checkpoint outputs_dino/checkpoints/best.pt \
  --output-dir outputs_dino/features \
  --splits train test
```

## 6. k-NN evaluation

```bash
python scripts/knn_eval.py \
  --train-features outputs_dino/features/train_features.npz \
  --test-features outputs_dino/features/test_features.npz \
  --k 20 \
  --temperature 0.07
```

## 7. Linear probe

```bash
python scripts/linear_probe.py \
  --data-dir data \
  --checkpoint outputs_dino/checkpoints/best.pt \
  --output-dir outputs_dino_linear_probe \
  --epochs 50 \
  --batch-size 256 \
  --amp
```

## 8. Visualize learned embeddings

```bash
python scripts/visualize_embeddings.py \
  --features outputs_dino/features/test_features.npz \
  --output outputs_dino/plots/pca_embeddings.png \
  --max-points 5000
```

## What is implemented from scratch?

In `minidino_stl10/model.py`:
- patch embedding
- multi-head self-attention
- transformer encoder block
- ViT backbone
- optional register tokens
- DINO projection head

In `minidino_stl10/losses.py`:
- DINO self-distillation loss
- teacher centering
- teacher sharpening
- cross-view loss over global and local crops

In `scripts/pretrain_dino.py`:
- student and teacher networks
- EMA teacher update
- cosine learning-rate schedule
- cosine weight-decay schedule
- cosine teacher-momentum schedule

## Suggested GitHub repo description

> Mini-DINOv2-style self-supervised ViT from scratch on STL-10 unlabeled data with feature extraction, k-NN evaluation, PCA plots, and frozen linear probing.

## Notes

- STL-10 images are 96x96.
- Default global crop size is 96.
- Default local crop size is 48.
- `patch_size=8` works for both 96 and 48 crops.
- Use `model-size tiny` for CPU/small GPU experiments.
- Use `model-size small` for a better representation-learning baseline.
