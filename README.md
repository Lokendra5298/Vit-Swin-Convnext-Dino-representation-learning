# Vision Models from Scratch for Representation Learning

A collection of modern computer vision architectures implemented from scratch in PyTorch, focused on image classification, feature extraction, and representation learning.

This repository contains four independent projects:

1. Vision Transformer on CIFAR-100
2. Swin Transformer on CIFAR-100
3. ConvNeXt on Tiny ImageNet
4. Mini-DINOv2-style self-supervised model on STL-10

The goal of this repository is to provide clean, educational, and research-friendly implementations of important vision backbones. Each model is organized as a separate project with its own training scripts, evaluation scripts, plotting utilities, and detailed README.

---

## Repository Overview

```text
vision-transformers-representation-learning/
├── README.md
├── requirements.txt
├── vit-cifar100/
├── swin-cifar100/
├── convnext-tinyimagenet/
├── minidino-stl10/
├── assets/
│   ├── architecture_vit.png
│   ├── architecture_swin.png
│   ├── architecture_convnext.png
│   └── architecture_dino.png
└── docs/
    ├── ViT_README.md
    ├── Swin_README.md
    ├── ConvNeXt_README.md
    └── MiniDINO_README.md
```

---

## Models Included

| Model                   | Dataset                       | Learning Type                        | Main Focus                                                                      |
| ----------------------- | ----------------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| Vision Transformer      | CIFAR-100                     | Supervised learning                  | Patch embeddings, transformer encoder, classification, feature extraction       |
| Swin Transformer        | CIFAR-100                     | Supervised + representation learning | Shifted window attention, hierarchical vision transformer, feature extraction   |
| ConvNeXt                | Tiny ImageNet                 | Supervised + contrastive learning    | Modern convolutional backbone, stage-wise features, linear probing              |
| Mini-DINOv2-style Model | STL-10 unlabeled + train/test | Self-supervised learning             | Student-teacher training, multi-crop views, frozen features, k-NN, linear probe |

---

# 1. Vision Transformer from Scratch on CIFAR-100

The Vision Transformer project implements a ViT model from scratch for CIFAR-100 classification.

## Key Components

* Patch embedding
* Learnable class token
* Positional embeddings
* Multi-head self-attention
* Transformer encoder blocks
* MLP classification head
* Training and validation curves
* Confusion matrix
* Per-class accuracy
* Feature extraction

## Dataset

CIFAR-100 contains 100 image classes with small RGB images of size 32 × 32.

## Project Directory

```text
vit-cifar100/
├── vit_cifar100/
│   ├── model.py
│   ├── data.py
│   ├── metrics.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── visualize_dataset.py
└── README.md
```

## Example Usage

```bash
cd vit-cifar100
pip install -r requirements.txt

python scripts/visualize_dataset.py --data-dir data --output-dir outputs/plots

python scripts/train.py \
  --data-dir data \
  --output-dir outputs \
  --epochs 100 \
  --batch-size 128 \
  --amp

python scripts/evaluate.py \
  --data-dir data \
  --checkpoint outputs/checkpoints/best.pt \
  --output-dir outputs
```

---

# 2. Swin Transformer from Scratch on CIFAR-100

The Swin Transformer project implements a hierarchical transformer using shifted-window self-attention. It is designed for both classification and representation learning.

## Key Components

* Patch embedding
* Window-based multi-head self-attention
* Shifted-window attention
* Patch merging
* Hierarchical transformer stages
* Classification head
* Supervised contrastive training
* Feature extraction
* PCA visualization
* Linear probing

## Dataset

This project uses CIFAR-100 for supervised classification and representation learning.

## Project Directory

```text
swin-cifar100/
├── swin_cifar100_rep/
│   ├── model.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_features.py
│   ├── linear_probe.py
│   └── visualize_embeddings.py
└── README.md
```

## Example Usage

```bash
cd swin-cifar100
pip install -r requirements.txt

python scripts/train.py \
  --task classification \
  --data-dir data \
  --output-dir outputs_classification \
  --epochs 100 \
  --batch-size 128 \
  --amp
```

For supervised contrastive representation learning:

```bash
python scripts/train.py \
  --task supcon \
  --data-dir data \
  --output-dir outputs_supcon \
  --epochs 100 \
  --batch-size 128 \
  --temperature 0.1 \
  --amp
```

Feature extraction:

```bash
python scripts/extract_features.py \
  --data-dir data \
  --checkpoint outputs_supcon/checkpoints/best.pt \
  --output-dir outputs_supcon/features \
  --splits train val test
```

Linear probing:

```bash
python scripts/linear_probe.py \
  --data-dir data \
  --checkpoint outputs_supcon/checkpoints/best.pt \
  --output-dir outputs_supcon_linear_probe \
  --epochs 50
```

---

# 3. ConvNeXt from Scratch on Tiny ImageNet

The ConvNeXt project implements a modern convolutional neural network inspired by transformer-era design principles. It is trained on Tiny ImageNet for classification and representation learning.

## Key Components

* ConvNeXt stem
* Depthwise convolution
* LayerNorm
* Pointwise MLP-style convolution layers
* GELU activation
* LayerScale
* Residual connections
* Stage-wise downsampling
* Supervised classification
* Supervised contrastive learning
* Feature extraction
* Linear probing
* PCA embedding visualization

## Dataset

Tiny ImageNet contains 200 classes with 64 × 64 RGB images.

## Project Directory

```text
convnext-tinyimagenet/
├── convnext_tinyimagenet/
│   ├── model.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── prepare_tinyimagenet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_features.py
│   ├── linear_probe.py
│   └── visualize_embeddings.py
└── README.md
```

## Example Usage

Download and prepare Tiny ImageNet:

```bash
cd convnext-tinyimagenet
pip install -r requirements.txt

python scripts/prepare_tinyimagenet.py --data-dir data
```

Train ConvNeXt for classification:

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

Train ConvNeXt with supervised contrastive learning:

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

Extract features:

```bash
python scripts/extract_features.py \
  --data-dir data \
  --checkpoint outputs_supcon/checkpoints/best.pt \
  --output-dir outputs_supcon/features \
  --splits train val
```

Run linear probe:

```bash
python scripts/linear_probe.py \
  --data-dir data \
  --checkpoint outputs_supcon/checkpoints/best.pt \
  --output-dir outputs_supcon_linear_probe \
  --epochs 50 \
  --batch-size 256 \
  --amp
```

---

# 4. Mini-DINOv2-Style Self-Supervised Model on STL-10

The Mini-DINOv2-style project implements an educational self-supervised student-teacher training pipeline inspired by DINO and DINOv2-style representation learning.

This implementation is designed for learning and experimentation. It does not reproduce the full-scale DINOv2 training recipe, but it includes the core ideas needed to understand self-supervised visual representation learning.

## Key Components

* Vision Transformer backbone from scratch
* Optional register tokens
* DINO projection head
* Multi-crop augmentation
* Student network
* Teacher network
* Exponential moving average teacher update
* Centering and sharpening
* Self-distillation loss
* Feature extraction
* k-NN evaluation
* Frozen-backbone linear probing
* PCA visualization

## Dataset

This project uses STL-10:

* `unlabeled` split for self-supervised pretraining
* `train` split for linear probing
* `test` split for evaluation

## Project Directory

```text
minidino-stl10/
├── minidino_stl10/
│   ├── model.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── prepare_stl10.py
│   ├── pretrain_dino.py
│   ├── extract_features.py
│   ├── knn_eval.py
│   ├── linear_probe.py
│   └── visualize_embeddings.py
└── README.md
```

## Example Usage

Download STL-10:

```bash
cd minidino-stl10
pip install -r requirements.txt

python scripts/prepare_stl10.py --data-dir data
```

Pretrain on STL-10 unlabeled:

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

Extract frozen features:

```bash
python scripts/extract_features.py \
  --data-dir data \
  --checkpoint outputs_dino/checkpoints/best.pt \
  --output-dir outputs_dino/features \
  --splits train test
```

Run k-NN evaluation:

```bash
python scripts/knn_eval.py \
  --train-features outputs_dino/features/train_features.npz \
  --test-features outputs_dino/features/test_features.npz \
  --k 20 \
  --temperature 0.07
```

Run linear probe:

```bash
python scripts/linear_probe.py \
  --data-dir data \
  --checkpoint outputs_dino/checkpoints/best.pt \
  --output-dir outputs_dino_linear_probe \
  --epochs 50 \
  --batch-size 256 \
  --amp
```

---

## Installation

Each project contains its own `requirements.txt`. A common environment can also be created from the root of the repository.

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision numpy matplotlib pillow tqdm
```

For CUDA-enabled training, install the PyTorch version that matches your CUDA version from the official PyTorch installation page.

---

## Common Workflow

Most projects follow this workflow:

```text
1. Prepare or download dataset
2. Visualize dataset samples
3. Train model
4. Evaluate model
5. Extract features
6. Visualize learned embeddings
7. Run linear probe or k-NN evaluation
8. Save plots and metrics
```

---

## Outputs

Each project saves experiment outputs inside its own `outputs/` directory.

Typical outputs include:

```text
outputs/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── history.csv
│   └── history.json
├── metrics/
│   ├── test_metrics.json
│   └── per_class_metrics.csv
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── pca_embeddings.png
└── reports/
    └── experiment_report.md
```

---

## Architecture Diagrams

Architecture diagrams can be placed inside the `assets/` directory:

```text
assets/
├── architecture_vit.png
├── architecture_swin.png
├── architecture_convnext.png
└── architecture_dino.png
```

You can reference them in this README as:

```markdown
![ViT Architecture](assets/architecture_vit.png)
![Swin Transformer Architecture](assets/architecture_swin.png)
![ConvNeXt Architecture](assets/architecture_convnext.png)
![Mini-DINO Architecture](assets/architecture_dino.png)
```

---

## Recommended Order to Study

For better understanding, study the models in this order:

1. Vision Transformer
2. Swin Transformer
3. ConvNeXt
4. Mini-DINOv2-style self-supervised model

This order moves from a basic transformer image classifier to hierarchical transformers, modern convolutional networks, and finally self-supervised representation learning.

---

## Learning Objectives

By working through this repository, you will learn:

* How patch embeddings work in vision transformers
* How multi-head self-attention is applied to images
* How Swin Transformer uses local windows and shifted windows
* How ConvNeXt modernizes convolutional neural networks
* How self-supervised student-teacher learning works
* How to extract visual features from trained backbones
* How to evaluate representations using k-NN and linear probing
* How to visualize learned embeddings using PCA
* How to organize deep learning experiments for GitHub

---

## Disclaimer

These implementations are designed for education, experimentation, and portfolio projects. They are intentionally written to be readable and modular.

The models are implemented from scratch where possible, but the datasets, image transforms, PyTorch training utilities, and standard tensor operations use PyTorch and torchvision.

The Mini-DINOv2-style project is inspired by DINO-style self-supervised representation learning, but it is not an official DINOv2 reproduction.

---

## License
MIT

This project is intended for educational and research use. Add your preferred license file, such as MIT License, before publishing the repository.
