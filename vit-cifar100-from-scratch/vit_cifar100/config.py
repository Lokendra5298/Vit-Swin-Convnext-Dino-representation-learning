from dataclasses import dataclass


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    drop_path: float = 0.0


@dataclass
class TrainConfig:
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    epochs: int = 100
    batch_size: int = 128
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    val_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = False
    rand_augment: bool = False
    random_erasing: float = 0.0
