from dataclasses import dataclass
from typing import Tuple


@dataclass
class ConvNeXtConfig:
    image_size: int = 64
    in_channels: int = 3
    num_classes: int = 200
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2)
    dims: Tuple[int, int, int, int] = (64, 128, 256, 512)
    drop_path_rate: float = 0.1
    layer_scale_init_value: float = 1e-6
    head_dropout: float = 0.0


@dataclass
class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    task: str = "classification"
    epochs: int = 100
    batch_size: int = 128
    lr: float = 4e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    temperature: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = False
