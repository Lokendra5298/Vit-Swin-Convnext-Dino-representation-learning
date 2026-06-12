from dataclasses import dataclass


@dataclass
class ViTConfig:
    image_size: int = 96
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    num_register_tokens: int = 4


@dataclass
class DINOConfig:
    out_dim: int = 4096
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    teacher_momentum: float = 0.996
    global_size: int = 96
    local_size: int = 48
    local_crops_number: int = 6
