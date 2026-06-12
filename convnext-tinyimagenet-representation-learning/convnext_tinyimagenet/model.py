from typing import List, Sequence, Tuple

import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth per sample.

    With drop_prob=0, this module is exactly an identity layer.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """LayerNorm that supports both channel-last and channel-first image tensors."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ):
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError("data_format must be channels_last or channels_first")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        # channels_first: x is [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block.

    Structure:
    depthwise 7x7 convolution -> channel-last LayerNorm -> pointwise MLP
    -> optional layer scale -> stochastic depth residual.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = residual + self.drop_path(x)
        return x


class ProjectionHead(nn.Module):
    """Small MLP projection head for contrastive representation learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNeXt(nn.Module):
    """ConvNeXt implemented from scratch for Tiny ImageNet classification/features."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 200,
        depths: Sequence[int] = (2, 2, 6, 2),
        dims: Sequence[int] = (64, 128, 256, 512),
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must each contain 4 values")

        self.num_classes = num_classes
        self.depths = tuple(depths)
        self.dims = tuple(dims)
        self.feature_dim = int(dims[-1])

        self.downsample_layers = nn.ModuleList()

        # Stem: 4x4 convolution with stride 4.
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_idx = 0

        self.stages = nn.ModuleList()
        for stage_idx in range(4):
            blocks = []
            for _ in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[stage_idx],
                        drop_path=dp_rates[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, LayerNorm2d)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features_map(x)
        x = x.mean(dim=(-2, -1))  # global average pool
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        features = self.head_dropout(features)
        logits = self.head(features)
        return logits


def convnext_nano(num_classes: int = 200, **kwargs) -> ConvNeXt:
    return ConvNeXt(
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 256, 512),
        **kwargs,
    )


def convnext_tiny(num_classes: int = 200, **kwargs) -> ConvNeXt:
    return ConvNeXt(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs,
    )


def convnext_small(num_classes: int = 200, **kwargs) -> ConvNeXt:
    return ConvNeXt(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        **kwargs,
    )


def create_convnext(model_size: str, num_classes: int = 200, **kwargs) -> ConvNeXt:
    model_size = model_size.lower()
    if model_size == "nano":
        return convnext_nano(num_classes=num_classes, **kwargs)
    if model_size == "tiny":
        return convnext_tiny(num_classes=num_classes, **kwargs)
    if model_size == "small":
        return convnext_small(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model_size: {model_size}")
