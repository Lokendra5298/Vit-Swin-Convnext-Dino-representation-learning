import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Stochastic depth per sample."""

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


class PatchEmbedding(nn.Module):
    """Image to patch embeddings using a convolutional projection."""

    def __init__(self, image_size: int = 96, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.base_grid_size = image_size // patch_size
        self.num_patches = self.base_grid_size * self.base_grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        return x, (grid_h, grid_w)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention implemented directly with torch operations."""

    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0, projection_dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-normalized transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Small ViT backbone for DINO-style representation learning.

    Supports variable crop sizes by interpolating patch positional embeddings.
    Optional register tokens are included between the class token and patch tokens.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = int(num_register_tokens)
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.base_grid_size = self.patch_embed.base_grid_size
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim))
            if self.num_register_tokens > 0
            else None
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.data.zero_()

    def interpolate_pos_encoding(self, grid_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        base = self.base_grid_size
        patch_pos = patch_pos.reshape(1, base, base, self.embed_dim).permute(0, 3, 1, 2)
        if grid_size != (base, base):
            patch_pos = F.interpolate(patch_pos, size=grid_size, mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], self.embed_dim)
        return cls_pos, patch_pos

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patch_tokens, grid_size = self.patch_embed(x)
        cls_pos, patch_pos = self.interpolate_pos_encoding(grid_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) + cls_pos
        patch_tokens = patch_tokens + patch_pos
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, register_tokens, patch_tokens], dim=1)
        else:
            x = torch.cat([cls_tokens, patch_tokens], dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)[:, 0]

    def forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        start = 1 + self.num_register_tokens
        return tokens[:, start:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class DINOHead(nn.Module):
    """Projection head used for DINO-style self-distillation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: List[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class DINOModel(nn.Module):
    """Backbone + DINO head wrapper."""

    def __init__(self, backbone: VisionTransformer, head: DINOHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    @property
    def feature_dim(self) -> int:
        return self.backbone.embed_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def create_vit(
    model_size: str = "small",
    image_size: int = 96,
    patch_size: int = 8,
    num_register_tokens: int = 4,
    drop_path_rate: float = 0.1,
) -> VisionTransformer:
    model_size = model_size.lower()
    if model_size == "tiny":
        return VisionTransformer(image_size, patch_size, 3, 192, 6, 3, 4.0, 0.0, 0.0, drop_path_rate, num_register_tokens)
    if model_size == "small":
        return VisionTransformer(image_size, patch_size, 3, 384, 6, 6, 4.0, 0.0, 0.0, drop_path_rate, num_register_tokens)
    if model_size == "base":
        return VisionTransformer(image_size, patch_size, 3, 768, 8, 12, 4.0, 0.0, 0.0, drop_path_rate, num_register_tokens)
    raise ValueError(f"Unknown model_size: {model_size}")


def create_dino_model(
    model_size: str = "small",
    image_size: int = 96,
    patch_size: int = 8,
    num_register_tokens: int = 4,
    drop_path_rate: float = 0.1,
    out_dim: int = 4096,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
) -> DINOModel:
    backbone = create_vit(model_size, image_size, patch_size, num_register_tokens, drop_path_rate)
    head = DINOHead(backbone.embed_dim, out_dim, hidden_dim, bottleneck_dim)
    return DINOModel(backbone=backbone, head=head)
