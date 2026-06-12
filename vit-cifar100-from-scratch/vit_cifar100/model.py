import math
from typing import List, Optional, Tuple

import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth per sample.

    With drop_prob=0 it behaves exactly like an identity layer.
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


class PatchEmbedding(nn.Module):
    """Split image into patches and project each patch to an embedding vector."""

    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('image_size must be divisible by patch_size')

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention implemented from scratch."""

    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0, projection_dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, tokens, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attention
        return x, None


class MLP(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attention = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attention


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification.

    This implementation is intentionally educational and repository-friendly:
    it does not call torchvision's ViT model. The transformer components are
    implemented in this file.
    """

    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3, num_classes: int = 100, embed_dim: int = 192, depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1, attention_dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        if depth > 1:
            drop_path_rates = torch.linspace(0, drop_path, depth).tolist()
        else:
            drop_path_rates = [drop_path]

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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

    def forward_features(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        attentions = [] if return_attention else None
        for block in self.blocks:
            x, attention = block(x, return_attention=return_attention)
            if return_attention:
                attentions.append(attention)

        x = self.norm(x)
        cls_embedding = x[:, 0]
        return cls_embedding, attentions

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        features, attentions = self.forward_features(x, return_attention=return_attention)
        logits = self.head(features)
        if return_attention:
            return logits, attentions
        return logits
