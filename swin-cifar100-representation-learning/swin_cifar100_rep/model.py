import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DropPath(nn.Module):
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


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 2, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('image_size must be divisible by patch_size')
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        num_relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        self.shift_size = 0 if self.window_size <= 1 else min(shift_size, self.window_size // 2)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

        if self.shift_size > 0:
            attn_mask = self._create_attention_mask(input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def _create_attention_mask(self, resolution: Tuple[int, int]) -> torch.Tensor:
        h, w = resolution
        img_mask = torch.zeros((1, h, w, 1))
        cnt = 0
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, l, c = x.shape
        if l != h * w:
            raise ValueError(f'Input feature has wrong size: got {l}, expected {h*w}')

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, l, c = x.shape
        if l != h * w:
            raise ValueError('Input feature has wrong size for patch merging')
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError('PatchMerging requires even spatial dimensions')

        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path_rates: List[float],
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path_rates[i],
                )
            )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None
        self.output_resolution = (input_resolution[0] // 2, input_resolution[1] // 2) if downsample else input_resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dim: int = 64,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (2, 4, 8, 16),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
        projection_dim: int = 128,
    ):
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError('depths and num_heads must have the same length')
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        patches_resolution = self.patch_embed.grid_size
        self.pos_drop = nn.Dropout(p=dropout)

        total_depth = sum(depths)
        drop_path_rates = torch.linspace(0, drop_path, total_depth).tolist()
        rate_index = 0
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            resolution = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
            stage = SwinStage(
                dim=dim,
                input_resolution=resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path_rates=drop_path_rates[rate_index: rate_index + depths[i_layer]],
                downsample=i_layer < self.num_layers - 1,
            )
            self.stages.append(stage)
            rate_index += depths[i_layer]

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.GELU(),
            nn.Linear(self.num_features, projection_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.projection_head(features)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits
