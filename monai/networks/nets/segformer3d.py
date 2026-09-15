# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers import DropPath, trunc_normal_

__all__ = ["SegFormer3D"]


class OverlapPatchEmbedding(nn.Module):
    """
    Overlapping patch embedding for volumetric inputs, consisting of a strided ``Conv3d``
    followed by a ``LayerNorm`` over the flattened token sequence. The overlapping kernels
    preserve local continuity between neighboring voxels, unlike the non-overlapping
    patchify used by vanilla ViT.

    Args:
        in_channels: number of input channels.
        embed_dim: embedding (token) dimension of this stage.
        patch_size: kernel size of the strided convolution.
        stride: stride of the strided convolution.
        padding: padding of the strided convolution.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)  # (B, C, D', H', W')
        spatial_shape = (int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1]))
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return self.norm(x), spatial_shape


class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with spatial reduction of keys and values. Keys and values are
    projected from a spatially reduced token map: the tokens are restored to a volumetric
    feature map, reduced with a strided ``Conv3d`` (by ``sr_ratio`` along each spatial axis,
    i.e. ``sr_ratio ** 3`` times fewer tokens) followed by a ``LayerNorm``, and flattened back
    to a sequence. Queries keep the full resolution. This reduces the cost of attention from
    O(n^2) to O(n^2 / sr_ratio ** 3), which keeps 3D attention tractable. When ``sr_ratio``
    is 1 no reduction is applied.

    Args:
        dim: token dimension.
        num_heads: number of attention heads, must divide ``dim``.
        sr_ratio: spatial reduction ratio applied to keys and values along each axis.
        qkv_bias: enable bias for the query and key/value projections.
        attn_drop_rate: dropout rate on the attention weights.
        dropout: dropout rate on the output projection.
    """

    def __init__(
        self, dim: int, num_heads: int, sr_ratio: int, qkv_bias: bool, attn_drop_rate: float, dropout: float
    ) -> None:
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be positive and divide dim ({dim}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = nn.Identity()
            self.sr_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)

        if self.sr_ratio > 1:
            d, h, w = spatial_shape
            y = x.transpose(1, 2).reshape(b, c, d, h, w)
            y = self.sr(y).flatten(2).transpose(1, 2)  # (B, N/r^3, C)
            y = self.sr_norm(y)
        else:
            y = x

        kv = self.kv(y).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, heads, N/r^3, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N/r^3)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)

        return self.proj_drop(self.proj(x))


class MixFFN(nn.Module):
    """
    Mix-FFN: a feed-forward network that encodes positional information implicitly with a
    3x3x3 depthwise convolution instead of an explicit positional embedding, making the model
    robust to resolution changes between training and inference.

    The computation is ``Linear -> DepthwiseConv3d(3x3x3) -> GELU -> Linear`` applied to the
    token sequence, where the depthwise convolution operates on the hidden expansion of the
    first linear layer.

    Args:
        dim: token dimension.
        mlp_ratio: hidden dimension expansion ratio of the first linear layer.
        dropout: dropout rate after the activation.
    """

    def __init__(self, dim: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        b, _, _ = x.shape
        d, h, w = spatial_shape
        x = self.fc1(x)  # (B, N, hidden)
        x = self.dwconv(x.transpose(1, 2).reshape(b, -1, d, h, w)).flatten(2).transpose(1, 2)
        x = self.drop(self.act(x))
        return self.fc2(x)


class SegFormer3DBlock(nn.Module):
    """
    Pre-norm transformer block combining efficient self-attention and Mix-FFN, each with a
    residual connection and stochastic depth.

    Args:
        dim: token dimension.
        num_heads: number of attention heads.
        sr_ratio: spatial reduction ratio for the attention keys and values.
        mlp_ratio: hidden dimension expansion ratio of the Mix-FFN.
        qkv_bias: enable bias for the attention projections.
        dropout: dropout rate after the attention output projection and the Mix-FFN.
        attn_drop_rate: dropout rate on the attention weights.
        drop_path: stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: int,
        qkv_bias: bool,
        dropout: float,
        attn_drop_rate: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), spatial_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x), spatial_shape))
        return x


class SegFormer3DStage(nn.Module):
    """
    One encoder stage: overlapping patch embedding followed by a stack of transformer blocks
    and a final ``LayerNorm``. The stage consumes a volumetric feature map and returns the
    coarsest-to-finest hierarchical feature map of the pyramid.

    Args:
        in_channels: number of input channels (embedding dimension of the previous stage).
        embed_dim: embedding dimension of this stage.
        depth: number of transformer blocks.
        num_heads: number of attention heads.
        sr_ratio: spatial reduction ratio for the attention keys and values.
        mlp_ratio: hidden dimension expansion ratio of the Mix-FFN.
        patch_size: kernel size of the overlapping patch embedding.
        stride: stride of the overlapping patch embedding.
        padding: padding of the overlapping patch embedding.
        qkv_bias: enable bias for the attention projections.
        dropout: dropout rate.
        attn_drop_rate: dropout rate on the attention weights.
        drop_path_rates: stochastic depth rate of each block in this stage.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: int,
        patch_size: int,
        stride: int,
        padding: int,
        qkv_bias: bool,
        dropout: float,
        attn_drop_rate: float,
        drop_path_rates: Sequence[float],
    ) -> None:
        super().__init__()
        self.embedding = OverlapPatchEmbedding(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, stride=stride, padding=padding
        )
        self.blocks = nn.ModuleList(
            [
                SegFormer3DBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_drop_rate=attn_drop_rate,
                    drop_path=drop_path_rates[block_idx],
                )
                for block_idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, spatial_shape = self.embedding(x)
        for block in self.blocks:
            tokens = block(tokens, spatial_shape)
        tokens = self.norm(tokens)
        b, n, c = tokens.shape
        d, h, w = spatial_shape
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class SegFormer3D(nn.Module):
    """
    SegFormer3D based on: "Perera et al.,
    SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation
    <https://arxiv.org/abs/2404.10156>".

    SegFormer3D is a hierarchical 3D transformer for volumetric medical image segmentation.
    The encoder is a 4-stage Mix Vision Transformer that produces a feature pyramid: each
    stage embeds overlapping patches with a strided convolution, then applies transformer
    blocks made of efficient (spatially reduced) self-attention and a Mix-FFN whose 3x3x3
    depthwise convolution encodes positional information implicitly (no positional
    embedding). The decoder is a lightweight all-MLP head: every stage feature is projected
    to a common embedding dimension, upsampled to the first stage resolution, concatenated,
    fused, and classified per voxel.

    This implementation is a clean-room reimplementation of the architecture described in the
    paper, written from the paper text only.

    Args:
        spatial_dims: number of spatial dimensions, must be 3 (volumetric inputs only).
        in_channels: number of input channels.
        out_channels: number of output segmentation classes.
        depths: number of transformer blocks in each encoder stage.
        embed_dims: embedding dimension of each encoder stage.
        num_heads: number of attention heads in each encoder stage.
        sr_ratios: spatial reduction ratio of the attention keys and values in each stage.
        mlp_ratios: hidden dimension expansion ratio of the Mix-FFN in each stage.
        decoder_head_embedding_dim: common embedding dimension of the all-MLP decoder.
        dropout: dropout rate after the attention output projections and the Mix-FFNs.
        drop_path_rate: stochastic depth rate, linearly scaled from 0 across all blocks.
        attn_drop_rate: dropout rate on the attention weights.
        qkv_bias: enable bias for the attention query and key/value projections.

    References:
        Perera, S., Navard, P., Yilmaz, A. "SegFormer3D: an Efficient Transformer for
        3D Medical Image Segmentation" arXiv:2404.10156 (2024).
        https://arxiv.org/abs/2404.10156
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        embed_dims: Sequence[int] = (32, 64, 160, 256),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        sr_ratios: Sequence[int] = (4, 2, 1, 1),
        mlp_ratios: Sequence[int] = (4, 4, 4, 4),
        decoder_head_embedding_dim: int = 128,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dims != 3:
            raise ValueError(f"SegFormer3D only supports 3D inputs, expected spatial_dims=3 but got {spatial_dims}.")
        num_stages = len(depths)
        if any(len(seq) != num_stages for seq in (embed_dims, num_heads, sr_ratios, mlp_ratios)):
            raise ValueError(
                f"depths, embed_dims, num_heads, sr_ratios and mlp_ratios must have the same length, got "
                f"{len(depths)}, {len(embed_dims)}, {len(num_heads)}, {len(sr_ratios)} and {len(mlp_ratios)}."
            )

        # Overlapping patch embeddings: a large patch in the first stage, smaller ones afterwards.
        patch_sizes = (7,) + (3,) * (num_stages - 1)
        strides = (4,) + (2,) * (num_stages - 1)
        paddings = (3,) + (1,) * (num_stages - 1)

        drop_path_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.stages = nn.ModuleList()
        for stage_idx in range(num_stages):
            self.stages.append(
                SegFormer3DStage(
                    in_channels=in_channels if stage_idx == 0 else embed_dims[stage_idx - 1],
                    embed_dim=embed_dims[stage_idx],
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    sr_ratio=sr_ratios[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    patch_size=patch_sizes[stage_idx],
                    stride=strides[stage_idx],
                    padding=paddings[stage_idx],
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rates=drop_path_rates[sum(depths[:stage_idx]) : sum(depths[: stage_idx + 1])],
                )
            )

        # All-MLP decoder: per-stage projection to a common embedding, then fusion and classification.
        self.decoder_projections = nn.ModuleList(
            [nn.Conv3d(dim, decoder_head_embedding_dim, kernel_size=1) for dim in embed_dims]
        )
        self.fusion = nn.Conv3d(decoder_head_embedding_dim * num_stages, decoder_head_embedding_dim, kernel_size=1)
        self.fusion_norm = nn.LayerNorm(decoder_head_embedding_dim)
        self.fusion_act = nn.GELU()
        self.classifier = nn.Conv3d(decoder_head_embedding_dim, out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial_shape = x.shape[2:]

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        projected = [projection(feature) for projection, feature in zip(self.decoder_projections, features)]
        stage_one_shape = projected[0].shape[2:]
        upsampled = [F.interpolate(f, size=stage_one_shape, mode="trilinear", align_corners=False) for f in projected]
        fused = self.fusion(torch.cat(upsampled, dim=1))
        fused = self.fusion_act(self.fusion_norm(fused.movedim(1, -1))).movedim(-1, 1)

        logits = self.classifier(fused)
        return F.interpolate(logits, size=input_spatial_shape, mode="trilinear", align_corners=False)
