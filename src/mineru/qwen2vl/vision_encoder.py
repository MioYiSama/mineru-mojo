"""Qwen2-VL vision tower (ViT + patch merger) in ModuleV3.

Mirrors HF ``Qwen2VisionTransformerPretrainedModel``:

- ``PatchEmbed``: a bias-free Conv3d whose stride equals its kernel, run here
  as a Linear over each pre-cut ``C*T*P*P`` patch row.
- ``Qwen2VLVisionBlock``: pre-LayerNorm residual blocks. Attention uses a
  fused biased ``qkv`` projection, 2D rotary embeddings in the
  ``rotate_half`` layout, and full (non-windowed) attention within each
  image frame. The MLP is ``fc2(quick_gelu(fc1(x)))``.
- ``PatchMerger``: LayerNorm, then each group of ``spatial_merge_size**2``
  neighboring patches is concatenated and projected by
  ``Linear -> GELU -> Linear`` to the text decoder's width.

The image preprocessor orders patches so that each merge group is
contiguous, so the merger is a plain reshape.
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import LayerNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged_gpu as _flash_attention_ragged_gpu,
)

from .model_config import Qwen2VLVisionConfig

flash_attention_ragged_gpu = F.functional(_flash_attention_ragged_gpu)

# HF ``VisionRotaryEmbedding`` default base.
_VISION_ROPE_THETA = 10000.0


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x.split([half, half], axis=-1)
    return F.concat([-x2, x1], axis=-1)


class PatchEmbed(Module[[Tensor], Tensor]):
    """Projects flattened ``[n_patches, C*T*P*P]`` patches to ``embed_dim``."""

    def __init__(self, patch_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = Linear(patch_dim, embed_dim, bias=False)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.proj(pixel_values)


class VisionAttention(Module[..., Tensor]):
    """Bidirectional self-attention over each image frame's patches."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        n_patches = x.shape[0]
        qkv = self.qkv(x).reshape([n_patches, 3, self.num_heads, self.head_dim])
        q, k, v = (t.squeeze(1) for t in qkv.split([1, 1, 1], axis=1))

        # HF apply_rotary_pos_emb_vision: rotate in float32, cast back.
        dtype = q.dtype
        q32 = q.cast(DType.float32)
        k32 = k.cast(DType.float32)
        q = (q32 * cos + _rotate_half(q32) * sin).cast(dtype)
        k = (k32 * cos + _rotate_half(k32) * sin).cast(dtype)

        attn = flash_attention_ragged_gpu(
            q=q,
            k=k,
            v=v,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )
        return self.proj(attn.reshape([n_patches, -1]))


class VisionMlp(Module[[Tensor], Tensor]):
    """``fc2(act(fc1(x)))`` with HF's ``quick_gelu`` activation."""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        if hidden_act != "quick_gelu":
            raise ValueError(f"Unsupported Qwen2-VL vision activation: {hidden_act!r}")
        self.fc1 = Linear(dim, hidden_dim, bias=True)
        self.fc2 = Linear(hidden_dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="quick"))


class VisionBlock(Module[..., Tensor]):
    """HF ``Qwen2VLVisionBlock``: ``x + attn(norm1(x))``, ``x + mlp(norm2(x))``."""

    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config.embed_dim, config.num_heads)
        self.mlp = VisionMlp(config.embed_dim, config.mlp_hidden_dim, config.hidden_act)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.norm2(x))


class PatchMerger(Module[[Tensor], Tensor]):
    """HF ``PatchMerger``: LayerNorm, merge-group concat, Linear-GELU-Linear."""

    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.merge_unit = config.spatial_merge_size**2
        self.merged_dim = config.embed_dim * self.merge_unit
        self.ln_q = LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.fc1 = Linear(self.merged_dim, self.merged_dim, bias=True)
        self.fc2 = Linear(self.merged_dim, config.out_hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        n_patches, dim = x.shape
        # Every image contributes whole merge groups (its grid h and w are
        # multiples of spatial_merge_size), so n_patches divides evenly.
        x = F.rebind(
            self.ln_q(x),
            [(n_patches // self.merge_unit) * self.merge_unit, dim],
            message="vision patches must form whole merge groups",
        )
        x = x.reshape([-1, self.merged_dim])
        # nn.GELU() default: exact (erf) GELU.
        return self.fc2(F.gelu(self.fc1(x)))


class Qwen2VLVisionModel(Module[..., Tensor]):
    """Encodes patchified images into text-decoder-width embeddings.

    Inputs (one row per patch, across every image in the batch):
        pixel_values: ``[n_patches, C*T*P*P]`` normalized patches.
        rot_pos_ids: ``[n_patches, 2]`` int64 (row, col) patch positions.
        cu_seqlens: ``[n_frames + 1]`` uint32 frame boundaries; attention
            stays within a frame.
        max_seqlen: ``[1]`` uint32 on CPU, the longest frame.

    Returns:
        ``[n_patches / spatial_merge_size**2, out_hidden_size]`` embeddings.
    """

    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(config.patch_dim, config.embed_dim)
        self.blocks = ModuleList(VisionBlock(config) for _ in range(config.depth))
        self.merger = PatchMerger(config)
        # HF builds VisionRotaryEmbedding(head_dim // 2): each of the row
        # and column positions rotates head_dim // 4 frequency pairs.
        self.rope_dim = (config.embed_dim // config.num_heads) // 2

    def _rotary_cos_sin(self, rot_pos_ids: Tensor) -> tuple[Tensor, Tensor]:
        """HF ``VisionRotaryEmbedding`` + ``cat((emb, emb))`` cos/sin.

        Returns ``[n_patches, 1, head_dim]`` float32 tensors, broadcast over
        heads.
        """
        iota = F.arange(
            0,
            self.rope_dim,
            step=2,
            dtype=DType.float32,
            device=rot_pos_ids.device,
        )
        inv_freq = 1.0 / (_VISION_ROPE_THETA ** (iota / self.rope_dim))
        # [n, 2, 1] * [rope_dim // 2] -> [n, 2, rope_dim // 2] -> [n, rope_dim]
        freqs = rot_pos_ids.cast(DType.float32).unsqueeze(-1) * inv_freq
        freqs = freqs.reshape([rot_pos_ids.shape[0], self.rope_dim])
        emb = F.concat([freqs, freqs], axis=-1).unsqueeze(1)
        return F.cos(emb), F.sin(emb)

    def forward(
        self,
        pixel_values: Tensor,
        rot_pos_ids: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        h = self.patch_embed(pixel_values)
        cos, sin = self._rotary_cos_sin(rot_pos_ids)
        for block in self.blocks:
            h = block(h, cos, sin, cu_seqlens, max_seqlen)
        return self.merger(h)
