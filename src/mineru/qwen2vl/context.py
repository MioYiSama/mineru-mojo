"""Request context for Qwen2-VL.

Same per-request state as the native ``qwen2_5vl`` context (M-RoPE decoder
positions, ``rope_delta``, image placeholder indices) minus Qwen2.5-VL's
window-attention metadata: every Qwen2-VL vision block attends over the full
image frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context.context import TextAndVisionContext


@dataclass(kw_only=True, slots=True)
class Qwen2VLVisionData:
    """Vision tower inputs for one request's images, built by the tokenizer."""

    image_grid_thw: npt.NDArray[np.int32]
    """``[n_images, 3]`` patch grid (t, h, w) of each image."""

    vision_position_ids: npt.NDArray[np.integer[Any]]
    """``[n_patches, 2]`` (row, col) of each patch, in merge-group order."""

    cu_seqlens: npt.NDArray[np.uint32]
    """``[n_frames + 1]`` cumulative patch counts per image frame."""

    max_seqlen: int
    """Patch count of the largest frame."""

    concatenated_pixel_values: npt.NDArray[np.uint16]
    """``[n_patches, C*T*P*P]`` normalized patches as bfloat16 bit patterns."""


@dataclass(kw_only=True, slots=True)
class Qwen2VLTextAndVisionContext(TextAndVisionContext):
    """Text-and-vision context carrying Qwen2-VL's M-RoPE state."""

    spatial_merge_size: int
    rope_delta: int
    """Offset from a token's index to its M-RoPE position after the prompt."""

    image_token_id: int
    video_token_id: int
    vision_start_token_id: int

    tokens_per_second: int
    """Read by the shared ``get_rope_index`` for video grids only; this
    port's tokenizer accepts images, so it never takes effect."""

    image_token_indices: npt.NDArray[np.int32]
    decoder_position_ids: npt.NDArray[np.int64]
    """``[3, prompt_len]`` M-RoPE positions of the prompt tokens."""

    vision_data: Qwen2VLVisionData | None
