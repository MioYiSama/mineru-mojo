"""Tokenizer and image preprocessing for Qwen2-VL.

Reuses the native ``qwen2_5vl`` tokenizer's NumPy/PIL pipeline: the patch
layout, normalization, placeholder expansion, and M-RoPE position indexing
are shared by Qwen2-VL and Qwen2.5-VL. Qwen2-VL differs in two ways handled
here:

- The resize bounds come from the checkpoint's ``preprocessor_config.json``
  (``min_pixels`` / ``max_pixels``), as HF ``Qwen2VLImageProcessor`` does,
  rather than the Qwen2.5-VL defaults.
- The vision tower attends over whole frames, so no window index is built.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import fetch_image
from max.pipelines.architectures.qwen2_5vl.tokenizer import (
    Qwen2_5VLImageProcessor,
    Qwen2_5VLTokenizer,
)
from max.pipelines.context import EOSTracker, ImageMetadata, TokenBuffer
from max.pipelines.lib import VisionPreprocessCache, resolve_eos_token_ids
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import encode_dkv_cache_hint
from max.pipelines.modeling.types import TextGenerationRequest
from max.support.image import find_contiguous_ranges, hash_image
from transformers import AutoTokenizer
from transformers.utils import cached_file

from .context import Qwen2VLTextAndVisionContext, Qwen2VLVisionData

# HF ``Qwen2VLImageProcessor`` defaults, used when the checkpoint ships no
# resize bounds.
_HF_MIN_PIXELS = 56 * 56
_HF_MAX_PIXELS = 28 * 28 * 1280


def _resize_bounds(model_path: str, revision: str | None) -> tuple[int, int]:
    """Reads ``(min_pixels, max_pixels)`` like HF ``Qwen2VLImageProcessor``.

    ``size`` (``shortest_edge`` / ``longest_edge``, in pixels) takes
    precedence over the ``min_pixels`` / ``max_pixels`` keys.
    """
    path = cached_file(
        model_path,
        "preprocessor_config.json",
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if path is None:
        return _HF_MIN_PIXELS, _HF_MAX_PIXELS
    with open(path) as f:
        config = json.load(f)
    size = config.get("size") or {}
    min_pixels = size.get("shortest_edge", config.get("min_pixels", _HF_MIN_PIXELS))
    max_pixels = size.get("longest_edge", config.get("max_pixels", _HF_MAX_PIXELS))
    return int(min_pixels), int(max_pixels)


class Qwen2VLTokenizer(Qwen2_5VLTokenizer):
    """Qwen2-VL tokenizer producing :class:`Qwen2VLTextAndVisionContext`."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        # The parent constructor reads Qwen2.5-VL-only vision_config fields
        # (window_size, tokens_per_second), so this sets the same attributes
        # from the Qwen2-VL config instead of delegating.
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        config = pipeline_config.model.huggingface_config
        vision_config = config.vision_config
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.min_pixels, self.max_pixels = _resize_bounds(model_path, revision)
        self.img_processor = Qwen2_5VLImageProcessor(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            merge_size=self.spatial_merge_size,
        )
        self._preprocess_cache = VisionPreprocessCache.for_images(
            pipeline_config.runtime
        )
        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        # Only read for video grids, which this tokenizer never produces.
        self.tokens_per_second = 1
        self.executor: ThreadPoolExecutor | None = None

    def _process_images(
        self, request: TextGenerationRequest
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        list[npt.NDArray[np.uint16]],
        list[int | None],
    ]:
        """Resizes within the checkpoint's pixel bounds, then patchifies."""
        image_inputs = [
            fetch_image(
                {
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
            for image in request.images_for_processing()
        ]
        image_hashes: list[int | None] = (
            [hash_image(raw_bytes, self.max_pixels) for raw_bytes in request.images]
            if self.enable_prefix_caching or self._preprocess_cache.enabled
            else [None] * len(request.images)
        )

        per_image = [
            self._preprocess_image(image_hash, image)
            for image_hash, image in zip(image_hashes, image_inputs, strict=True)
        ]
        pixel_values_list = [pixels for pixels, _ in per_image]
        return (
            np.vstack(pixel_values_list),
            np.array([grid for _, grid in per_image], dtype=np.int32),
            pixel_values_list,
            image_hashes,
        )

    def _create_context(
        self,
        request: TextGenerationRequest,
        input_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
        image_grid_thw: npt.NDArray[np.int32] | None,
        images: list[ImageMetadata],
        vision_data: Qwen2VLVisionData | None,  # type: ignore[override]
        eos_tracker: EOSTracker,
    ) -> Qwen2VLTextAndVisionContext:
        image_token_indices = (
            (input_ids == self.image_token_id).nonzero()[0].astype(np.int32)
        )
        decoder_position_ids, rope_delta = self._calculate_rope_params(
            input_ids, attention_mask, image_grid_thw
        )
        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )
        return Qwen2VLTextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=eos_tracker,
            tokens=TokenBuffer(input_ids),
            max_length=self._max_length_of_request(input_ids.shape[0], request),
            json_schema=json_schema,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            images=images,
            vision_token_ids=[self.image_token_id],
            spatial_merge_size=self.spatial_merge_size,
            rope_delta=rope_delta,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            tokens_per_second=self.tokens_per_second,
            image_token_indices=image_token_indices,
            decoder_position_ids=decoder_position_ids,
            vision_data=vision_data,
            vocab_size=self.tokenizer_vocab_size,
            cache_salt=request.cache_salt,
        )

    def new_context_blocking(
        self, request: TextGenerationRequest
    ) -> Qwen2VLTextAndVisionContext:
        eos_tracker = self._build_eos_tracker(request)
        if not request.images:
            input_ids, attention_mask = self._tokenize_inputs(request, None)
            return self._create_context(
                request=request,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=None,
                images=[],
                vision_data=None,
                eos_tracker=eos_tracker,
            )

        (
            concatenated_pixel_values,
            image_grid_thw,
            pixel_values_list,
            image_hashes,
        ) = self._process_images(request)
        input_ids, attention_mask = self._tokenize_inputs(request, image_grid_thw)

        # One attention segment per frame: HF get_vision_cu_seqlens.
        frame_sizes = np.repeat(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        )
        cu_seqlens = np.concatenate([[0], np.cumsum(frame_sizes)]).astype(np.uint32)
        vision_data = Qwen2VLVisionData(
            image_grid_thw=image_grid_thw,
            vision_position_ids=mrope_pos_ids_3d(
                grid_thw=image_grid_thw,
                spatial_merge_size=self.spatial_merge_size,
            ),
            cu_seqlens=cu_seqlens,
            max_seqlen=int(frame_sizes.max()),
            concatenated_pixel_values=concatenated_pixel_values,
        )

        image_ranges = find_contiguous_ranges(input_ids, [self.image_token_id])
        images = [
            ImageMetadata(
                start_idx=start_idx,
                end_idx=end_idx,
                pixel_values=pixel_values,
                image_hash=image_hash if self.enable_prefix_caching else None,
            )
            for (start_idx, end_idx), pixel_values, image_hash in zip(
                image_ranges, pixel_values_list, image_hashes, strict=True
            )
        ]
        return self._create_context(
            request=request,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            images=images,
            vision_data=vision_data,
            eos_tracker=eos_tracker,
        )
