"""Ragged batching for Qwen2-VL's vision and language modules.

Builds the language module's ragged token inputs plus the ``[3, n_tokens]``
M-RoPE positions, and, for a prefill that carries images, the vision
module's patch rows, patch positions, and per-frame attention boundaries.
Decoder positions and merge indices reuse the native ``qwen2_5vl`` batch
processor's helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputs
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.architectures.qwen2_5vl.batch_processor import (
    Qwen2_5VLBatchProcessor,
)
from max.pipelines.lib import ModelInputs
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    process_ragged_kv_outputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices
from max.pipelines.modeling.config_enums import supported_encoding_dtype

from .context import Qwen2VLTextAndVisionContext
from .model_config import Qwen2VLConfig


@dataclass
class Qwen2VLInputs(ModelInputs):
    """One step's inputs for the Qwen2-VL vision and language modules."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    position_ids: Buffer
    """``[3, total_seq_len]`` uint32 M-RoPE positions."""

    image_token_indices: Buffer
    """``[n_image_tokens]`` int32 rows of ``tokens`` that take image
    embeddings; negative entries are skipped by the merge."""

    pixel_values: Buffer | None = None
    """``[n_patches, C*T*P*P]`` bfloat16 patches, when images are encoded."""

    vision_position_ids: Buffer | None = None
    """``[n_patches, 2]`` int64 (row, col) patch positions."""

    cu_seqlens: Buffer | None = None
    """``[n_frames + 1]`` uint32 frame boundaries over the patch rows."""

    max_seqlen: Buffer | None = None
    """``[1]`` uint32 (CPU) patch count of the largest frame."""

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None


class Qwen2VLBatchProcessor(BatchProcessor[Qwen2VLTextAndVisionContext, Qwen2VLInputs]):
    """Single-device ragged batching with optional image inputs."""

    def __init__(self, config: Qwen2VLConfig, runtime: BatchProcessorRuntime) -> None:
        super().__init__(config, runtime)
        self._qwen_config = config
        device = runtime.devices[0]
        assert config.quantization_encoding is not None
        self._empty_image_embeddings = Buffer.zeros(
            shape=[0, config.llm_config.hidden_size],
            dtype=supported_encoding_dtype(config.quantization_encoding),
        ).to(device)

    def empty_image_embeddings(self) -> Buffer:
        """Zero-row image embeddings for steps that encode no image."""
        return self._empty_image_embeddings

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Language module inputs in ``Qwen2VLLanguageModel.forward`` order."""
        device_ref = device_refs[0]
        return [
            TensorType(DType.int64, ["total_seq_len"], device=device_ref),
            TensorType(DType.int64, ["return_n_logits"], device=DeviceRef.CPU()),
            TensorType(DType.uint32, ["input_row_offsets_len"], device=device_ref),
            TensorType(
                self._empty_image_embeddings.dtype,
                ["num_image_tokens", self._qwen_config.llm_config.hidden_size],
                device=device_ref,
            ),
            TensorType(DType.int32, ["total_image_tokens"], device=device_ref),
            TensorType(DType.uint32, [3, "total_seq_len"], device=device_ref),
            *kv_params.flattened_kv_inputs(),
        ]

    def get_vision_symbolic_inputs(self, device_ref: DeviceRef) -> list[TensorType]:
        """Vision module inputs in ``Qwen2VLVisionModel.forward`` order."""
        vision = self._qwen_config.vision_config
        return [
            TensorType(
                vision.dtype, ["n_patches", vision.patch_dim], device=device_ref
            ),
            TensorType(DType.int64, ["n_patches", 2], device=device_ref),
            TensorType(DType.uint32, ["n_frames_plus_1"], device=device_ref),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
        ]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Qwen2VLTextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen2VLInputs:
        if len(replica_batches) > 1:
            raise ValueError("Qwen2-VL does not support data parallelism")
        if kv_cache_inputs is None:
            raise ValueError("Qwen2-VL requires KV cache inputs")
        context_batch = replica_batches[0]
        device = self.runtime.devices[0]

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        inputs = Qwen2VLInputs(
            tokens=Buffer.from_numpy(tokens).to(device),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(device),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            position_ids=Qwen2_5VLBatchProcessor.prepare_decoder_position_ids(
                context_batch,  # type: ignore[arg-type]
                [device],
            ),
            image_token_indices=Buffer.from_numpy(
                compute_multimodal_merge_indices(context_batch)
            ).to(device),
            kv_cache_inputs=kv_cache_inputs,
        )

        vision_datas = [
            ctx.vision_data
            for ctx in context_batch
            if ctx.needs_vision_encoding and ctx.vision_data is not None
        ]
        if not vision_datas:
            return inputs

        pixel_values = Buffer.from_numpy(
            np.concatenate([v.concatenated_pixel_values for v in vision_datas])
        )
        inputs.pixel_values = pixel_values.view(DType.bfloat16, pixel_values.shape).to(
            device
        )
        inputs.vision_position_ids = Buffer.from_numpy(
            np.concatenate([v.vision_position_ids for v in vision_datas]).astype(
                np.int64
            )
        ).to(device)
        # Shift each request's frame boundaries past the earlier requests'.
        cu_parts = [np.zeros(1, dtype=np.uint32)]
        offset = 0
        for v in vision_datas:
            cu_parts.append(v.cu_seqlens[1:] + offset)
            offset += int(v.cu_seqlens[-1])
        inputs.cu_seqlens = Buffer.from_numpy(
            np.concatenate(cu_parts).astype(np.uint32)
        ).to(device)
        inputs.max_seqlen = Buffer.from_numpy(
            np.array([max(v.max_seqlen for v in vision_datas)], np.uint32)
        )
        return inputs

    def process_outputs(self, outputs: Sequence[Buffer | object]) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
