"""Config for the Qwen2-VL port.

``Qwen2VLConfig`` composes a ``Qwen2VLVisionConfig`` for the ViT tower and a
ModuleV3 ``Llama3Config`` for the Qwen2 text decoder. The text decoder's
Qwen2-specific settings (q/k/v bias, tied embeddings, M-RoPE sections) are
read from the Hugging Face ``text_config`` here, and the KV cache is sized
from that same sub-config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self, override

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.llama3_modulev3.model_config import (
    Llama3Config,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchVLConfigWithTextSubconfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig

# HF ``Qwen2VLVisionBlock`` and ``PatchMerger`` hard-code this epsilon for
# every LayerNorm in the vision tower; ``vision_config`` doesn't carry it.
_VISION_LAYER_NORM_EPS = 1e-6


@dataclass(kw_only=True)
class Qwen2VLVisionConfig:
    """Qwen2-VL ViT hyperparameters, read from HF ``vision_config``."""

    dtype: DType
    """Dtype of the vision tower weights and activations."""

    patch_size: int
    temporal_patch_size: int
    in_channels: int

    embed_dim: int
    """ViT width (HF ``embed_dim``)."""

    num_heads: int
    depth: int

    mlp_hidden_dim: int
    """``embed_dim * mlp_ratio``: width of each block's fc1/fc2 MLP."""

    hidden_act: str
    """Block MLP activation (``quick_gelu`` for Qwen2-VL)."""

    out_hidden_size: int
    """Merger output width (HF ``vision_config.hidden_size``), equal to the
    text decoder's hidden size."""

    spatial_merge_size: int
    layer_norm_eps: float = _VISION_LAYER_NORM_EPS

    @property
    def patch_dim(self) -> int:
        """Flattened length of one patch: ``C * T * P * P``."""
        return (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

    @classmethod
    def from_hf(cls, hf_vision_config: AutoConfig, dtype: DType) -> Self:
        return cls(
            dtype=dtype,
            patch_size=hf_vision_config.patch_size,
            temporal_patch_size=hf_vision_config.temporal_patch_size,
            in_channels=hf_vision_config.in_channels,
            embed_dim=hf_vision_config.embed_dim,
            num_heads=hf_vision_config.num_heads,
            depth=hf_vision_config.depth,
            mlp_hidden_dim=int(hf_vision_config.embed_dim * hf_vision_config.mlp_ratio),
            hidden_act=hf_vision_config.hidden_act,
            out_hidden_size=hf_vision_config.hidden_size,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
        )


@dataclass(kw_only=True)
class Qwen2VLConfig(ArchVLConfigWithTextSubconfig, ArchConfigWithKVCache):
    """Configuration for Qwen2-VL (``Qwen2VLForConditionalGeneration``)."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    devices: list[DeviceRef]

    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    spatial_merge_size: int

    mrope_section: list[int]
    """Per-axis (temporal, height, width) split of the rotary frequencies."""

    vision_config: Qwen2VLVisionConfig
    llm_config: Llama3Config
    quantization_encoding: SupportedEncoding | None = None

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the text decoder's KV cache parameters."""
        return self.llm_config.get_kv_params()

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}'."
            )
        hf_text_config = huggingface_config.text_config
        hf_vision_config = huggingface_config.vision_config

        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        device_refs = [
            DeviceRef(spec.device_type, spec.id) for spec in model_config.device_specs
        ]

        rope_parameters = hf_text_config.rope_parameters
        return cls(
            devices=device_refs,
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            vision_start_token_id=huggingface_config.vision_start_token_id,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            mrope_section=list(rope_parameters["mrope_section"]),
            vision_config=Qwen2VLVisionConfig.from_hf(hf_vision_config, dtype),
            llm_config=_text_config(
                pipeline_config,
                model_config,
                hf_text_config,
                quantization_encoding,
                device_refs,
                max_seq_len,
            ),
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Sets the text-decoder fields that depend on the checkpoint."""
        # Qwen2 attention projects Q/K/V with bias (HF Qwen2VLAttention).
        self.llm_config.finalize(
            huggingface_config=huggingface_config.text_config,
            state_dict=llm_state_dict,
            return_logits=return_logits,
            attention_bias=True,
        )


def _text_config(
    pipeline_config: PipelineConfig,
    model_config: MAXModelConfig,
    hf_text_config: AutoConfig,
    quantization_encoding: SupportedEncoding,
    device_refs: list[DeviceRef],
    max_seq_len: int,
) -> Llama3Config:
    """Builds the Qwen2 text decoder's ``Llama3Config`` from ``text_config``.

    ``Llama3Config.initialize`` reads the top-level HF config, which for
    Qwen2-VL carries no text fields, so this builds it from the sub-config.
    The rotary frequencies are default (unscaled) RoPE; M-RoPE only changes
    which position each frequency reads, which the attention applies.
    """
    rope_type = hf_text_config.rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"Unsupported Qwen2-VL rope_type: {rope_type!r}")
    if getattr(hf_text_config, "use_sliding_window", False):
        raise ValueError("Qwen2-VL sliding-window attention is not supported")

    cache_dtype = cache_dtype_for_encoding(
        quantization_encoding, model_config.kv_cache.kv_cache_format
    )
    return Llama3Config(
        hidden_size=hf_text_config.hidden_size,
        num_attention_heads=hf_text_config.num_attention_heads,
        num_key_value_heads=hf_text_config.num_key_value_heads,
        num_hidden_layers=hf_text_config.num_hidden_layers,
        rope_theta=get_rope_theta(hf_text_config),
        rope_scaling_params=None,
        intermediate_size=hf_text_config.intermediate_size,
        interleaved_rope_weights=_interleaved_rope_weights(model_config),
        vocab_size=hf_text_config.vocab_size,
        dtype=supported_encoding_dtype(quantization_encoding),
        max_seq_len=max_seq_len,
        kv_params=Llama3Config.construct_kv_params(
            huggingface_config=hf_text_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype,
        ),
        attention_multiplier=Llama3Config.calculate_attention_multiplier(
            hf_text_config
        ),
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=device_refs,
        quantization_encoding=quantization_encoding,
    )
