"""Pipeline model for Qwen2-VL: a vision module and a language module.

Compiles two ModuleV3 modules from one checkpoint. ``Qwen2VLVisionModel``
runs only for a prefill step that carries images; its embeddings replace the
image placeholder rows in ``Qwen2VLLanguageModel``, which owns the paged KV
cache. Single device.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.modeling.eager_validation import eager_validator

from .batch_processor import Qwen2VLBatchProcessor, Qwen2VLInputs
from .context import Qwen2VLTextAndVisionContext
from .model_config import Qwen2VLConfig
from .qwen2vl import Qwen2VLLanguageModel
from .vision_encoder import Qwen2VLVisionModel
from .weight_adapters import LANGUAGE_PREFIX, VISION_PREFIX


class Qwen2VLModel(
    ModuleV3MultiGraphPipelineModelWithKVCache[Qwen2VLTextAndVisionContext]
):
    """Qwen2-VL pipeline model (``Qwen2VLForConditionalGeneration``)."""

    model_config_cls: ClassVar[type[Any]] = Qwen2VLConfig
    batch_processor_cls: ClassVar[type[Qwen2VLBatchProcessor]] = Qwen2VLBatchProcessor

    vision_model: Callable[..., Any] | None
    language_model: Callable[..., Any]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            memory_plan=memory_plan,
            max_batch_size=max_batch_size,
        )
        if len(devices) != 1:
            raise ValueError("Qwen2-VL port runs on a single device")
        self.vision_model, self.language_model = self.load_model()

    @property
    def _qwen_batch_processor(self) -> Qwen2VLBatchProcessor:
        assert isinstance(self._batch_processor, Qwen2VLBatchProcessor)
        return self._batch_processor

    def _load_state_dict(self) -> dict[str, Any]:
        """Splits the adapted checkpoint per tower.

        The vision module is compiled from its own root, so its keys drop the
        ``vision_encoder.`` prefix; the language root keeps
        ``language_model.``.
        """
        assert self.adapter is not None
        state_dict = self.adapter(dict(self.weights.items()))
        self._vision_weights_dict = {
            key.removeprefix(VISION_PREFIX): value
            for key, value in state_dict.items()
            if key.startswith(VISION_PREFIX)
        }
        self._language_weights_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(LANGUAGE_PREFIX)
        }
        return state_dict

    def _create_model_config(self, state_dict: dict[str, Any]) -> Qwen2VLConfig:
        model_config = Qwen2VLConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=self._language_weights_dict,
            return_logits=self.return_logits,
        )
        return model_config

    def _compile_vision_model(
        self, model_config: Qwen2VLConfig, state_dict: dict[str, Any]
    ) -> Callable[..., Any]:
        with F.lazy(), default_dtype(model_config.vision_config.dtype):
            vision_nn = Qwen2VLVisionModel(model_config.vision_config)
            vision_nn.to(self.devices[0])
        input_types = self._qwen_batch_processor.get_vision_symbolic_inputs(
            DeviceRef.from_device(self.devices[0])
        )
        return vision_nn.compile(*input_types, weights=state_dict)

    def _compile_language_model(
        self, model_config: Qwen2VLConfig, state_dict: dict[str, Any]
    ) -> Callable[..., Any]:
        with F.lazy(), default_dtype(model_config.llm_config.dtype):
            language_nn = Qwen2VLLanguageModel(model_config, self.kv_params)
            language_nn.to(self.devices[0])
        input_types = self._qwen_batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params, device_refs=self.device_refs
        )
        return language_nn.compile(*input_types, weights=state_dict)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        inputs = cast(Qwen2VLInputs, model_inputs)
        assert inputs.kv_cache_inputs is not None

        image_embeddings: Buffer
        if inputs.has_vision_inputs:
            assert self.vision_model is not None
            with eager_validator.graph_break(
                reason="the tower runs only for a prefill carrying images"
            ):
                vision_out = self.vision_model(
                    inputs.pixel_values,
                    inputs.vision_position_ids,
                    inputs.cu_seqlens,
                    inputs.max_seqlen,
                )
            image_embeddings = _buffer(vision_out)
        else:
            image_embeddings = self._qwen_batch_processor.empty_image_embeddings()

        outputs = self.language_model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            image_embeddings,
            inputs.image_token_indices,
            inputs.position_ids,
            *tree.leaves(inputs.kv_cache_inputs),
        )
        if len(outputs) == 3:
            return ModelOutputs(
                next_token_logits=_buffer(outputs[0]),
                logits=_buffer(outputs[1]),
                logit_offsets=_buffer(outputs[2]),
            )
        logits = _buffer(outputs[0])
        return ModelOutputs(next_token_logits=logits, logits=logits)


def _buffer(t: Tensor | tuple[Tensor, ...]) -> Buffer:
    if isinstance(t, tuple):
        (t,) = t
    return cast(Buffer, t.driver_tensor)
