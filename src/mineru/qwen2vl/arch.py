"""Registers ``Qwen2VLForConditionalGeneration`` (Qwen2-VL) for ``max serve``."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import PixelContext, TextAndVisionContext, TextContext
from max.pipelines.context.exceptions import InputError
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Qwen2VLBatchProcessor
from .context import Qwen2VLTextAndVisionContext
from .model import Qwen2VLModel
from .model_config import Qwen2VLConfig
from .tokenizer import Qwen2VLTokenizer
from .weight_adapters import convert_qwen2vl_state_dict


def validate_qwen2vl_context(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Rejects contexts this pipeline can't batch.

    Raises:
        InputError: If the context isn't a ``Qwen2VLTextAndVisionContext``,
            or needs vision encoding without vision data.
    """
    if not isinstance(context, Qwen2VLTextAndVisionContext):
        raise InputError(
            f"context must be Qwen2VLTextAndVisionContext, got {type(context).__name__}"
        )
    if context.needs_vision_encoding and context.vision_data is None:
        raise InputError("vision_data is required when encoding images")


qwen2vl_arch = SupportedArchitecture(
    name="Qwen2VLForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["opendatalab/MinerU2.5-Pro-2605-1.2B"],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=Qwen2VLConfig.DEFAULT_ENCODING,
    supported_encodings=Qwen2VLConfig.SUPPORTED_ENCODINGS,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen2vl_state_dict,
    },
    pipeline_model=Qwen2VLModel,
    batching=Qwen2VLBatchProcessor,
    tokenizer=Qwen2VLTokenizer,
    context_type=Qwen2VLTextAndVisionContext,
    # The vision tower encodes whole images in the first prefill step, so a
    # prompt can't be split across steps (same as the native qwen2_5vl).
    required_arguments={"enable_chunked_prefill": False},
    context_validators=[validate_qwen2vl_context],
    config=Qwen2VLConfig,
    # Headroom for the vision tower's activations, which the KV planner
    # doesn't budget: at the checkpoint's max_pixels (8192 patches) one
    # block's qkv, MLP, and float32 rotary temporaries stay under 1 GiB.
    memory_planner=PagedMemoryPlanner.with_activation_reservation(2 * 1024**3),
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
