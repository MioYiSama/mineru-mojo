"""Maps Qwen2-VL safetensors keys onto the port's two compiled modules.

The checkpoint uses the Transformers 4 layout: ``model.*`` for the Qwen2
text decoder and ``visual.*`` for the ViT tower, with no ``lm_head`` (the
head is tied to ``model.embed_tokens``). The adapter returns one dict whose
keys carry a tower prefix, ``language_model.`` or ``vision_encoder.``;
``Qwen2VLModel._load_state_dict`` splits it per tower.
"""

from __future__ import annotations

from typing import Any

from max.driver import Buffer
from max.graph.weights import WeightData, Weights

LANGUAGE_PREFIX = "language_model."
VISION_PREFIX = "vision_encoder."

# HF ``PatchMerger.mlp`` is ``nn.Sequential(Linear, GELU, Linear)``.
_VISION_RENAMES = (
    ("merger.mlp.0.", "merger.fc1."),
    ("merger.mlp.2.", "merger.fc2."),
)


def convert_qwen2vl_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs: Any
) -> dict[str, WeightData]:
    """Renames checkpoint keys to ``language_model.*`` / ``vision_encoder.*``.

    Raises:
        ValueError: If a key belongs to neither tower.
    """
    adapted: dict[str, WeightData] = {}
    for key, weights in state_dict.items():
        data = weights.data()
        if key.startswith("model."):
            adapted[LANGUAGE_PREFIX + key.removeprefix("model.")] = data
        elif key.startswith("lm_head."):
            adapted[LANGUAGE_PREFIX + key] = data
        elif key.startswith("visual."):
            name = key.removeprefix("visual.")
            for old, new in _VISION_RENAMES:
                if name.startswith(old):
                    name = new + name.removeprefix(old)
            if name == "patch_embed.proj.weight":
                data = _conv3d_to_linear(data)
            adapted[VISION_PREFIX + name] = data
        else:
            raise ValueError(f"Unexpected Qwen2-VL checkpoint key: {key}")
    return adapted


def _conv3d_to_linear(data: WeightData) -> WeightData:
    """Views a ``[out, C, T, P, P]`` Conv3d kernel as ``[out, C*T*P*P]``.

    HF ``PatchEmbed`` runs ``nn.Conv3d`` with stride equal to its kernel on
    one pre-cut patch per row, which is a Linear over the flattened patch.
    """
    kernel = Buffer.from_dlpack(data.data)
    out_dim = kernel.shape[0]
    flat = kernel.view(
        dtype=kernel.dtype, shape=(out_dim, kernel.num_elements // out_dim)
    )
    return WeightData(
        data=flat,
        name=data.name,
        dtype=data.dtype,
        shape=data.shape.__class__(flat.shape),
        quantization_encoding=data.quantization_encoding,
    )
