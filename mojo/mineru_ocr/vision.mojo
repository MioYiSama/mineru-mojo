"""Qwen2-VL vision transformer and patch merger."""
from max.gpu.host import DeviceContext
from .config import MERGE_SIZE, MERGE_UNIT, MERGED_DIM, PATCH_DIM, VISION_LAYERS
from .tensor import Tensor
from .safetensors import Weights
from .dense import (
    linear,
    linear_bias,
    linear_bias_gelu,
    linear_bias_residual,
    layer_norm,
    upload,
)
from .attention import vision_attention


def _layer_norm(
    ctx: DeviceContext, x: Tensor, weights: Weights, name: String
) raises -> Tensor:
    return layer_norm(
        ctx, x, weights.get(name + ".weight"), weights.get(name + ".bias")
    )


def encode_vision(
    ctx: DeviceContext,
    weights: Weights,
    pixels: List[Float32],
    grid_h: Int,
    grid_w: Int,
) raises -> Tensor:
    """Encode one image's merge-group-ordered normalized patches.

    Returns one text-decoder-width embedding per merge group.
    """
    var count = grid_h * grid_w
    if (
        grid_h <= 0
        or grid_w <= 0
        or grid_h % MERGE_SIZE
        or grid_w % MERGE_SIZE
        or len(pixels) != count * PATCH_DIM
    ):
        raise Error("Invalid image patch grid")
    var patches = upload(ctx, pixels, count, PATCH_DIM)
    var h = linear(ctx, patches, weights.get("visual.patch_embed.proj.weight"))
    for layer in range(VISION_LAYERS):
        var p = "visual.blocks." + String(layer)
        var n1 = _layer_norm(ctx, h, weights, p + ".norm1")
        var qkv = linear_bias(
            ctx,
            n1,
            weights.get(p + ".attn.qkv.weight"),
            weights.get(p + ".attn.qkv.bias"),
        )
        var attended = vision_attention(ctx, qkv, grid_h, grid_w)
        h = linear_bias_residual(
            ctx,
            attended,
            weights.get(p + ".attn.proj.weight"),
            weights.get(p + ".attn.proj.bias"),
            h,
        )
        var n2 = _layer_norm(ctx, h, weights, p + ".norm2")
        var hidden = linear_bias_gelu[exact=False](
            ctx,
            n2,
            weights.get(p + ".mlp.fc1.weight"),
            weights.get(p + ".mlp.fc1.bias"),
        )
        h = linear_bias_residual(
            ctx,
            hidden,
            weights.get(p + ".mlp.fc2.weight"),
            weights.get(p + ".mlp.fc2.bias"),
            h,
        )
    # Each merge group's patches are consecutive rows, so concatenating a
    # group is a row-major reshape.
    var merged = _layer_norm(ctx, h, weights, "visual.merger.ln_q").reshape(
        count // MERGE_UNIT, MERGED_DIM
    )
    var expanded = linear_bias_gelu[exact=True](
        ctx,
        merged,
        weights.get("visual.merger.mlp.0.weight"),
        weights.get("visual.merger.mlp.0.bias"),
    )
    return linear_bias(
        ctx,
        expanded,
        weights.get("visual.merger.mlp.2.weight"),
        weights.get("visual.merger.mlp.2.bias"),
    )
