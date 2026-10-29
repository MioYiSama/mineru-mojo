"""Qwen2 decoder with contiguous per-layer KV storage and multimodal positions."""
from std.math import max
from max.gpu.host import DeviceContext
from .config import (
    CONTEXT_LENGTH,
    MERGE_SIZE,
    MERGE_UNIT,
    TEXT_HEAD_DIM,
    TEXT_KV_HEADS,
    TEXT_LAYERS,
)
from .tensor import Tensor
from .safetensors import Weights
from .dense import (
    linear,
    linear_bias,
    linear_residual,
    linear_swiglu,
    rms_norm,
    embed,
    last_row,
    argmax,
)
from .attention import text_attention


struct KVCache(Movable):
    """Rotated keys and values for one request, bounded by the context length.
    """

    var keys: List[Tensor]
    var values: List[Tensor]
    var capacity: Int
    var length: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        if capacity <= 0 or capacity > CONTEXT_LENGTH:
            raise Error("KV cache capacity must be between 1 and 8192")
        self.keys = List[Tensor](capacity=TEXT_LAYERS)
        self.values = List[Tensor](capacity=TEXT_LAYERS)
        self.capacity = capacity
        self.length = 0
        for _ in range(TEXT_LAYERS):
            self.keys.append(
                Tensor.empty(ctx, capacity, TEXT_KV_HEADS * TEXT_HEAD_DIM)
            )
            self.values.append(
                Tensor.empty(ctx, capacity, TEXT_KV_HEADS * TEXT_HEAD_DIM)
            )


def multimodal_positions(
    ids: List[Int], grid_h: Int, grid_w: Int, image_token: Int
) raises -> List[Int]:
    """HF get_rope_index for a single still image, in axis-major [3,T] order."""
    var count = grid_h * grid_w // MERGE_UNIT
    var merged_w = grid_w // MERGE_SIZE
    var start = -1
    for i in range(len(ids)):
        if ids[i] == image_token:
            start = i
            break
    if start < 0 or start + count > len(ids):
        raise Error("Prompt is missing its image placeholders")
    for i in range(count):
        if ids[start + i] != image_token:
            raise Error("Image placeholders must form one contiguous range")
    var result = List[Int](length=3 * len(ids), fill=0)
    var after = start + max(grid_h // MERGE_SIZE, merged_w)
    for axis in range(3):
        for i in range(len(ids)):
            var p = i
            if i >= start and i < start + count:
                var n = i - start
                p = start
                if axis == 1:
                    p += n // merged_w
                elif axis == 2:
                    p += n % merged_w
            elif i >= start + count:
                p = after + i - start - count
            result[axis * len(ids) + i] = p
    return result^


def prefill(
    ctx: DeviceContext,
    weights: Weights,
    mut cache: KVCache,
    ids: List[Int],
    positions: List[Int],
    image_embeddings: Tensor,
    image_token: Int,
) raises -> Int:
    """Run the prompt through the decoder, filling the cache; return the argmax.
    """
    if cache.length + len(ids) > cache.capacity or len(positions) != 3 * len(
        ids
    ):
        raise Error(
            "Request exceeds KV cache capacity or has invalid positions"
        )
    var h = embed(
        ctx,
        ids,
        weights.get("model.embed_tokens.weight"),
        image_embeddings,
        image_token,
    )
    for layer in range(TEXT_LAYERS):
        var p = "model.layers." + String(layer) + "."
        var n1 = rms_norm(ctx, h, weights.get(p + "input_layernorm.weight"))
        var q = linear_bias(
            ctx,
            n1,
            weights.get(p + "self_attn.q_proj.weight"),
            weights.get(p + "self_attn.q_proj.bias"),
        )
        var k = linear_bias(
            ctx,
            n1,
            weights.get(p + "self_attn.k_proj.weight"),
            weights.get(p + "self_attn.k_proj.bias"),
        )
        var v = linear_bias(
            ctx,
            n1,
            weights.get(p + "self_attn.v_proj.weight"),
            weights.get(p + "self_attn.v_proj.bias"),
        )
        var attended = text_attention(
            ctx,
            q,
            k,
            v,
            positions,
            cache.keys[layer],
            cache.values[layer],
            cache.length,
        )
        h = linear_residual(
            ctx, attended, weights.get(p + "self_attn.o_proj.weight"), h
        )
        var n2 = rms_norm(
            ctx, h, weights.get(p + "post_attention_layernorm.weight")
        )
        var gate = linear(ctx, n2, weights.get(p + "mlp.gate_proj.weight"))
        var hidden = linear_swiglu(
            ctx, n2, weights.get(p + "mlp.up_proj.weight"), gate
        )
        h = linear_residual(
            ctx, hidden, weights.get(p + "mlp.down_proj.weight"), h
        )
    cache.length += len(ids)
    var last = last_row(ctx, h)
    var normalized = rms_norm(ctx, last, weights.get("model.norm.weight"))
    var logits = linear(
        ctx, normalized, weights.get("model.embed_tokens.weight")
    )
    return argmax(ctx, logits)
