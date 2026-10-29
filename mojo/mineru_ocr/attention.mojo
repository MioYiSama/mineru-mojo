"""Rotary embeddings feeding BF16 vision and causal, cached text attention.

Four-wide RoPE preserves the model's BF16 rounding points; image values are
read directly from the QKV projection instead of copied. Kernel dimensions
use Int32 for Metal argument encoding.
"""

from std.math import ceildiv, cos, pow, sin
from std.utils import IndexList
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.fused_qk_rope import rope_value
from .config import (
    MERGE_SIZE,
    MERGE_UNIT,
    MROPE_HEIGHT_PAIRS,
    MROPE_TEMPORAL_PAIRS,
    TEXT_HEADS,
    TEXT_HEAD_DIM,
    TEXT_KV_HEADS,
    TEXT_ROPE_THETA,
    VISION_HEADS,
    VISION_HEAD_DIM,
    VISION_HIDDEN,
    VISION_ROPE_THETA,
)
from .tensor import BF16, MatrixLayout, Tensor
from .flash_attention import flash_attention

comptime _THREADS = 128
# `rotate_half` pairs channel d with d + half within each head.
comptime _VISION_HALF = VISION_HEAD_DIM // 2
# The first half of the vision frequencies rotate by row, the second by column.
comptime _VISION_AXIS_PAIRS = _VISION_HALF // 2
comptime _TEXT_HALF = TEXT_HEAD_DIM // 2
comptime _TEXT_ROTATED_HEADS = TEXT_HEADS + TEXT_KV_HEADS


def _vision_rope(
    qkv: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    q: TileTensor[BF16, MatrixLayout, MutAnyOrigin],
    k: TileTensor[BF16, MatrixLayout, MutAnyOrigin],
    tokens_arg: Int32,
    grid_w_arg: Int32,
):
    comptime assert qkv.flat_rank == 2 and q.flat_rank == 2
    comptime assert k.flat_rank == 2
    var tokens = Int(tokens_arg)
    var grid_w = Int(grid_w_arg)
    var idx = global_idx.x * 4
    if idx >= tokens * VISION_HEADS * _VISION_HALF:
        return
    var token = idx // (VISION_HEADS * _VISION_HALF)
    var head = idx // _VISION_HALF % VISION_HEADS
    var dim = idx % _VISION_HALF
    # Patches arrive merge-group-major: MERGE_UNIT consecutive rows form one
    # MERGE_SIZE x MERGE_SIZE block of the patch grid.
    var group = token // MERGE_UNIT
    var within = token % MERGE_UNIT
    var merged_w = grid_w // MERGE_SIZE
    var row = (group // merged_w) * MERGE_SIZE + within // MERGE_SIZE
    var col = (group % merged_w) * MERGE_SIZE + within % MERGE_SIZE
    var position = row if dim < _VISION_AXIS_PAIRS else col
    var base = Float32(dim % _VISION_AXIS_PAIRS)
    var dims = SIMD[.float32, 4](base, base + 1, base + 2, base + 3)
    var frequency = pow(
        SIMD[.float32, 4](VISION_ROPE_THETA), -dims / _VISION_AXIS_PAIRS
    )
    var angle = Float32(position) * frequency
    var c = cos(angle)
    var s = sin(angle)
    var channel = head * VISION_HEAD_DIM + dim
    var k_channel = VISION_HIDDEN + channel
    var qr = qkv.load_linear[width=4, alignment=8](
        IndexList[2](token, channel)
    ).cast[.float32]()
    var qi = qkv.load_linear[width=4, alignment=8](
        IndexList[2](token, channel + _VISION_HALF)
    ).cast[.float32]()
    var kr = qkv.load_linear[width=4, alignment=8](
        IndexList[2](token, k_channel)
    ).cast[.float32]()
    var ki = qkv.load_linear[width=4, alignment=8](
        IndexList[2](token, k_channel + _VISION_HALF)
    ).cast[.float32]()
    q.store_linear[alignment=8](
        IndexList[2](token, channel), (qr * c - qi * s).cast[BF16]()
    )
    q.store_linear[alignment=8](
        IndexList[2](token, channel + _VISION_HALF),
        (qi * c + qr * s).cast[BF16](),
    )
    k.store_linear[alignment=8](
        IndexList[2](token, channel), (kr * c - ki * s).cast[BF16]()
    )
    k.store_linear[alignment=8](
        IndexList[2](token, channel + _VISION_HALF),
        (ki * c + kr * s).cast[BF16](),
    )


def _text_rope_store(
    q: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    k: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    v: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    positions: TileTensor[.int64, MatrixLayout, ImmutAnyOrigin],
    rotated_q: TileTensor[BF16, MatrixLayout, MutAnyOrigin],
    key_cache: TileTensor[BF16, MatrixLayout, MutAnyOrigin],
    value_cache: TileTensor[BF16, MatrixLayout, MutAnyOrigin],
    rows_arg: Int32,
    offset_arg: Int32,
):
    comptime assert q.flat_rank == 2 and k.flat_rank == 2
    comptime assert v.flat_rank == 2 and positions.flat_rank == 2
    comptime assert rotated_q.flat_rank == 2
    comptime assert key_cache.flat_rank == 2 and value_cache.flat_rank == 2
    var rows = Int(rows_arg)
    var offset = Int(offset_arg)
    var idx = global_idx.x * 4
    if idx >= rows * _TEXT_ROTATED_HEADS * _TEXT_HALF:
        return
    var row = idx // (_TEXT_ROTATED_HEADS * _TEXT_HALF)
    var head = idx // _TEXT_HALF % _TEXT_ROTATED_HEADS
    var dim = idx % _TEXT_HALF
    var axis = 0
    if dim >= MROPE_TEMPORAL_PAIRS + MROPE_HEIGHT_PAIRS:
        axis = 2
    elif dim >= MROPE_TEMPORAL_PAIRS:
        axis = 1
    var base = Float32(dim)
    var dims = SIMD[.float32, 4](base, base + 1, base + 2, base + 3)
    var angle = positions[axis, row].cast[.float32]() * pow(
        SIMD[.float32, 4](TEXT_ROPE_THETA), -dims / _TEXT_HALF
    )
    # Preserve BF16 coefficient and complex-arithmetic rounding.
    var coefficients = (
        cos(angle).cast[BF16]().interleave(sin(angle).cast[BF16]())
    )
    if head < TEXT_HEADS:
        var channel = head * TEXT_HEAD_DIM + dim
        var real = q.load_linear[width=4, alignment=8](
            IndexList[2](row, channel)
        )
        var imag = q.load_linear[width=4, alignment=8](
            IndexList[2](row, channel + _TEXT_HALF)
        )
        var real_out, imag_out = rope_value(
            real.interleave(imag), coefficients
        ).deinterleave()
        rotated_q.store_linear[alignment=8](
            IndexList[2](row, channel), real_out
        )
        rotated_q.store_linear[alignment=8](
            IndexList[2](row, channel + _TEXT_HALF), imag_out
        )
    else:
        var channel = (head - TEXT_HEADS) * TEXT_HEAD_DIM + dim
        var real = k.load_linear[width=4, alignment=8](
            IndexList[2](row, channel)
        )
        var imag = k.load_linear[width=4, alignment=8](
            IndexList[2](row, channel + _TEXT_HALF)
        )
        var real_out, imag_out = rope_value(
            real.interleave(imag), coefficients
        ).deinterleave()
        key_cache.store_linear[alignment=8](
            IndexList[2](offset + row, channel), real_out
        )
        key_cache.store_linear[alignment=8](
            IndexList[2](offset + row, channel + _TEXT_HALF), imag_out
        )
        value_cache.store_linear[alignment=8](
            IndexList[2](offset + row, channel),
            v.load_linear[width=4, alignment=8](IndexList[2](row, channel)),
        )
        value_cache.store_linear[alignment=8](
            IndexList[2](offset + row, channel + _TEXT_HALF),
            v.load_linear[width=4, alignment=8](
                IndexList[2](row, channel + _TEXT_HALF)
            ),
        )


def vision_attention(
    ctx: DeviceContext,
    qkv: Tensor,
    grid_h: Int,
    grid_w: Int,
) raises -> Tensor:
    """Apply merge-ordered 2-D RoPE and full bidirectional image attention."""
    var tokens = grid_h * grid_w
    var q = Tensor.empty(ctx, tokens, VISION_HIDDEN)
    var k = Tensor.empty(ctx, tokens, VISION_HIDDEN)
    if tokens > 0:
        ctx.enqueue_function[_vision_rope](
            qkv.view().as_imm().as_unsafe_any_origin(),
            q.view().as_unsafe_any_origin(),
            k.view().as_unsafe_any_origin(),
            Int32(tokens),
            Int32(grid_w),
            grid_dim=ceildiv(
                tokens * VISION_HEADS * _VISION_HALF, _THREADS * 4
            ),
            block_dim=_THREADS,
        )
    # V is read in place from the projection's third column block.
    return flash_attention[
        VISION_HEAD_DIM, VISION_HEADS, VISION_HEADS, causal=False
    ](ctx, q, k, qkv, tokens, 0, 2 * VISION_HIDDEN)


def text_attention(
    ctx: DeviceContext,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    positions: List[Int],
    mut key_cache: Tensor,
    mut value_cache: Tensor,
    offset: Int,
) raises -> Tensor:
    """Rotate Q/K with M-RoPE, append contiguous KV, then run causal GQA."""
    var rotated_q = Tensor.empty(ctx, q.rows, TEXT_HEADS * TEXT_HEAD_DIM)
    if q.rows > 0:
        var host_positions = ctx.enqueue_create_host_buffer[.int64](3 * q.rows)
        for i in range(3 * q.rows):
            host_positions[i] = Int64(positions[i])
        var device_positions = ctx.enqueue_create_buffer[.int64](3 * q.rows)
        ctx.enqueue_copy(device_positions, host_positions)
        var position_view = TileTensor(
            device_positions, row_major(Int32(3), Int32(q.rows))
        )
        ctx.enqueue_function[_text_rope_store](
            q.view().as_imm().as_unsafe_any_origin(),
            k.view().as_imm().as_unsafe_any_origin(),
            v.view().as_imm().as_unsafe_any_origin(),
            position_view.as_imm().as_unsafe_any_origin(),
            rotated_q.view().as_unsafe_any_origin(),
            key_cache.view().as_unsafe_any_origin(),
            value_cache.view().as_unsafe_any_origin(),
            Int32(q.rows),
            Int32(offset),
            grid_dim=ceildiv(
                q.rows * _TEXT_ROTATED_HEADS * _TEXT_HALF, _THREADS * 4
            ),
            block_dim=_THREADS,
        )
    return flash_attention[
        TEXT_HEAD_DIM, TEXT_HEADS, TEXT_KV_HEADS, causal=True
    ](ctx, rotated_q, key_cache, value_cache, offset + q.rows, offset)
