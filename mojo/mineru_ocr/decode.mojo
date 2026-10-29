"""Allocation-free single-token decoder, preserving BF16 graph boundaries.

Every fused kernel retains the unfused BF16 rounding points. Decode positions
are identical on all three M-RoPE axes; prefill lives in text.mojo.
"""
from std.math import ceildiv, cos, sin, pow, rsqrt, exp, max, min, sqrt
from max.gpu import WARP_SIZE, global_idx, lane_id, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.primitives import warp
from layout import Coord, TileTensor, row_major
from linalg.matmul.gpu.apple.gemv import enqueue_apple_gemv_config
from nn.activations import silu
from nn.fused_qk_rope import rope_value
from .tensor import Tensor, BF16, MatrixLayout
from .safetensors import Weights
from .config import (
    CONTEXT_LENGTH,
    TEXT_HEAD_DIM,
    TEXT_HEADS,
    TEXT_HIDDEN,
    TEXT_INTERMEDIATE,
    TEXT_KV_HEADS,
    TEXT_LAYERS,
    TEXT_ROPE_THETA,
    VOCAB_SIZE,
)
from .dense import ARGMAX_GROUP, _argmax_partials

comptime _Input = TileTensor[BF16, MatrixLayout, ImmutAnyOrigin]
comptime _Output = TileTensor[BF16, MatrixLayout, MutAnyOrigin]
comptime _Indices = TileTensor[.int64, MatrixLayout, MutAnyOrigin]
comptime _FloatInput = TileTensor[.float32, MatrixLayout, ImmutAnyOrigin]
comptime _IndexInput = TileTensor[.int64, MatrixLayout, ImmutAnyOrigin]
comptime _FloatOutput = TileTensor[.float32, MatrixLayout, MutAnyOrigin]

comptime _BLOCK = 256
comptime _WARPS = _BLOCK // WARP_SIZE
comptime _VECTOR = 8
comptime _HALF = TEXT_HEAD_DIM // 2
"""`rotate_half` pairs channel d with d + _HALF; one warp owns a pair column."""
comptime _GROUP = TEXT_HEADS // TEXT_KV_HEADS
comptime _Q_PAIRS = TEXT_HEADS * _HALF
comptime _KV_PAIRS = TEXT_KV_HEADS * _HALF
comptime _PAIRS = _Q_PAIRS + 2 * _KV_PAIRS
comptime _KEY_SPLIT = WARP_SIZE
"""Keys per decode-attention partition: one warp, one key per lane."""
comptime _PARTITIONS = CONTEXT_LENGTH // _KEY_SPLIT
comptime _PARTIAL_COLS = TEXT_HEAD_DIM + 2
"""Per head: numerator channels, then running maximum and denominator."""
comptime _ARGMAX_GROUPS = ceildiv(VOCAB_SIZE, ARGMAX_GROUP)


def _attention_partial(
    q: _Input,
    keys: _Input,
    values: _Input,
    partial: _FloatOutput,
    length_arg: Int32,
):
    # A warp owns one KV head and one key split, shared by its whole query
    # group: each lane scores one key for every grouped head, then lanes
    # switch to owning two value channels for the probability-weighted sum.
    comptime assert (
        q.flat_rank == 2
        and keys.flat_rank == 2
        and values.flat_rank == 2
        and partial.flat_rank == 2
    )
    comptime assert TEXT_HEAD_DIM == 2 * WARP_SIZE
    comptime assert _GROUP <= _VECTOR
    comptime chunks = TEXT_HEAD_DIM // _VECTOR
    comptime scale = Float32(1) / sqrt(Float32(TEXT_HEAD_DIM))
    var kv = block_idx.y
    var partition = global_idx.x // WARP_SIZE
    var begin = partition * _KEY_SPLIT
    var length = Int(length_arg)
    if begin >= length:
        return
    var lane = lane_id()
    var valid = begin + lane < length
    # Out-of-range lanes reread the last key and are masked below.
    var key = min(begin + lane, length - 1)
    var key_row = Array[SIMD[.float32, _VECTOR], chunks](uninitialized=True)
    comptime for j in range(chunks):
        key_row[j] = keys.load[width=_VECTOR](
            Coord(key, kv * TEXT_HEAD_DIM + j * _VECTOR)
        ).cast[.float32]()
    var probabilities = SIMD[.float32, _VECTOR](0)
    var maxima = SIMD[.float32, _VECTOR](0)
    var denominators = SIMD[.float32, _VECTOR](0)
    comptime for h in range(_GROUP):
        var head = kv * _GROUP + h
        var score: Float32 = 0
        comptime for j in range(chunks):
            score += (
                q.load[width=_VECTOR](
                    Coord(0, head * TEXT_HEAD_DIM + j * _VECTOR)
                ).cast[.float32]()
                * key_row[j]
            ).reduce_add()
        score = score * scale if valid else Float32(-3.0e38)
        var maximum = warp.max(score)
        var probability = exp(score - maximum) if valid else Float32(0)
        probabilities[h] = probability
        maxima[h] = maximum
        denominators[h] = warp.sum(probability)
    var channel = kv * TEXT_HEAD_DIM + lane
    var first = SIMD[.float32, _VECTOR](0)
    var second = SIMD[.float32, _VECTOR](0)
    for j in range(min(_KEY_SPLIT, length - begin)):
        var weight = SIMD[.float32, _VECTOR](0)
        comptime for h in range(_GROUP):
            weight[h] = warp.shuffle_idx(probabilities[h], UInt32(j))
        first += weight * values[begin + j, channel].cast[.float32]()
        second += (
            weight * values[begin + j, channel + WARP_SIZE].cast[.float32]()
        )
    comptime for h in range(_GROUP):
        var col = (kv * _GROUP + h) * _PARTIAL_COLS
        partial[partition, col + lane] = first[h]
        partial[partition, col + lane + WARP_SIZE] = second[h]
        if lane == 0:
            partial[partition, col + TEXT_HEAD_DIM] = maxima[h]
            partial[partition, col + TEXT_HEAD_DIM + 1] = denominators[h]


def _attention_finish(
    partial: _FloatInput, output: _Output, partitions_arg: Int32
):
    comptime assert partial.flat_rank == 2 and output.flat_rank == 2
    var c = global_idx.x
    if c >= TEXT_HEADS * TEXT_HEAD_DIM:
        return
    var col = c // TEXT_HEAD_DIM * _PARTIAL_COLS
    var maximum: Float32 = -3.0e38
    var denominator: Float32 = 0
    var numerator: Float32 = 0
    for p in range(Int(partitions_arg)):
        var local_max = partial[p, col + TEXT_HEAD_DIM]
        var next_max = max(maximum, local_max)
        var alpha = exp(maximum - next_max)
        var beta = exp(local_max - next_max)
        numerator = (
            numerator * alpha + partial[p, col + c % TEXT_HEAD_DIM] * beta
        )
        denominator = (
            denominator * alpha + partial[p, col + TEXT_HEAD_DIM + 1] * beta
        )
        maximum = next_max
    output[0, c] = (numerator / denominator).cast[BF16]()


@inline(.always)
def _dot(x: _Input, w: _Input, row: Int, k: Int) -> Float32:
    comptime assert x.flat_rank == 2 and w.flat_rank == 2
    var acc = SIMD[.float32, _VECTOR](0)
    var chunk = lane_id()
    while chunk < k // _VECTOR:
        acc += (
            x.load[width=_VECTOR](Coord(0, chunk * _VECTOR)).cast[.float32]()
            * w.load[width=_VECTOR](Coord(row, chunk * _VECTOR)).cast[
                .float32
            ]()
        )
        chunk += WARP_SIZE
    return warp.sum(acc.reduce_add())


@inline(.always)
def _rms_scale(x: _Input) -> Float32:
    comptime assert x.flat_rank == 2
    var total: Float32 = 0
    for c in range(lane_id(), TEXT_HIDDEN, WARP_SIZE):
        var value = x[0, c].cast[.float32]()
        total += value * value
    return rsqrt(warp.sum(total) / Float32(TEXT_HIDDEN) + 1.0e-6)


@inline(.always)
def _normalized_pair(
    x: _Input,
    first_weight: _Input,
    second_weight: _Input,
    norm: _Input,
    first_row: Int,
    second_row: Int,
    scale: Float32,
) -> Tuple[Float32, Float32]:
    """Dot two weight rows with the BF16-rounded RMSNorm of `x`."""
    comptime assert (
        x.flat_rank == 2
        and first_weight.flat_rank == 2
        and second_weight.flat_rank == 2
        and norm.flat_rank == 2
    )
    var first = SIMD[.float32, _VECTOR](0)
    var second = SIMD[.float32, _VECTOR](0)
    var chunk = lane_id()
    while chunk < TEXT_HIDDEN // _VECTOR:
        var c = chunk * _VECTOR
        var activation = (
            (
                x.load[width=_VECTOR](Coord(0, c)).cast[.float32]()
                * scale
                * norm.load[width=_VECTOR](Coord(0, c)).cast[.float32]()
            )
            .cast[BF16]()
            .cast[.float32]()
        )
        first += (
            activation
            * first_weight.load[width=_VECTOR](Coord(first_row, c)).cast[
                .float32
            ]()
        )
        second += (
            activation
            * second_weight.load[width=_VECTOR](Coord(second_row, c)).cast[
                .float32
            ]()
        )
        chunk += WARP_SIZE
    return warp.sum(first.reduce_add()), warp.sum(second.reduce_add())


def _norm(x: _Input, weight: _Input, dest: _Output):
    comptime assert (
        x.flat_rank == 2 and weight.flat_rank == 2 and dest.flat_rank == 2
    )
    var scale = _rms_scale(x)
    for c in range(lane_id(), TEXT_HIDDEN, WARP_SIZE):
        dest[0, c] = (
            x[0, c].cast[.float32]() * scale * weight[0, c].cast[.float32]()
        ).cast[BF16]()


def _embed(tokens: _Indices, weight: _Input, dest: _Output, step_arg: Int32):
    comptime assert (
        tokens.flat_rank == 2 and weight.flat_rank == 2 and dest.flat_rank == 2
    )
    var c = global_idx.x
    if c < TEXT_HIDDEN:
        dest[0, c] = weight[Int(tokens[0, Int(step_arg)]), c]


def _qkv(
    x: _Input,
    norm: _Input,
    qw: _Input,
    kw: _Input,
    vw: _Input,
    qb: _Input,
    kb: _Input,
    vb: _Input,
    q: _Output,
    keys: _Output,
    values: _Output,
    offset_arg: Int32,
    position_arg: Int32,
):
    """RMSNorm + Q/K/V GEMV + bias + RoPE; K/V land in the cache row."""
    comptime assert (
        x.flat_rank == 2
        and qw.flat_rank == 2
        and kw.flat_rank == 2
        and vw.flat_rank == 2
    )
    comptime assert (
        qb.flat_rank == 2 and kb.flat_rank == 2 and vb.flat_rank == 2
    )
    comptime assert (
        q.flat_rank == 2 and keys.flat_rank == 2 and values.flat_rank == 2
    )
    # Pair p covers channels (c, c + _HALF): Q pairs, then K, then V.
    var pair = global_idx.x // WARP_SIZE
    if pair >= _PAIRS:
        return
    var w = qw
    var b = qb
    var local = pair
    if pair >= _Q_PAIRS + _KV_PAIRS:
        w = vw
        b = vb
        local = pair - _Q_PAIRS - _KV_PAIRS
    elif pair >= _Q_PAIRS:
        w = kw
        b = kb
        local = pair - _Q_PAIRS
    var frequency = local % _HALF
    var channel = local // _HALF * TEXT_HEAD_DIM + frequency
    var scale = _rms_scale(x)
    var first_dot, second_dot = _normalized_pair(
        x, w, w, norm, channel, channel + _HALF, scale
    )
    var first = (
        first_dot.cast[BF16]().cast[.float32]() + b[0, channel].cast[.float32]()
    ).cast[BF16]()
    var second = (
        second_dot.cast[BF16]().cast[.float32]()
        + b[0, channel + _HALF].cast[.float32]()
    ).cast[BF16]()
    if lane_id() == 0:
        var offset = Int(offset_arg)
        if pair >= _Q_PAIRS + _KV_PAIRS:
            values[offset, channel] = first
            values[offset, channel + _HALF] = second
        else:
            var angle = Float32(position_arg) * pow(
                TEXT_ROPE_THETA, -Float32(frequency) / Float32(_HALF)
            )
            var rotated = rope_value(
                SIMD[BF16, 2](first, second),
                SIMD[BF16, 2](cos(angle).cast[BF16](), sin(angle).cast[BF16]()),
            )
            if pair < _Q_PAIRS:
                q[0, channel] = rotated[0]
                q[0, channel + _HALF] = rotated[1]
            else:
                keys[offset, channel] = rotated[0]
                keys[offset, channel + _HALF] = rotated[1]


def _residual(x: _Input, weight: _Input, h: _Output, k_arg: Int32):
    """GEMV `x @ weight.T` added in place to the residual stream `h`."""
    comptime assert (
        x.flat_rank == 2 and weight.flat_rank == 2 and h.flat_rank == 2
    )
    var row = global_idx.x // WARP_SIZE
    if row < TEXT_HIDDEN:
        var value = _dot(x, weight, row, Int(k_arg)).cast[BF16]()
        if lane_id() == 0:
            h[0, row] = (
                h[0, row].cast[.float32]() + value.cast[.float32]()
            ).cast[BF16]()


def _gate_up(x: _Input, norm: _Input, gate: _Input, up: _Input, dest: _Output):
    """RMSNorm + gate/up GEMV + SwiGLU for one intermediate channel per warp."""
    comptime assert (
        x.flat_rank == 2
        and gate.flat_rank == 2
        and up.flat_rank == 2
        and dest.flat_rank == 2
    )
    var row = global_idx.x // WARP_SIZE
    if row < TEXT_INTERMEDIATE:
        var scale = _rms_scale(x)
        var first_dot, second_dot = _normalized_pair(
            x, gate, up, norm, row, row, scale
        )
        var g = first_dot.cast[BF16]()
        var u = second_dot.cast[BF16]()
        if lane_id() == 0:
            dest[0, row] = (silu(g).cast[.float32]() * u.cast[.float32]()).cast[
                BF16
            ]()


def _argmax_finish(
    maxima: _FloatInput,
    indices: _IndexInput,
    tokens: _Indices,
    groups_arg: Int32,
    step_arg: Int32,
):
    comptime assert (
        maxima.flat_rank == 2
        and indices.flat_rank == 2
        and tokens.flat_rank == 2
    )
    var best: Float32 = -3.0e38
    var best_index = Int64.MAX
    for i in range(lane_id(), Int(groups_arg), WARP_SIZE):
        var value = maxima[0, i]
        var index = indices[0, i]
        if value > best or (value == best and index < best_index):
            best = value
            best_index = index
    var maximum = warp.max(best)
    if best != maximum:
        best_index = Int64.MAX
    var index = warp.min(best_index)
    if lane_id() == 0:
        tokens[0, Int(step_arg)] = index


struct DecodeState(Movable):
    """Request-local scratch and resolved layer indices; no per-step allocations.
    """

    var h: Tensor
    var normalized: Tensor
    var q: Tensor
    var attended: Tensor
    var hidden: Tensor
    var logits: Tensor
    var partial: DeviceBuffer[.float32]
    var maxima: DeviceBuffer[.float32]
    var indices: DeviceBuffer[.int64]
    var tokens: DeviceBuffer[.int64]
    """Token slot s feeds decode step s, which writes slot s + 1."""
    var weight_indices: List[Int]
    var embedding_index: Int
    var final_norm_index: Int
    var token_capacity: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        weights: Weights,
        capacity: Int,
        first_token: Int,
    ) raises:
        self.h = Tensor.empty(ctx, 1, TEXT_HIDDEN)
        self.normalized = Tensor.empty(ctx, 1, TEXT_HIDDEN)
        self.q = Tensor.empty(ctx, 1, TEXT_HEADS * TEXT_HEAD_DIM)
        self.attended = Tensor.empty(ctx, 1, TEXT_HEADS * TEXT_HEAD_DIM)
        self.hidden = Tensor.empty(ctx, 1, TEXT_INTERMEDIATE)
        self.logits = Tensor.empty(ctx, 1, VOCAB_SIZE)
        self.partial = ctx.enqueue_create_buffer[.float32](
            _PARTITIONS * TEXT_HEADS * _PARTIAL_COLS
        )
        self.maxima = ctx.enqueue_create_buffer[.float32](_ARGMAX_GROUPS)
        self.indices = ctx.enqueue_create_buffer[.int64](_ARGMAX_GROUPS)
        self.tokens = ctx.enqueue_create_buffer[.int64](capacity)
        self.token_capacity = capacity
        self.tokens.enqueue_fill(Int64(first_token))
        self.embedding_index = weights.index["model.embed_tokens.weight"]
        self.final_norm_index = weights.index["model.norm.weight"]
        var suffixes: List[String] = [
            ".input_layernorm.weight",
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.q_proj.bias",
            ".self_attn.k_proj.bias",
            ".self_attn.v_proj.bias",
            ".self_attn.o_proj.weight",
            ".post_attention_layernorm.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ]
        self.weight_indices = List[Int](capacity=TEXT_LAYERS * len(suffixes))
        for layer in range(TEXT_LAYERS):
            for suffix in suffixes:
                self.weight_indices.append(
                    weights.index["model.layers." + String(layer) + suffix]
                )

    def enqueue(
        mut self,
        ctx: DeviceContext,
        weights: Weights,
        mut keys: List[Tensor],
        mut values: List[Tensor],
        offset: Int,
        position: Int,
        step: Int,
    ) raises:
        """Enqueue decode step `step` at cache row `offset` and `position`."""
        var token_view = TileTensor(
            self.tokens, row_major(Int32(1), Int32(self.token_capacity))
        )
        ctx.enqueue_function[_embed](
            token_view,
            weights.tensors[self.embedding_index].view(),
            self.h.view(),
            Int32(step),
            grid_dim=ceildiv(TEXT_HIDDEN, _BLOCK),
            block_dim=_BLOCK,
        )
        var per_layer = len(self.weight_indices) // TEXT_LAYERS
        for layer in range(TEXT_LAYERS):
            var base = layer * per_layer
            ctx.enqueue_function[_qkv](
                self.h.view(),
                weights.tensors[self.weight_indices[base]].view(),
                weights.tensors[self.weight_indices[base + 1]].view(),
                weights.tensors[self.weight_indices[base + 2]].view(),
                weights.tensors[self.weight_indices[base + 3]].view(),
                weights.tensors[self.weight_indices[base + 4]].view(),
                weights.tensors[self.weight_indices[base + 5]].view(),
                weights.tensors[self.weight_indices[base + 6]].view(),
                self.q.view(),
                keys[layer].view(),
                values[layer].view(),
                Int32(offset),
                Int32(position),
                grid_dim=ceildiv(_PAIRS, _WARPS),
                block_dim=_BLOCK,
            )
            self._attention(ctx, keys[layer], values[layer], offset + 1)
            ctx.enqueue_function[_residual](
                self.attended.view(),
                weights.tensors[self.weight_indices[base + 7]].view(),
                self.h.view(),
                Int32(TEXT_HEADS * TEXT_HEAD_DIM),
                grid_dim=ceildiv(TEXT_HIDDEN, _WARPS),
                block_dim=_BLOCK,
            )
            ctx.enqueue_function[_gate_up](
                self.h.view(),
                weights.tensors[self.weight_indices[base + 8]].view(),
                weights.tensors[self.weight_indices[base + 9]].view(),
                weights.tensors[self.weight_indices[base + 10]].view(),
                self.hidden.view(),
                grid_dim=ceildiv(TEXT_INTERMEDIATE, _WARPS),
                block_dim=_BLOCK,
            )
            ctx.enqueue_function[_residual](
                self.hidden.view(),
                weights.tensors[self.weight_indices[base + 11]].view(),
                self.h.view(),
                Int32(TEXT_INTERMEDIATE),
                grid_dim=ceildiv(TEXT_HIDDEN, _WARPS),
                block_dim=_BLOCK,
            )
        ctx.enqueue_function[_norm](
            self.h.view(),
            weights.tensors[self.final_norm_index].view(),
            self.normalized.view(),
            grid_dim=1,
            block_dim=WARP_SIZE,
        )
        enqueue_apple_gemv_config[tile_m=1, rows_per_warp=1, unroll=4](
            self.logits.view(),
            self.normalized.view(),
            weights.tensors[self.embedding_index].view(),
            ctx,
        )
        var maxima_view = TileTensor(
            self.maxima, row_major(Int32(1), Int32(_ARGMAX_GROUPS))
        )
        var indices_view = TileTensor(
            self.indices, row_major(Int32(1), Int32(_ARGMAX_GROUPS))
        )
        ctx.enqueue_function[_argmax_partials](
            self.logits.view(),
            maxima_view,
            indices_view,
            Int32(VOCAB_SIZE),
            Int32(VOCAB_SIZE),
            grid_dim=ceildiv(_ARGMAX_GROUPS, _WARPS),
            block_dim=_BLOCK,
        )
        ctx.enqueue_function[_argmax_finish](
            maxima_view.as_imm(),
            indices_view.as_imm(),
            token_view,
            Int32(_ARGMAX_GROUPS),
            Int32(step + 1),
            grid_dim=1,
            block_dim=WARP_SIZE,
        )

    def _attention(
        mut self, ctx: DeviceContext, keys: Tensor, values: Tensor, length: Int
    ) raises:
        var partitions = ceildiv(length, _KEY_SPLIT)
        var partial = TileTensor(
            self.partial,
            row_major(Int32(_PARTITIONS), Int32(TEXT_HEADS * _PARTIAL_COLS)),
        )
        ctx.enqueue_function[_attention_partial](
            self.q.view(),
            keys.view(),
            values.view(),
            partial,
            Int32(length),
            grid_dim=(ceildiv(partitions, _WARPS), TEXT_KV_HEADS),
            block_dim=_BLOCK,
        )
        ctx.enqueue_function[_attention_finish](
            partial.as_imm(),
            self.attended.view(),
            Int32(partitions),
            grid_dim=ceildiv(TEXT_HEADS * TEXT_HEAD_DIM, _BLOCK),
            block_dim=_BLOCK,
        )
