"""Register-tiled 8x8 simdgroup flash attention for Apple M1–M4.

Each simdgroup owns independent query rows; Q, scores, and the online
softmax numerator stay in registers. K/V are read directly through L2.
Depth-permuted Q/K and column-permuted V use contiguous vector loads;
the output store reverses the column permutation. The unmasked prefix
and ragged/causal tail have separate loops.
Scores, row maxima/sums, and the numerator use FP32; P is rounded to BF16
only for the P·V MMA. No quadratic intermediate or threadgroup barrier.
"""
from std.collections import Array
from std.math import ceildiv, exp2, max, min, sqrt
from std.utils import IndexList
from std.math.constants import log2e
from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp
from layout import TileTensor
from .tensor import BF16, MatrixLayout, Tensor

comptime _VISION_QUERY_TILE = 24
comptime _VISION_KEY_TILE = 48
comptime _TEXT_TILE = 16
comptime _FLASH_GROUPS = 4
comptime _NEG_INF = Float32(-3.0e38)


@inline(.always)
def _row_max(x: Float32) -> Float32:
    # Fragment rows share lanes differing only in bits 0 and 3.
    var y = max(x, warp.shuffle_xor(x, UInt32(1)))
    return max(y, warp.shuffle_xor(y, UInt32(8)))


@inline(.always)
def _row_sum(x: Float32) -> Float32:
    var y = x + warp.shuffle_xor(x, UInt32(1))
    return y + warp.shuffle_xor(y, UInt32(8))


def _flash_kernel[
    depth: Int,
    heads: Int,
    kv_heads: Int,
    causal: Bool,
](
    q: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    k: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    v: TileTensor[BF16, MatrixLayout, ImmutAnyOrigin],
    output: TileTensor[BF16, MatrixLayout, MutAnyOrigin],
    rows_arg: Int32,
    keys_arg: Int32,
    offset_arg: Int32,
    v_offset_arg: Int32,
):
    comptime assert q.flat_rank == 2 and k.flat_rank == 2
    comptime assert v.flat_rank == 2 and output.flat_rank == 2
    comptime assert depth % 16 == 0
    comptime assert WARP_SIZE == 32
    comptime tile_m = _TEXT_TILE if causal else _VISION_QUERY_TILE
    comptime tile_n = _TEXT_TILE if causal else _VISION_KEY_TILE
    comptime groups = _FLASH_GROUPS
    comptime k_width = 2 if causal else 8
    comptime v_width = 1 if causal else 8
    comptime MT = tile_m // 8
    comptime NT = tile_n // 8
    comptime DT = depth // 8
    # Fold the base conversion into scaling rather than each exponential.
    comptime scale = Float32(log2e) / sqrt(Float32(depth))
    # Image matrices fit 32-bit coordinates; preserve the text path's indexing.
    comptime index_dtype = DType.int64 if causal else DType.int32
    comptime IndexType = SIMD[index_dtype, 1]
    comptime Coord = IndexList[2, element_type=index_dtype]
    var rows = rows_arg.cast[index_dtype]()
    var keys = keys_arg.cast[index_dtype]()
    var offset = offset_arg.cast[index_dtype]()
    var v_offset = v_offset_arg.cast[index_dtype]()
    var lane = IndexType(thread_idx.x % WARP_SIZE)
    var sg = IndexType(thread_idx.x // WARP_SIZE)
    var fr = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fc = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var query_start = (
        IndexType(block_idx.x) * IndexType(groups) + sg
    ) * IndexType(tile_m)
    var head = IndexType(block_idx.y)
    var kv = head // IndexType(heads // kv_heads)
    if query_start >= rows:
        return
    var end = keys
    comptime if causal:
        end = min(end, offset + min(query_start + IndexType(tile_m), rows))
    var queries = Array[SIMD[BF16, 2], MT * DT](fill=SIMD[BF16, 2](0))
    comptime for mi in range(MT):
        if query_start + IndexType(mi * 8) + fr < rows:
            comptime for db in range(ceildiv(depth, 8 * k_width)):
                comptime base = db * 8 * k_width
                comptime kw = min(k_width, (depth - base) // 8)
                var run = q.load_linear[width=2 * kw, alignment=4 * kw](
                    Coord(
                        (
                            query_start + IndexType(mi * 8) + fr,
                            head * IndexType(depth)
                            + IndexType(base)
                            + fc * IndexType(kw),
                        )
                    )
                )
                comptime for s in range(kw):
                    queries[mi * DT + base // 8 + s] = SIMD[BF16, 2](
                        run[s], run[kw + s]
                    )
    var numerator = Array[SIMD[.float32, 2], MT * DT](fill=SIMD[.float32, 2](0))
    var maxima = Array[Float32, MT](fill=_NEG_INF)
    var sums = Array[Float32, MT](fill=Float32(0))
    var interior_end = end
    comptime if causal:
        interior_end = min(interior_end, offset + query_start + 1)
    interior_end -= interior_end % IndexType(tile_n)
    # Run the wholly unmasked prefix separately from the ragged/causal tail.
    comptime for pass_index in range(2):
        comptime mode = 1 - pass_index
        var begin = IndexType(0) if mode == 1 else interior_end
        var limit = interior_end if mode == 1 else end
        var kb = begin
        while kb < limit:
            var scores = Array[SIMD[.float32, 2], MT * NT](
                fill=SIMD[.float32, 2](0)
            )
            comptime for db in range(ceildiv(depth, 8 * k_width)):
                comptime base = db * 8 * k_width
                comptime kw = min(k_width, (depth - base) // 8)
                comptime for ni in range(NT):
                    var k0 = SIMD[BF16, kw](0)
                    var k1 = SIMD[BF16, kw](0)
                    if mode == 1 or kb + IndexType(ni * 8) + fc < end:
                        k0 = k.load_linear[width=kw, alignment=2 * kw](
                            Coord(
                                (
                                    kb + IndexType(ni * 8) + fc,
                                    kv * IndexType(depth)
                                    + IndexType(base)
                                    + fr * IndexType(kw),
                                )
                            )
                        )
                    if mode == 1 or kb + IndexType(ni * 8) + fc + 1 < end:
                        k1 = k.load_linear[width=kw, alignment=2 * kw](
                            Coord(
                                (
                                    kb + IndexType(ni * 8) + fc + 1,
                                    kv * IndexType(depth)
                                    + IndexType(base)
                                    + fr * IndexType(kw),
                                )
                            )
                        )
                    comptime for s in range(kw):
                        var bf = SIMD[BF16, 2](k0[s], k1[s])
                        comptime for mi in range(MT):
                            var previous = scores[mi * NT + ni]
                            _mma_apple_8x8(
                                scores[mi * NT + ni],
                                queries[mi * DT + base // 8 + s],
                                bf,
                                previous,
                            )
            var probabilities = Array[SIMD[BF16, 2], MT * NT](
                uninitialized=True
            )
            comptime for mi in range(MT):
                var row = query_start + IndexType(mi * 8) + fr
                var local_max = _NEG_INF
                comptime for ni in range(NT):
                    comptime if mode == 1:
                        scores[mi * NT + ni] *= scale
                    else:
                        comptime for e in range(2):
                            var key = kb + IndexType(ni * 8) + fc + IndexType(e)
                            var valid = row < rows and key < end
                            comptime if causal:
                                valid = valid and key <= offset + row
                            scores[mi * NT + ni][e] = (
                                scores[mi * NT + ni][e]
                                * scale if valid else _NEG_INF
                            )
                    local_max = max(
                        local_max, scores[mi * NT + ni].reduce_max()
                    )
                var new_max = max(maxima[mi], _row_max(local_max))
                var alpha = exp2(maxima[mi] - new_max)
                var local_sum = Float32(0)
                comptime for ni in range(NT):
                    var p = exp2(scores[mi * NT + ni] - new_max)
                    # Also zero entirely masked rows in a ragged query tail.
                    comptime if mode == 0:
                        comptime for e in range(2):
                            if scores[mi * NT + ni][e] == _NEG_INF:
                                p[e] = 0
                    local_sum += p.reduce_add()
                    probabilities[mi * NT + ni] = p.cast[BF16]()
                sums[mi] = sums[mi] * alpha + _row_sum(local_sum)
                maxima[mi] = new_max
                comptime for di in range(DT):
                    numerator[mi * DT + di] *= alpha
            comptime for ni in range(NT):
                comptime for db in range(ceildiv(depth, 8 * v_width)):
                    comptime base = db * 8 * v_width
                    comptime kw = min(v_width, (depth - base) // 8)
                    var run = SIMD[BF16, 2 * kw](0)
                    if mode == 1 or kb + IndexType(ni * 8) + fr < end:
                        run = v.load_linear[width=2 * kw, alignment=4 * kw](
                            Coord(
                                (
                                    kb + IndexType(ni * 8) + fr,
                                    v_offset
                                    + kv * IndexType(depth)
                                    + IndexType(base)
                                    + fc * IndexType(kw),
                                )
                            )
                        )
                    comptime for s in range(kw):
                        var vf = SIMD[BF16, 2](run[2 * s], run[2 * s + 1])
                        comptime for mi in range(MT):
                            var previous = numerator[mi * DT + base // 8 + s]
                            _mma_apple_8x8(
                                numerator[mi * DT + base // 8 + s],
                                probabilities[mi * NT + ni],
                                vf,
                                previous,
                            )
            kb += IndexType(tile_n)
    comptime for mi in range(MT):
        var row = query_start + IndexType(mi * 8) + fr
        if row < rows:
            comptime for db in range(ceildiv(depth, 8 * v_width)):
                comptime base = db * 8 * v_width
                comptime kw = min(v_width, (depth - base) // 8)
                var result = SIMD[BF16, 2 * kw](0)
                comptime for s in range(kw):
                    var pair = (
                        numerator[mi * DT + base // 8 + s] / sums[mi]
                    ).cast[BF16]()
                    result[2 * s] = pair[0]
                    result[2 * s + 1] = pair[1]
                output.store_linear[alignment=4 * kw](
                    Coord(
                        (
                            row,
                            head * IndexType(depth)
                            + IndexType(base)
                            + fc * IndexType(kw),
                        )
                    ),
                    result,
                )


def flash_attention[
    depth: Int, heads: Int, kv_heads: Int, causal: Bool
](
    ctx: DeviceContext,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Int,
    offset: Int,
    v_offset: Int = 0,
) raises -> Tensor:
    """Attend `q` [rows, heads*depth] over `keys` rows of K/V.

    `causal` lets query i see keys up to `offset + i`; V columns start at
    `v_offset`, so V may be a column block of a wider projection.
    """
    var output = Tensor.empty(ctx, q.rows, heads * depth)
    if q.rows > 0:
        comptime query_tile = _TEXT_TILE if causal else _VISION_QUERY_TILE
        comptime kernel = _flash_kernel[depth, heads, kv_heads, causal]
        ctx.enqueue_function[kernel](
            q.view().as_imm().as_unsafe_any_origin(),
            k.view().as_imm().as_unsafe_any_origin(),
            v.view().as_imm().as_unsafe_any_origin(),
            output.view().as_unsafe_any_origin(),
            Int32(q.rows),
            Int32(keys),
            Int32(offset),
            Int32(v_offset),
            grid_dim=(ceildiv(q.rows, query_tile * _FLASH_GROUPS), heads),
            block_dim=_FLASH_GROUPS * WARP_SIZE,
        )
    return output^
