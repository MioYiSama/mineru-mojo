"""BF16 dense kernels with Int32 device arguments and dynamic layout dimensions."""
from std.math import ceildiv, rsqrt
from std.utils.numerics import min_or_neg_inf
from max.gpu import WARP_SIZE, global_idx, lane_id
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp
from layout import Coord, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.apple import enqueue_apple_matmul_8x8
from linalg.matmul.gpu.apple.gemv import enqueue_apple_gemv_config
from linalg.utils import ElementwiseComputeFn
from nn.activations import gelu, gelu_quick, silu
from .tensor import Tensor, BF16, MatrixLayout

comptime _Input = TileTensor[BF16, MatrixLayout, ImmutAnyOrigin]
comptime _Output = TileTensor[BF16, MatrixLayout, MutAnyOrigin]
comptime _FloatInput = TileTensor[DType.float32, MatrixLayout, ImmutAnyOrigin]
comptime _FloatOutput = TileTensor[DType.float32, MatrixLayout, MutAnyOrigin]
comptime _IndexInput = TileTensor[DType.int64, MatrixLayout, ImmutAnyOrigin]
comptime _IndexOutput = TileTensor[DType.int64, MatrixLayout, MutAnyOrigin]
comptime _BLOCK = 256
# Elements per lane per vector access in the row kernels.
comptime _VECTOR = 8
comptime ARGMAX_GROUP = 1024
"""Elements one warp reduces in the first argmax pass."""


def _pad_columns(
    source: _Input,
    dest: _Output,
    rows_arg: Int32,
    cols_arg: Int32,
    padded_arg: Int32,
):
    comptime assert source.flat_rank == 2 and dest.flat_rank == 2
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var padded = Int(padded_arg)
    var i = global_idx.x
    if i < rows * padded:
        var r, c = divmod(i, padded)
        if c < cols:
            dest[r, c] = source[r, c]
        else:
            dest[r, c] = 0


def linear(ctx: DeviceContext, x: Tensor, weight: Tensor) raises -> Tensor:
    """Compute x[M,K] @ weight[N,K].T with f32 accumulation."""
    var result = Tensor.empty(ctx, x.rows, weight.rows)
    if x.rows == 1:
        # Bypass the tiled M3 GEMM dispatch: stream weights with vector loads.
        enqueue_apple_gemv_config[tile_m=1, rows_per_warp=1, unroll=4](
            result.view(), x.view(), weight.view(), ctx
        )
    elif x.cols % 16 == 0:
        _matmul_gpu[transpose_b=True](
            result.view(), x.view(), weight.view(), ctx
        )
    else:
        # The vision patch projection has K=1176; M3 MMA requires K % 16 == 0.
        var k = ceildiv(x.cols, 16) * 16
        var padded_x = Tensor.empty(ctx, x.rows, k)
        var padded_weight = Tensor.empty(ctx, weight.rows, k)
        ctx.enqueue_function[_pad_columns](
            x.view(),
            padded_x.view(),
            Int32(x.rows),
            Int32(x.cols),
            Int32(k),
            grid_dim=ceildiv(x.rows * k, _BLOCK),
            block_dim=_BLOCK,
        )
        ctx.enqueue_function[_pad_columns](
            weight.view(),
            padded_weight.view(),
            Int32(weight.rows),
            Int32(weight.cols),
            Int32(k),
            grid_dim=ceildiv(weight.rows * k, _BLOCK),
            block_dim=_BLOCK,
        )
        _matmul_gpu[transpose_b=True](
            result.view(), padded_x.view(), padded_weight.view(), ctx
        )
    return result^


def _linear_into[
    ComputeFn: ElementwiseComputeFn
](
    ctx: DeviceContext,
    x: Tensor,
    weight: Tensor,
    mut result: Tensor,
    compute_fn: ComputeFn,
) raises:
    """Store `compute_fn(coords, bf16(x @ weight.T))` with no extra pass."""
    if x.rows == 1:
        enqueue_apple_gemv_config[
            tile_m=1, rows_per_warp=1, unroll=4, has_compute_fn=True
        ](result.view(), x.view(), weight.view(), ctx, compute_fn=compute_fn)
    elif x.cols % 16 == 0:
        enqueue_apple_matmul_8x8[transpose_b=True, has_compute_fn=True](
            result.view(),
            x.view(),
            weight.view(),
            x.rows,
            weight.rows,
            x.cols,
            ctx,
            compute_fn=compute_fn,
        )
    else:
        raise Error("Fused linear requires K % 16 == 0")


@always_inline
def _plus[
    dtype: DType, width: SIMDLength, //, broadcast: Bool
](coords: Coord, value: SIMD[dtype, width], addend: _Input) -> SIMD[
    dtype, width
]:
    """Add an addend tile in f32 and round once, like `add_bias`/`add`.

    `broadcast` reads row 0 of `addend` (a bias) for every output row.
    """
    comptime assert addend.flat_rank == 2
    var row = 0 if broadcast else Int(coords[0].value())
    var col = Int(coords[1].value())
    var result = SIMD[dtype, width]()
    comptime for i in range(width):
        result[i] = (
            value[i].cast[DType.float32]()
            + rebind[Float32](addend[row, col + i].cast[DType.float32]())
        ).cast[dtype]()
    return result


def linear_bias(
    ctx: DeviceContext, x: Tensor, weight: Tensor, bias: Tensor
) raises -> Tensor:
    """Compute x @ weight.T + bias with the bias fused into the GEMM store."""
    var result = Tensor.empty(ctx, x.rows, weight.rows)
    var b = bias.view().as_imm().as_unsafe_any_origin()

    @always_inline
    def epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var b} -> SIMD[dtype, width]:
        return _plus[broadcast=True](coords, value, b)

    _linear_into(ctx, x, weight, result, epilogue)
    return result^


def linear_bias_gelu[
    exact: Bool
](ctx: DeviceContext, x: Tensor, weight: Tensor, bias: Tensor) raises -> Tensor:
    """Compute GELU(x @ weight.T + bias), rounding to BF16 after each step."""
    var result = Tensor.empty(ctx, x.rows, weight.rows)
    var b = bias.view().as_imm().as_unsafe_any_origin()

    @always_inline
    def epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var b} -> SIMD[dtype, width]:
        var biased = _plus[broadcast=True](coords, value, b)
        comptime if exact:
            return gelu(biased)
        else:
            return gelu_quick(biased)

    _linear_into(ctx, x, weight, result, epilogue)
    return result^


def linear_residual(
    ctx: DeviceContext, x: Tensor, weight: Tensor, residual: Tensor
) raises -> Tensor:
    """Compute residual + x @ weight.T, rounding after each step."""
    var result = Tensor.empty(ctx, x.rows, weight.rows)
    var r = residual.view().as_imm().as_unsafe_any_origin()

    @always_inline
    def epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var r} -> SIMD[dtype, width]:
        return _plus[broadcast=False](coords, value, r)

    _linear_into(ctx, x, weight, result, epilogue)
    return result^


def linear_bias_residual(
    ctx: DeviceContext,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    residual: Tensor,
) raises -> Tensor:
    """Compute residual + (x @ weight.T + bias), rounding after each step."""
    var result = Tensor.empty(ctx, x.rows, weight.rows)
    var b = bias.view().as_imm().as_unsafe_any_origin()
    var r = residual.view().as_imm().as_unsafe_any_origin()

    @always_inline
    def epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var b, var r} -> SIMD[
        dtype, width
    ]:
        return _plus[broadcast=False](
            coords, _plus[broadcast=True](coords, value, b), r
        )

    _linear_into(ctx, x, weight, result, epilogue)
    return result^


def linear_swiglu(
    ctx: DeviceContext, x: Tensor, up_weight: Tensor, gate: Tensor
) raises -> Tensor:
    """Compute SiLU(gate) * (x @ up_weight.T), preserving `swiglu` rounding."""
    var result = Tensor.empty(ctx, x.rows, up_weight.rows)
    var g = gate.view().as_imm().as_unsafe_any_origin()

    @always_inline
    def epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](coords: Coord, value: SIMD[dtype, width]) {var g} -> SIMD[dtype, width]:
        comptime assert g.flat_rank == 2
        var row = Int(coords[0].value())
        var col = Int(coords[1].value())
        var result = SIMD[dtype, width]()
        comptime for i in range(width):
            var activated = silu(g[row, col + i])
            result[i] = (
                rebind[Float32](activated.cast[DType.float32]())
                * value[i].cast[DType.float32]()
            ).cast[dtype]()
        return result

    _linear_into(ctx, x, up_weight, result, epilogue)
    return result^


def _norm[
    rms: Bool
](
    x: _Input,
    weight: _Input,
    bias: _Input,
    dest: _Output,
    rows_arg: Int32,
    cols_arg: Int32,
):
    comptime assert x.flat_rank == 2 and weight.flat_rank == 2
    comptime assert bias.flat_rank == 2 and dest.flat_rank == 2
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var row = global_idx.x // WARP_SIZE
    if row >= rows:
        return
    var lane = lane_id()
    # Each lane owns `_VECTOR`-wide column chunks; `cols % _VECTOR == 0`.
    var start = lane * _VECTOR
    comptime stride = WARP_SIZE * _VECTOR
    var total: Float32 = 0
    for c in range(start, cols, stride):
        var value = x.load[width=_VECTOR](Coord(row, c)).cast[DType.float32]()
        comptime if rms:
            total += (value * value).reduce_add()
        else:
            total += value.reduce_add()
    var mean: Float32 = 0
    var variance: Float32 = 0
    comptime if rms:
        variance = warp.sum(total) / Float32(cols)
    else:
        mean = warp.sum(total) / Float32(cols)
        var squared: Float32 = 0
        for c in range(start, cols, stride):
            var delta = (
                x.load[width=_VECTOR](Coord(row, c)).cast[DType.float32]()
                - mean
            )
            squared += (delta * delta).reduce_add()
        variance = warp.sum(squared) / Float32(cols)
    var scale = rsqrt(variance + 1.0e-6)
    for c in range(start, cols, stride):
        var value = (
            x.load[width=_VECTOR](Coord(row, c)).cast[DType.float32]() - mean
        ) * scale
        value *= weight.load[width=_VECTOR](Coord(0, c)).cast[DType.float32]()
        comptime if not rms:
            value += bias.load[width=_VECTOR](Coord(0, c)).cast[DType.float32]()
        dest.store(Coord(row, c), value.cast[BF16]())


def _enqueue_norm[
    rms: Bool
](ctx: DeviceContext, x: Tensor, weight: Tensor, bias: Tensor) raises -> Tensor:
    if x.cols % _VECTOR:
        raise Error("Normalized width must be a multiple of 8")
    var result = Tensor.empty(ctx, x.rows, x.cols)
    comptime kernel = _norm[rms]
    ctx.enqueue_function[kernel](
        x.view(),
        weight.view(),
        bias.view(),
        result.view(),
        Int32(x.rows),
        Int32(x.cols),
        grid_dim=ceildiv(x.rows, _BLOCK // WARP_SIZE),
        block_dim=_BLOCK,
    )
    return result^


def layer_norm(
    ctx: DeviceContext, x: Tensor, weight: Tensor, bias: Tensor
) raises -> Tensor:
    """Row-wise LayerNorm with f32 statistics and eps=1e-6."""
    return _enqueue_norm[False](ctx, x, weight, bias)


def rms_norm(ctx: DeviceContext, x: Tensor, weight: Tensor) raises -> Tensor:
    """Row-wise RMSNorm with f32 statistics and eps=1e-6."""
    # The RMS kernel never reads its bias operand.
    return _enqueue_norm[True](ctx, x, weight, weight)


def upload(
    ctx: DeviceContext, data: List[Float32], rows: Int, cols: Int
) raises -> Tensor:
    """Round host f32 data to BF16 and upload a row-major matrix."""
    var result = Tensor.empty(ctx, rows, cols)
    var host = ctx.enqueue_create_host_buffer[BF16](rows * cols)
    var values = TileTensor(host, row_major(Int32(rows), Int32(cols)))
    comptime assert values.flat_rank == 2
    for r in range(rows):
        for c in range(cols):
            values[r, c] = data[r * cols + c].cast[BF16]()
    ctx.enqueue_copy(result.buffer, host)
    return result^


def _embed(
    indices: _IndexInput,
    weight: _Input,
    vision: _Input,
    dest: _Output,
    rows_arg: Int32,
    cols_arg: Int32,
):
    comptime assert indices.flat_rank == 2 and weight.flat_rank == 2
    comptime assert vision.flat_rank == 2 and dest.flat_rank == 2
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var i = global_idx.x
    if i < rows * cols:
        var r, c = divmod(i, cols)
        var image_row = Int(indices[r, 1])
        if image_row >= 0:
            dest[r, c] = vision[image_row, c]
        else:
            dest[r, c] = weight[Int(indices[r, 0]), c]


def embed(
    ctx: DeviceContext,
    ids: List[Int],
    weight: Tensor,
    vision: Tensor,
    image_token: Int,
) raises -> Tensor:
    """Look up tokens and splice sequential vision rows at image tokens."""
    var rows = len(ids)
    var result = Tensor.empty(ctx, rows, weight.cols)
    var host = ctx.enqueue_create_host_buffer[DType.int64](rows * 2)
    var indices = TileTensor(host, row_major(Int32(rows), Int32(2)))
    comptime assert indices.flat_rank == 2
    var image_row = 0
    for r in range(rows):
        indices[r, 0] = Int64(ids[r])
        if ids[r] == image_token:
            indices[r, 1] = Int64(image_row)
            image_row += 1
        else:
            indices[r, 1] = -1
    var device = ctx.enqueue_create_buffer[DType.int64](rows * 2)
    ctx.enqueue_copy(device, host)
    var device_indices = TileTensor(device, row_major(Int32(rows), Int32(2)))
    ctx.enqueue_function[_embed](
        device_indices,
        weight.view(),
        vision.view(),
        result.view(),
        Int32(rows),
        Int32(weight.cols),
        grid_dim=ceildiv(rows * weight.cols, _BLOCK),
        block_dim=_BLOCK,
    )
    return result^


def _last_row(x: _Input, dest: _Output, rows_arg: Int32, cols_arg: Int32):
    comptime assert x.flat_rank == 2 and dest.flat_rank == 2
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var c = global_idx.x
    if c < cols:
        dest[0, c] = x[rows - 1, c]


def last_row(ctx: DeviceContext, x: Tensor) raises -> Tensor:
    """Copy the last input row into an independently owned GPU matrix."""
    var result = Tensor.empty(ctx, 1, x.cols)
    ctx.enqueue_function[_last_row](
        x.view(),
        result.view(),
        Int32(x.rows),
        Int32(x.cols),
        grid_dim=ceildiv(x.cols, _BLOCK),
        block_dim=_BLOCK,
    )
    return result^


def _argmax_partials(
    x: _Input,
    maxima: _FloatOutput,
    indices: _IndexOutput,
    count_arg: Int32,
    cols_arg: Int32,
):
    comptime assert (
        x.flat_rank == 2 and maxima.flat_rank == 2 and indices.flat_rank == 2
    )
    var count = Int(count_arg)
    var cols = Int(cols_arg)
    var group = global_idx.x // WARP_SIZE
    var lane = lane_id()
    var start = group * ARGMAX_GROUP
    var best: Float32 = min_or_neg_inf[DType.float32]()
    var best_index = Int64.MAX
    for offset in range(lane, ARGMAX_GROUP, WARP_SIZE):
        var i = start + offset
        if i < count:
            var r, c = divmod(i, cols)
            var value = x[r, c].cast[DType.float32]()
            if value > best or (value == best and Int64(i) < best_index):
                best = value
                best_index = Int64(i)
    var maximum = warp.max(best)
    if best != maximum:
        best_index = Int64.MAX
    var index = warp.min(best_index)
    if lane == 0 and start < count:
        maxima[0, group] = maximum
        indices[0, group] = index


def _argmax_finish(
    maxima: _FloatInput,
    indices: _IndexInput,
    dest: _IndexOutput,
    count_arg: Int32,
):
    comptime assert (
        maxima.flat_rank == 2 and indices.flat_rank == 2 and dest.flat_rank == 2
    )
    var count = Int(count_arg)
    var lane = lane_id()
    var best: Float32 = min_or_neg_inf[DType.float32]()
    var best_index = Int64.MAX
    for i in range(lane, count, WARP_SIZE):
        var value = maxima[0, i]
        var index = indices[0, i]
        if value > best or (value == best and index < best_index):
            best = value
            best_index = index
    var maximum = warp.max(best)
    if best != maximum:
        best_index = Int64.MAX
    var index = warp.min(best_index)
    if lane == 0:
        dest[0, 0] = index


def argmax(ctx: DeviceContext, x: Tensor) raises -> Int:
    """Return the flat maximum index, choosing the smallest index on ties."""
    var count = x.rows * x.cols
    var groups = ceildiv(count, ARGMAX_GROUP)
    var maxima = ctx.enqueue_create_buffer[DType.float32](groups)
    var indices = ctx.enqueue_create_buffer[DType.int64](groups)
    var result = ctx.enqueue_create_buffer[DType.int64](1)
    var maxima_view = TileTensor(maxima, row_major(Int32(1), Int32(groups)))
    var indices_view = TileTensor(indices, row_major(Int32(1), Int32(groups)))
    var result_view = TileTensor(result, row_major(Int32(1), Int32(1)))
    ctx.enqueue_function[_argmax_partials](
        x.view(),
        maxima_view,
        indices_view,
        Int32(count),
        Int32(x.cols),
        grid_dim=ceildiv(groups, _BLOCK // WARP_SIZE),
        block_dim=_BLOCK,
    )
    ctx.enqueue_function[_argmax_finish](
        maxima_view.as_imm(),
        indices_view.as_imm(),
        result_view,
        Int32(groups),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    with result.map_to_host() as host:
        var value = TileTensor(host, row_major(Int32(1), Int32(1)))
        comptime assert value.flat_rank == 2
        return Int(value[0, 0])
