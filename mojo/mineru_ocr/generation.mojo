"""Single-request greedy generation with separate prefill and decode timing."""
from std.math import min, max
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from .config import CONTEXT_LENGTH, END_OF_TEXT_TOKEN, IM_END_TOKEN
from .tensor import Tensor
from .safetensors import Weights
from .text import KVCache, multimodal_positions, prefill
from .decode import DecodeState


@fieldwise_init
struct GenerationResult(Movable):
    var tokens: List[Int]
    var prefill_seconds: Float64
    var decode_seconds: Float64
    # Device steps actually run, including at most one discarded after EOS.
    var decode_steps: Int


def generate(
    ctx: DeviceContext,
    weights: Weights,
    ids: List[Int],
    image_embeddings: Tensor,
    grid_h: Int,
    grid_w: Int,
    image_token: Int,
    max_new_tokens: Int,
) raises -> GenerationResult:
    """Greedy decoding, stopping at either checkpoint EOS or the context boundary.
    """
    if max_new_tokens < 0 or len(ids) >= CONTEXT_LENGTH:
        raise Error(
            "Invalid generation length: prompt must fit the 8192-token context"
        )
    var limit = min(max_new_tokens, CONTEXT_LENGTH - len(ids))
    var answer = List[Int](capacity=limit)
    if limit == 0:
        return GenerationResult(answer^, 0.0, 0.0, 0)
    var positions = multimodal_positions(ids, grid_h, grid_w, image_token)
    var next_position = 0
    for p in positions:
        next_position = max(next_position, p + 1)
    var cache = KVCache(ctx, len(ids) + limit)
    ctx.synchronize()
    var start = perf_counter_ns()
    var token = prefill(
        ctx, weights, cache, ids, positions, image_embeddings, image_token
    )
    ctx.synchronize()
    var prefill = Float64(perf_counter_ns() - start) * 1e-9
    if token == IM_END_TOKEN or token == END_OF_TEXT_TOKEN:
        return GenerationResult(answer^, prefill, 0.0, 0)
    answer.append(token)
    if limit == 1:
        return GenerationResult(answer^, prefill, 0.0, 0)
    var state = DecodeState(ctx, weights, limit, token)
    # Step s reads token slot s on the device and writes slot s + 1, which is
    # also copied into the same slot of `host` so the readback never races.
    var host = ctx.enqueue_create_host_buffer[.int64](limit)
    # One step stays queued past the readback, so two events alternate.
    var events = [ctx.create_event(), ctx.create_event()]
    var stream = ctx.stream()
    var total = limit - 1
    var enqueued = 0
    ctx.synchronize()
    var decode_start = perf_counter_ns()
    for step in range(total):
        # Keep the next step queued while waiting so the GPU never idles.
        while enqueued < min(step + 2, total):
            state.enqueue(
                ctx,
                weights,
                cache.keys,
                cache.values,
                cache.length,
                next_position,
                enqueued,
            )
            ctx.enqueue_copy(
                host.unsafe_ptr().unsafe_offset(enqueued + 1),
                state.tokens.create_sub_buffer[.int64](enqueued + 1, 1),
            )
            stream.record_event(events[enqueued % 2])
            cache.length += 1
            next_position += 1
            enqueued += 1
        events[step % 2].synchronize()
        token = Int(host[step + 1])
        if token == IM_END_TOKEN or token == END_OF_TEXT_TOKEN:
            break
        answer.append(token)
    ctx.synchronize()
    var decode = Float64(perf_counter_ns() - decode_start) * 1e-9
    return GenerationResult(answer^, prefill, decode, enqueued)
