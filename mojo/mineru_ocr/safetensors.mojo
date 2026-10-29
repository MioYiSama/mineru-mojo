"""Safetensors loading into one mapped device allocation with parallel reads."""
from std.ffi import external_call
from max.algorithm import parallelize
from max.gpu.host import DeviceContext
from .json import JsonDocument
from .tensor import BF16, Tensor

# Parallel `pread` calls saturate the page-cache copy far beyond one thread.
comptime _READ_TASKS = 16


def _read_exact(fd: Int32, dest: Int, count: Int, offset: Int) -> Bool:
    var done = 0
    while done < count:
        var n = external_call["pread", Int](
            fd, dest + done, count - done, offset + done
        )
        if n <= 0:
            return False
        done += n
    return True


def _read_header(fd: Int32) raises -> String:
    var prefix = List[UInt8](length=8, fill=0)
    if not _read_exact(fd, Int(prefix.unsafe_ptr()), 8, 0):
        raise Error("Truncated safetensors file")
    var header_length = 0
    for i in range(8):
        header_length |= Int(prefix[i]) << (i * 8)
    if header_length <= 0 or header_length > 100000000:
        raise Error("Invalid safetensors header length")
    var header = List[UInt8](length=header_length, fill=0)
    if not _read_exact(fd, Int(header.unsafe_ptr()), header_length, 8):
        raise Error("Truncated safetensors header")
    return String(unsafe_from_utf8=Span(header))


struct Weights(Movable):
    """All checkpoint BF16 tensors resident on the GPU, indexed by HF name.

    Each tensor is a sub-buffer view of one allocation holding the file's
    data section.
    """

    var tensors: List[Tensor]
    var index: Dict[String, Int]

    def __init__(out self):
        self.tensors = List[Tensor](capacity=681)
        self.index = Dict[String, Int]()

    @staticmethod
    def load(ctx: DeviceContext, var path: String) raises -> Weights:
        var fd = external_call["open", Int32, num_fixed_args=2](
            path.as_c_string_span().ptr(), Int32(0)
        )
        if fd < 0:
            raise Error("Cannot open checkpoint: " + path)
        try:
            var result = Weights._load(ctx, fd)
            _ = external_call["close", Int32](fd)
            return result^
        except err:
            _ = external_call["close", Int32](fd)
            raise err

    @staticmethod
    def _load(ctx: DeviceContext, fd: Int32) raises -> Weights:
        var header = _read_header(fd)
        var data_start = 8 + header.byte_length()
        var file_size = external_call["lseek", Int](fd, 0, Int32(2))
        var data_bytes = file_size - data_start
        if data_bytes <= 0 or data_bytes % 2:
            raise Error("Invalid safetensors data section")
        var storage = ctx.enqueue_create_buffer[BF16](data_bytes // 2)
        with storage.map_to_host() as mapped:
            var base = Int(mapped.unsafe_ptr())
            var chunk = (data_bytes + _READ_TASKS - 1) // _READ_TASKS
            var failed = List[Bool](length=_READ_TASKS, fill=False)

            def read_chunk(
                task: Int,
            ) {
                var fd,
                var base,
                var chunk,
                var data_bytes,
                var data_start,
                mut failed,
            }:
                var begin = task * chunk
                var count = min(chunk, data_bytes - begin)
                if count > 0:
                    failed[task] = not _read_exact(
                        fd, base + begin, count, data_start + begin
                    )

            parallelize(read_chunk, _READ_TASKS)
            if True in failed:
                raise Error("Truncated safetensors file")

        var doc = JsonDocument.parse(header)
        var names = doc.keys(doc.root())
        var result = Weights()
        for name in names:
            if name == "__metadata__":
                continue
            var node = doc.get(doc.root(), name)
            if doc.string(doc.get(node, "dtype")) != "BF16":
                raise Error("Expected BF16 checkpoint tensor: " + name)
            var shape = doc.get(node, "shape")
            var rows = doc.int(doc.at(shape, 0))
            var cols = 1
            for i in range(1, doc.len(shape)):
                cols *= doc.int(doc.at(shape, i))
            # Vectors are viewed as [1, width].
            if doc.len(shape) == 1:
                cols = rows
                rows = 1
            var offsets = doc.get(node, "data_offsets")
            var begin = doc.int(doc.at(offsets, 0))
            var end = doc.int(doc.at(offsets, 1))
            if end - begin != rows * cols * 2 or begin % 2 or end > data_bytes:
                raise Error("Safetensors shape/offset mismatch: " + name)
            result.index[name] = len(result.tensors)
            result.tensors.append(
                Tensor(
                    storage.create_sub_buffer[BF16](begin // 2, rows * cols),
                    rows,
                    cols,
                )
            )
        return result^

    def get(
        ref self, name: StringSlice
    ) raises -> ref[
        List[Tensor]._InteriorOrigin[origin_of(self.tensors)]
    ] Tensor:
        return self.tensors[self.index[String(name)]]
