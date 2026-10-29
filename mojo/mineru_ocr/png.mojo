"""PNG decoding and RFC 1950/1951 inflation without external libraries."""

from std.math import abs, max, min


@fieldwise_init
struct RGBImage(Movable):
    """Row-major RGB8 pixels after the reference Qwen PIL color conversion."""

    var width: Int
    var height: Int
    var pixels: List[UInt8]


@inline(.always)
def _be32(data: List[UInt8], pos: Int) -> Int:
    return (
        (Int(data[pos]) << 24)
        | (Int(data[pos + 1]) << 16)
        | (Int(data[pos + 2]) << 8)
        | Int(data[pos + 3])
    )


struct _Bits(Movable):
    var data: List[UInt8]
    var pos: Int
    var bits: Int
    var count: Int

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.pos = 2
        self.bits = 0
        self.count = 0

    @inline(.always)
    def peek(mut self, n: Int) raises -> Int:
        while self.count < n:
            if self.pos >= len(self.data):
                raise Error("Truncated DEFLATE stream")
            self.bits |= Int(self.data[self.pos]) << self.count
            self.count += 8
            self.pos += 1
        return self.bits & ((1 << n) - 1)

    @inline(.always)
    def take(mut self, n: Int) raises -> Int:
        var value = self.peek(n)
        self.bits >>= n
        self.count -= n
        return value


struct _Huffman(Movable):
    var table: List[Int32]

    def __init__(out self, lengths: List[Int]) raises:
        self.table = List[Int32](length=32768, fill=0)
        var counts = List[Int](length=16, fill=0)
        for n in lengths:
            if n < 0 or n > 15:
                raise Error("Invalid Huffman code length")
            if n > 0:
                counts[n] += 1
        var next_code = List[Int](length=16, fill=0)
        var code = 0
        for n in range(1, 16):
            code = (code + counts[n - 1]) << 1
            if code + counts[n] > (1 << n):
                raise Error("Oversubscribed Huffman tree")
            next_code[n] = code
        for symbol in range(len(lengths)):
            var n = lengths[symbol]
            if n == 0:
                continue
            code = next_code[n]
            next_code[n] += 1
            var reversed = 0
            for i in range(n):
                reversed = (reversed << 1) | ((code >> i) & 1)
            var entry = Int32((symbol << 4) | n)
            for i in range(reversed, 32768, 1 << n):
                self.table[i] = entry

    @inline(.always)
    def decode(self, mut bits: _Bits) raises -> Int:
        var entry = Int(self.table[bits.peek(15)])
        var n = entry & 15
        if n == 0:
            raise Error("Invalid Huffman symbol")
        bits.bits >>= n
        bits.count -= n
        return entry >> 4


def _inflate(var data: List[UInt8], expected: Int) raises -> List[UInt8]:
    if (
        len(data) < 6
        or (Int(data[0]) & 15) != 8
        or (Int(data[0]) >> 4) > 7
        or ((Int(data[0]) << 8) | Int(data[1])) % 31 != 0
        or (Int(data[1]) & 32) != 0
    ):
        raise Error("Invalid PNG zlib header")
    var bits = _Bits(data^)
    var output = List[UInt8](length=expected, fill=0)
    var written = 0
    var final = 0
    var length_base: List[Int] = [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        15,
        17,
        19,
        23,
        27,
        31,
        35,
        43,
        51,
        59,
        67,
        83,
        99,
        115,
        131,
        163,
        195,
        227,
        258,
    ]
    var length_extra: List[Int] = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        0,
    ]
    var distance_base: List[Int] = [
        1,
        2,
        3,
        4,
        5,
        7,
        9,
        13,
        17,
        25,
        33,
        49,
        65,
        97,
        129,
        193,
        257,
        385,
        513,
        769,
        1025,
        1537,
        2049,
        3073,
        4097,
        6145,
        8193,
        12289,
        16385,
        24577,
    ]
    var distance_extra: List[Int] = [
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
        12,
        12,
        13,
        13,
    ]
    while final == 0:
        final = bits.take(1)
        var block_type = bits.take(2)
        if block_type == 0:
            _ = bits.take(bits.count % 8)
            var n = bits.take(16)
            if (n ^ bits.take(16)) != 65535 or n > expected - written:
                raise Error("Invalid stored DEFLATE block")
            for _ in range(n):
                output[written] = UInt8(bits.take(8))
                written += 1
            continue
        if block_type == 3:
            raise Error("Reserved DEFLATE block type")
        var literal_lengths = List[Int](length=288, fill=0)
        var distance_lengths = List[Int](length=32, fill=5)
        if block_type == 1:
            for i in range(288):
                literal_lengths[i] = 8 if i < 144 else (
                    9 if i < 256 else (7 if i < 280 else 8)
                )
        else:
            var nlit = bits.take(5) + 257
            var ndist = bits.take(5) + 1
            var ncode = bits.take(4) + 4
            if nlit > 286:
                raise Error("Invalid DEFLATE literal count")
            # RFC 1951's code-length alphabet, not numerical order.
            var order: List[Int] = [
                16,
                17,
                18,
                0,
                8,
                7,
                9,
                6,
                10,
                5,
                11,
                4,
                12,
                3,
                13,
                2,
                14,
                1,
                15,
            ]
            var code_lengths = List[Int](length=19, fill=0)
            for i in range(ncode):
                code_lengths[order[i]] = bits.take(3)
            var code_tree = _Huffman(code_lengths)
            var lengths = List[Int](capacity=nlit + ndist)
            while len(lengths) < nlit + ndist:
                var symbol = code_tree.decode(bits)
                if symbol < 16:
                    lengths.append(symbol)
                else:
                    var value = 0
                    var repeat: Int
                    if symbol == 16:
                        if len(lengths) == 0:
                            raise Error(
                                "DEFLATE repeat without previous length"
                            )
                        value = lengths[len(lengths) - 1]
                        repeat = bits.take(2) + 3
                    elif symbol == 17:
                        repeat = bits.take(3) + 3
                    else:
                        repeat = bits.take(7) + 11
                    if repeat > nlit + ndist - len(lengths):
                        raise Error("DEFLATE code-length repeat overflow")
                    for _ in range(repeat):
                        lengths.append(value)
            literal_lengths = List[Int](capacity=nlit)
            distance_lengths = List[Int](capacity=ndist)
            for i in range(nlit):
                literal_lengths.append(lengths[i])
            for i in range(ndist):
                distance_lengths.append(lengths[nlit + i])
        if literal_lengths[256] == 0:
            raise Error("DEFLATE block has no end symbol")
        var literals = _Huffman(literal_lengths)
        var distances = _Huffman(distance_lengths)
        while True:
            var symbol = literals.decode(bits)
            if symbol < 256:
                if written >= expected:
                    raise Error("PNG inflated data exceeds expected size")
                output[written] = UInt8(symbol)
                written += 1
            elif symbol == 256:
                break
            else:
                if symbol > 285:
                    raise Error("Invalid DEFLATE length symbol")
                var index = symbol - 257
                var n = length_base[index] + bits.take(length_extra[index])
                var dsym = distances.decode(bits)
                if dsym >= 30:
                    raise Error("Invalid DEFLATE distance symbol")
                var distance = distance_base[dsym] + bits.take(
                    distance_extra[dsym]
                )
                if distance > written or n > expected - written:
                    raise Error("Invalid DEFLATE back-reference")
                for _ in range(n):
                    output[written] = output[written - distance]
                    written += 1
    if written != expected:
        raise Error("PNG inflated data has incorrect size")
    return output^


@inline(.always)
def _paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = abs(p - a)
    var pb = abs(p - b)
    var pc = abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


@inline(.always)
def _sample(row: List[UInt8], index: Int, depth: Int) -> Int:
    if depth == 8:
        return Int(row[index])
    if depth == 16:
        return (Int(row[index * 2]) << 8) | Int(row[index * 2 + 1])
    var bit = index * depth
    return (Int(row[bit // 8]) >> (8 - depth - bit % 8)) & ((1 << depth) - 1)


def decode_png(path: String) raises -> RGBImage:
    """Decode every PNG color/depth/interlace variant with Qwen's PIL semantics.

    RGBA (including PNG LA16, which PIL opens as RGBA) is pasted onto white;
    LA8 and palette alpha are discarded. PIL clips 16-bit gray to RGB8;
    multichannel 16-bit PNGs are reduced to high bytes by its decoder.
    CRC and Adler checks are intentionally omitted; bounds are always checked.
    """
    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    var signature: List[UInt8] = [137, 80, 78, 71, 13, 10, 26, 10]
    if len(data) < 33:
        raise Error("Truncated PNG")
    for i in range(8):
        if data[i] != signature[i]:
            raise Error("Not a PNG file")
    var width = 0
    var height = 0
    var depth = 0
    var color = 0
    var interlace = 0
    var palette = List[UInt8]()
    var compressed = List[UInt8]()
    var pos = 8
    var ended = False
    while pos + 12 <= len(data):
        var n = _be32(data, pos)
        var kind = _be32(data, pos + 4)
        pos += 8
        if n > len(data) - pos - 4:
            raise Error("Truncated PNG chunk")
        if kind == 0x49484452:
            if width != 0 or n != 13:
                raise Error("Invalid PNG IHDR")
            width = _be32(data, pos)
            height = _be32(data, pos + 4)
            depth = Int(data[pos + 8])
            color = Int(data[pos + 9])
            interlace = Int(data[pos + 12])
            if (
                width < 1
                or height < 1
                or width > 2147483647
                or height > 2147483647
                or width > (0x7FFFFFFFFFFFFFFF // 8) // height
                or data[pos + 10] != 0
                or data[pos + 11] != 0
                or interlace > 1
            ):
                raise Error("Invalid PNG dimensions or encoding")
            var valid = (
                (
                    color == 0
                    and (
                        depth == 1
                        or depth == 2
                        or depth == 4
                        or depth == 8
                        or depth == 16
                    )
                )
                or (
                    color == 3
                    and (depth == 1 or depth == 2 or depth == 4 or depth == 8)
                )
                or (
                    (color == 2 or color == 4 or color == 6)
                    and (depth == 8 or depth == 16)
                )
            )
            if not valid:
                raise Error("Invalid PNG color type or bit depth")
        elif width == 0:
            raise Error("PNG IHDR must be first")
        elif kind == 0x504C5445:
            if n == 0 or n > 768 or n % 3 != 0:
                raise Error("Invalid PNG palette")
            for i in range(n):
                palette.append(data[pos + i])
        elif kind == 0x49444154:
            for i in range(n):
                compressed.append(data[pos + i])
        elif kind == 0x49454E44:
            ended = True
            break
        elif (kind & 0x20000000) == 0:
            raise Error("Unsupported critical PNG chunk")
        # tRNS does not change the original PIL mode; to_rgb ignores it.
        pos += n + 4
    if not ended or width == 0 or (color == 3 and len(palette) == 0):
        raise Error("Incomplete PNG")
    var channels = 1 if color == 0 or color == 3 else (
        3 if color == 2 else (2 if color == 4 else 4)
    )
    var starts_x: List[Int] = [0, 4, 0, 2, 0, 1, 0]
    var starts_y: List[Int] = [0, 0, 4, 0, 2, 0, 1]
    var steps_x: List[Int] = [8, 8, 4, 4, 2, 2, 1]
    var steps_y: List[Int] = [8, 8, 8, 4, 4, 2, 2]
    var passes = 7 if interlace else 1
    if not interlace:
        steps_x[0] = 1
        steps_y[0] = 1
    var expected = 0
    for p in range(passes):
        var pw = max(0, (width - starts_x[p] + steps_x[p] - 1) // steps_x[p])
        var ph = max(0, (height - starts_y[p] + steps_y[p] - 1) // steps_y[p])
        if pw > 0 and ph > 0:
            expected += ph * (1 + (pw * channels * depth + 7) // 8)
    var raw = _inflate(compressed^, expected)
    var pixels = List[UInt8](length=width * height * 3, fill=0)
    var source = 0
    var bpp = max(1, (channels * depth + 7) // 8)
    for p in range(passes):
        var pw = max(0, (width - starts_x[p] + steps_x[p] - 1) // steps_x[p])
        var ph = max(0, (height - starts_y[p] + steps_y[p] - 1) // steps_y[p])
        if pw == 0 or ph == 0:
            continue
        var stride = (pw * channels * depth + 7) // 8
        var previous = List[UInt8](length=stride, fill=0)
        var row = List[UInt8](length=stride, fill=0)
        for y in range(ph):
            var filter = Int(raw[source])
            source += 1
            if filter > 4:
                raise Error("Invalid PNG filter")
            for i in range(stride):
                var a = Int(row[i - bpp]) if i >= bpp else 0
                var b = Int(previous[i])
                var c = Int(previous[i - bpp]) if i >= bpp else 0
                var prediction = 0
                if filter == 1:
                    prediction = a
                elif filter == 2:
                    prediction = b
                elif filter == 3:
                    prediction = (a + b) // 2
                elif filter == 4:
                    prediction = _paeth(a, b, c)
                row[i] = UInt8((Int(raw[source + i]) + prediction) & 255)
            source += stride
            for x in range(pw):
                var r = _sample(row, x * channels, depth)
                var g: Int
                var b: Int
                if color == 3:
                    if r * 3 + 2 >= len(palette):
                        raise Error("PNG palette index out of bounds")
                    g = Int(palette[r * 3 + 1])
                    b = Int(palette[r * 3 + 2])
                    r = Int(palette[r * 3])
                elif color == 0 or color == 4:
                    r = min(r, 255) if depth == 16 and color == 0 else (
                        r >> 8 if depth == 16 else r * 255 // ((1 << depth) - 1)
                    )
                    g = r
                    b = r
                else:
                    g = _sample(row, x * channels + 1, depth)
                    b = _sample(row, x * channels + 2, depth)
                    if depth == 16:
                        r >>= 8
                        g >>= 8
                        b >>= 8
                if color == 6 or (color == 4 and depth == 16):
                    var alpha = _sample(row, x * channels + channels - 1, depth)
                    if depth == 16:
                        alpha >>= 8
                    # Pillow's DIV255: exact nearest integer, including ties.
                    var rr = r * alpha + 255 * (255 - alpha) + 128
                    var gg = g * alpha + 255 * (255 - alpha) + 128
                    var bb = b * alpha + 255 * (255 - alpha) + 128
                    r = (rr + (rr >> 8)) >> 8
                    g = (gg + (gg >> 8)) >> 8
                    b = (bb + (bb >> 8)) >> 8
                var target = (
                    (starts_y[p] + y * steps_y[p]) * width
                    + starts_x[p]
                    + x * steps_x[p]
                ) * 3
                pixels[target] = UInt8(r)
                pixels[target + 1] = UInt8(g)
                pixels[target + 2] = UInt8(b)
            var swap = previous^
            previous = row^
            row = swap^
    return RGBImage(width, height, pixels^)
