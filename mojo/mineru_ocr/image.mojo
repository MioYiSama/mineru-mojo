"""Pillow-compatible bicubic resize and Qwen2-VL image patch extraction."""

from std.math import abs, ceil, floor, max, min, sqrt
from .png import RGBImage


@fieldwise_init
struct VisionPatches(Movable):
    """Merge-group-ordered [grid_t*grid_h*grid_w, 1176] float32 patches."""

    var grid_t: Int
    var grid_h: Int
    var grid_w: Int
    var data: List[Float32]


@inline(.always)
def _round_factor(value: Int) -> Int:
    # Python round uses ties-to-even, not round-away-from-zero.
    var q = value // 28
    var r = value % 28
    return (q + Int(r > 14 or (r == 14 and q % 2 != 0))) * 28


def smart_resize(
    height: Int, width: Int, min_pixels: Int, max_pixels: Int
) raises -> Tuple[Int, Int]:
    """Qwen smart_resize with factor 28 and Python's ties-to-even rounding."""
    if height <= 0 or width <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
        raise Error("Invalid image dimensions or pixel bounds")
    if Float64(max(height, width)) / Float64(min(height, width)) > 200.0:
        raise Error("Image aspect ratio exceeds 200")
    var h = max(28, _round_factor(height))
    var w = max(28, _round_factor(width))
    if h * w > max_pixels:
        var beta = sqrt(Float64(height * width) / Float64(max_pixels))
        h = max(28, Int(floor(Float64(height) / beta / 28.0)) * 28)
        w = max(28, Int(floor(Float64(width) / beta / 28.0)) * 28)
    elif h * w < min_pixels:
        var beta = sqrt(Float64(min_pixels) / Float64(height * width))
        h = Int(ceil(Float64(height) * beta / 28.0)) * 28
        w = Int(ceil(Float64(width) * beta / 28.0)) * 28
    return (h, w)


@inline(.always)
def _cubic(value: Float64) -> Float64:
    var x = abs(value)
    if x < 1.0:
        return ((1.5 * x - 2.5) * x) * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    return 0.0


struct _Coefficients(Movable):
    var starts: List[Int]
    var counts: List[Int]
    var weights: List[Int32]
    var stride: Int

    def __init__(out self, input_size: Int, output_size: Int):
        var scale = Float64(input_size) / Float64(output_size)
        var filter_scale = max(1.0, scale)
        var support = 2.0 * filter_scale
        self.stride = Int(ceil(support)) * 2 + 1
        self.starts = List[Int](capacity=output_size)
        self.counts = List[Int](capacity=output_size)
        self.weights = List[Int32](length=output_size * self.stride, fill=0)
        var temporary = List[Float64](length=self.stride, fill=0.0)
        var inverse = 1.0 / filter_scale
        for i in range(output_size):
            var center = (Float64(i) + 0.5) * scale
            # C casts truncate toward zero, including negative lower bounds.
            var start = max(0, Int(center - support + 0.5))
            var end = min(input_size, Int(center + support + 0.5))
            self.starts.append(start)
            self.counts.append(end - start)
            var total: Float64 = 0.0
            for j in range(end - start):
                var weight = _cubic(
                    (Float64(j + start) - center + 0.5) * inverse
                )
                temporary[j] = weight
                total += weight
            for j in range(end - start):
                var weight = temporary[j] / total
                self.weights[i * self.stride + j] = Int32(
                    weight * 4194304.0 + (-0.5 if weight < 0.0 else 0.5)
                )


def _resample(image: RGBImage, size: Int, horizontal: Bool) -> RGBImage:
    var width = size if horizontal else image.width
    var height = image.height if horizontal else size
    var coefficients = _Coefficients(
        image.width if horizontal else image.height, size
    )
    var pixels = List[UInt8](length=width * height * 3, fill=0)
    for y in range(height):
        for x in range(width):
            var index = x if horizontal else y
            var start = coefficients.starts[index]
            var count = coefficients.counts[index]
            var weight_start = index * coefficients.stride
            var offset = (
                (y * image.width + start)
                * 3 if horizontal else (start * image.width + x)
                * 3
            )
            var step = 3 if horizontal else image.width * 3
            var r = 2097152
            var g = 2097152
            var b = 2097152
            for j in range(count):
                var weight = Int(coefficients.weights[weight_start + j])
                r += Int(image.pixels[offset]) * weight
                g += Int(image.pixels[offset + 1]) * weight
                b += Int(image.pixels[offset + 2]) * weight
                offset += step
            var target = (y * width + x) * 3
            pixels[target] = UInt8(min(255, max(0, r >> 22)))
            pixels[target + 1] = UInt8(min(255, max(0, g >> 22)))
            pixels[target + 2] = UInt8(min(255, max(0, b >> 22)))
    return RGBImage(width, height, pixels^)


def resize_rgb(image: RGBImage, width: Int, height: Int) -> RGBImage:
    """Match Pillow RGB BICUBIC: 22-bit fixed weights and clipped uint8 passes.

    Pillow 13 runs the vertical pass first when its absolute reduction exceeds
    twice the horizontal reduction; clipping between passes makes order matter.
    """
    if width == image.width:
        if height == image.height:
            return RGBImage(width, height, image.pixels.copy())
        return _resample(image, height, False)
    if height == image.height:
        return _resample(image, width, True)
    if (
        image.height - height > 0
        and image.height - height > (image.width - width) * 2
    ):
        var intermediate = _resample(image, height, False)
        return _resample(intermediate, width, True)
    var intermediate = _resample(image, width, True)
    return _resample(intermediate, height, False)


def _patchify(image: RGBImage) -> VisionPatches:
    var gh = image.height // 14
    var gw = image.width // 14
    var result = List[Float32](length=gh * gw * 1176, fill=0.0)
    var mean: List[Float32] = [0.48145466, 0.4578275, 0.40821073]
    var stddev: List[Float32] = [0.26862954, 0.26130258, 0.27577711]
    var table = List[Float32](length=768, fill=0.0)
    for c in range(3):
        var scale = Float32(1.0) / (Float32(255.0) * stddev[c])
        var offset = -mean[c] / stddev[c]
        for pixel in range(256):
            var scaled = Float32(pixel) * scale
            table[c * 256 + pixel] = scaled + offset
    var patch = 0
    for group_y in range(gh // 2):
        for group_x in range(gw // 2):
            for merge_y in range(2):
                for merge_x in range(2):
                    var y = (group_y * 2 + merge_y) * 14
                    var x = (group_x * 2 + merge_x) * 14
                    for c in range(3):
                        for py in range(14):
                            for px in range(14):
                                var value = table[
                                    c * 256
                                    + Int(
                                        image.pixels[
                                            ((y + py) * image.width + x + px)
                                            * 3
                                            + c
                                        ]
                                    )
                                ]
                                var target = (
                                    patch * 1176 + c * 392 + py * 14 + px
                                )
                                result[target] = value
                                result[target + 196] = value
                    patch += 1
    return VisionPatches(1, gh, gw, result^)


def preprocess_image(
    image: RGBImage, min_pixels: Int, max_pixels: Int
) raises -> VisionPatches:
    """Resize, normalize in float32, duplicate time, and emit merged patch order.
    """
    var dimensions = smart_resize(
        image.height, image.width, min_pixels, max_pixels
    )
    var height = dimensions[0]
    var width = dimensions[1]
    if width == image.width and height == image.height:
        return _patchify(image)
    var resized = resize_rgb(image, width, height)
    return _patchify(resized)
