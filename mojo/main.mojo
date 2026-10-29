"""Standalone MinerU OCR: PNG + native tokenizer + Metal Qwen2-VL inference."""
from std.sys import argv, exit
from std.os import getenv
from std.ffi import external_call
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu.host import DeviceContext
from mineru_ocr.config import MERGE_UNIT
from mineru_ocr.json import JsonDocument
from mineru_ocr.tokenizer import Qwen2Tokenizer
from mineru_ocr.png import decode_png
from mineru_ocr.image import VisionPatches, preprocess_image
from mineru_ocr.safetensors import Weights
from mineru_ocr.vision import encode_vision
from mineru_ocr.generation import generate

comptime USAGE = (
    "Usage: mineru-ocr <image.png> [--prompt TEXT] [--max-new-tokens N]"
    " [--model-dir DIR]"
)


def stderr(var text: String):
    _ = external_call["write", Int](
        Int32(2), text.as_c_string_span().ptr(), text.byte_length()
    )


def default_model_dir() raises -> String:
    var home = getenv("HF_HOME", "")
    if home == "":
        home = getenv("HOME", "") + "/.cache/huggingface"
    var repo = home + "/hub/models--opendatalab--MinerU2.5-Pro-2605-1.2B"
    with open(repo + "/refs/main", "r") as f:
        var revision = String(f.read().strip())
        if revision == "" or "/" in revision or ".." in revision:
            raise Error("Invalid Hugging Face cache revision")
        return repo + "/snapshots/" + revision


def resize_bounds(model_dir: String) raises -> Tuple[Int, Int]:
    """Read `(min_pixels, max_pixels)` from `preprocessor_config.json`."""
    with open(model_dir + "/preprocessor_config.json", "r") as f:
        var doc = JsonDocument.parse(f.read())
        var root = doc.root()
        return (
            doc.int(doc.get(root, "min_pixels")),
            doc.int(doc.get(root, "max_pixels")),
        )


struct Request(Movable):
    """The tokenized chat prompt and preprocessed image for one OCR call."""

    var tokenizer: Qwen2Tokenizer
    var patches: VisionPatches
    var ids: List[Int]
    var image_token: Int

    def __init__(
        out self, model_dir: String, image_path: String, prompt: String
    ) raises:
        self.tokenizer = Qwen2Tokenizer.load(model_dir)
        var image = decode_png(image_path)
        var min_pixels, max_pixels = resize_bounds(model_dir)
        self.patches = preprocess_image(image, min_pixels, max_pixels)
        if self.patches.grid_t != 1:
            raise Error(
                "PNG preprocessing must produce exactly one temporal grid"
            )
        self.image_token = self.tokenizer.token_id("<|image_pad|>")
        var chat = String(
            "<|im_start|>system\nYou are a helpful"
            " assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>"
        )
        for _ in range(self.patches.grid_h * self.patches.grid_w // MERGE_UNIT):
            chat += "<|image_pad|>"
        chat += (
            "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        )
        self.ids = self.tokenizer.encode(chat)


def run() raises:
    var launch = perf_counter_ns()
    var args = argv()
    if len(args) == 2 and (args[1] == "--help" or args[1] == "-h"):
        print(USAGE)
        return
    if len(args) < 2:
        raise Error(USAGE)
    var image_path = String(args[1])
    var prompt = String("\nText Recognition:")
    var max_new_tokens = 2048
    var model_dir = String("")
    var i = 2
    while i < len(args):
        var flag = String(args[i])
        if i + 1 >= len(args):
            raise Error("Missing value for option: " + flag)
        if flag == "--prompt":
            prompt = String(args[i + 1])
        elif flag == "--max-new-tokens":
            max_new_tokens = Int(args[i + 1])
            if max_new_tokens < 0:
                raise Error("--max-new-tokens must be nonnegative")
        elif flag == "--model-dir":
            model_dir = String(args[i + 1])
        else:
            raise Error("Unknown option: " + flag)
        i += 2
    if model_dir == "":
        model_dir = default_model_dir()
    # CPU request preparation overlaps device setup and checkpoint reads.
    var ctx: Optional[DeviceContext] = None
    var weights: Optional[Weights] = None
    var request: Optional[Request] = None
    var errors = List[String](length=2, fill=String())

    def prepare(
        task: Int,
    ) {
        var model_dir,
        var image_path,
        var prompt,
        mut ctx,
        mut weights,
        mut request,
        mut errors,
    }:
        try:
            if task == 0:
                ctx = DeviceContext()
                weights = Weights.load(
                    ctx.value(), model_dir + "/model.safetensors"
                )
            else:
                request = Request(model_dir, image_path, prompt)
        except err:
            errors[task] = String(err)

    parallelize(prepare, 2)
    for err in errors:
        if err:
            raise Error(err)
    var device = ctx.take()
    var model = weights.take()
    var req = request.take()
    device.synchronize()
    var start = perf_counter_ns()
    var embeddings = encode_vision(
        device, model, req.patches.data, req.patches.grid_h, req.patches.grid_w
    )
    device.synchronize()
    var vision_seconds = Float64(perf_counter_ns() - start) * 1e-9
    var result = generate(
        device,
        model,
        req.ids,
        embeddings,
        req.patches.grid_h,
        req.patches.grid_w,
        req.image_token,
        max_new_tokens,
    )
    print(req.tokenizer.decode(result.tokens))
    var total_seconds = Float64(perf_counter_ns() - launch) * 1e-9
    var speed: Float64 = 0
    if result.decode_seconds > 0:
        speed = Float64(result.decode_steps) / result.decode_seconds
    stderr(
        "vision="
        + String(vision_seconds)
        + "s prefill="
        + String(result.prefill_seconds)
        + "s decode="
        + String(speed)
        + " tokens/s total="
        + String(total_seconds)
        + "s fps="
        + String(1.0 / total_seconds)
        + "\n"
    )


def main():
    try:
        run()
    except err:
        stderr("mineru-ocr: " + String(err) + "\n")
        exit(1)
