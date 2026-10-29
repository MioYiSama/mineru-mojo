# mineru

MinerU2.5-Pro-2605-1.2B (`Qwen2VLForConditionalGeneration`) on Modular MAX, in two forms:

- `src/mineru/qwen2vl/`: a MAX Python custom architecture for `max serve`.
- `mojo/`: a native Mojo OCR executable with no Python at runtime.

Both expect the checkpoint in the Hugging Face cache
(`hf download opendatalab/MinerU2.5-Pro-2605-1.2B`).

## Native executable

```sh
uv sync
.venv/bin/mojo build mojo/main.mojo -o build/mineru-ocr
build/mineru-ocr page.png
build/mineru-ocr table.png --prompt $'\nTable Recognition:' --max-new-tokens 4096
```

Options: `--prompt` (default `\nText Recognition:`), `--max-new-tokens` (default 2048,
capped by the 8192-token context), `--model-dir` (default: the cached snapshot named by
`refs/main` under `$HF_HOME/hub`, or `~/.cache/huggingface/hub`). The answer goes to
stdout. One stats line goes to stderr: vision and prefill seconds, decode tokens/s, total
seconds from launch to answer, and `fps` (images per second, `1 / total`).

The binary is one file, but it loads four Modular runtime libraries
(`libKGENCompilerRTShared`, `libAsyncRTMojoBindings`, `libAsyncRTRuntimeGlobals`,
`libMSupportGlobals`, 2.8 MB total) through an absolute rpath into
`.venv/lib/python3.13/site-packages/modular/lib`. The Mojo toolchain ships no static
variants, so moving the binary to another machine needs those libraries too.

Pipeline (`mojo/mineru_ocr/`):

| Module | Role |
|---|---|
| `png.mojo` | PNG decoding with a native DEFLATE inflater; alpha/palette conversion as PIL does |
| `image.mojo` | `smart_resize`, Pillow-exact bicubic resampling, normalization, merge-ordered patches |
| `json.mojo`, `tokenizer.mojo`, `unicode_tables.mojo` | Qwen2 byte-level BPE with NFC and special tokens |
| `safetensors.mojo` | Reads the checkpoint into one GPU allocation with parallel `pread`s; tensors are sub-buffers |
| `dense.mojo` | MAX `linalg` 8×8 simdgroup GEMM/GEMV with fused epilogues (bias, GELU, SwiGLU, residual), norms, argmax |
| `attention.mojo`, `flash_attention.mojo` | RoPE kernels; 8×8 simdgroup-MMA flash attention (vision and causal GQA prefill) |
| `vision.mojo`, `text.mojo` | Vision tower; Qwen2 prefill with KV cache and M-RoPE |
| `decode.mojo`, `generation.mojo` | Fused single-token decode step (norm+QKV+RoPE+cache, GQA split attention, gate/up+SwiGLU, residual GEMVs, LM head+argmax) with device-side token feedback; greedy loop that keeps one step queued ahead of each readback |
| `config.mojo` | Checkpoint dimensions; the engine is specialized to this model |

`mojo/tools/gen_unicode_tables.py` regenerates `unicode_tables.mojo` (development only).

Limits: one PNG per run, greedy decoding only, single Apple GPU, BF16.

Performance on an M3 Pro (18-core GPU), median wall-clock of three runs, weights in
the page cache:

| Image | Before | After | FPS before → after |
|---|---:|---:|---:|
| 900×200 rendered two-line text crop | 2.08 s | 0.86 s | 0.48 → 1.17 |
| 1600×1200 rendered page (8112 patches, 2053 prompt tokens) | 35.66 s | 6.80 s | 0.028 → 0.147 |

Per-stage, page: vision 26.9 s → 4.67 s, prefill 0.76 s → 0.36 s, decode 20 → 103 tok/s.
Weights load in ~0.26 s, overlapped with tokenizer loading and PNG preprocessing.

## MAX serve

```sh
.venv/bin/max serve --model-path opendatalab/MinerU2.5-Pro-2605-1.2B \
  --custom-architectures src/mineru/qwen2vl --device-memory-utilization 0.5
```

Then send images to `/v1/chat/completions`.
