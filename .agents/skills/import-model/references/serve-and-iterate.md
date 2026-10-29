# Serve and iterate (detail)

**Path:** `<port_dir>` is the slug folder with `arch.py` and `ARCHITECTURES` in
`__init__.py` (after scaffold, `<port_dir> = <output_dir>/<slug>`). Pass
the same `<port_dir>` to `--custom-architectures` and to
`run_oss_gates.py --port-dir`. Leave off the trailing slash. MAX reads
`<output_dir>/<slug>/` as an empty module name and fails with
`Failed to import custom model`.

MAX loads custom archs by taking `dirname(<port_dir>)` for `sys.path` and
importing `basename(<port_dir>)` as the module. Don't pass the parent
of `<port_dir>`. MAX then imports the wrong package (`custom-arch` in place
of your slug), and you see `AttributeError: module '…' has no attribute
'ARCHITECTURES'`.

Optional colon form: ``<parent_on_sys.path>:<module_name>`` (same effect as
passing `<port_dir>`).

## Prerequisites: run these before `pixi run max serve`

`max serve` compiles the whole model before it answers a request, so a
mistake that the checks below catch in seconds otherwise costs a compile.
`run_oss_gates.py` covers walls, checkpoint metadata, and the `arch.py` name
and encoding only.

### Port check

`check_port.py` imports the port the way `max serve` does, builds the pipeline
`max serve` would build, and stops in `load_model()` right after the port
constructs its root module under `F.lazy()`. It then compares the module's
parameters with the tensors your weight adapter returns for the real
checkpoint:

```bash
pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir>
```

For an embedding port, add `--task embeddings_generation`. The report lists:

- **missing**: a parameter with no tensor. `compile()` raises `KeyError`.
- **shape mismatch** and **dtype mismatch**: `compile()` raises
  `ValueError`. Cast float32 or bfloat16 tensors to the parameter's dtype in
  the adapter ([pitfalls-weights.md](pitfalls-weights.md)).
- **unconsumed**: a tensor no parameter reads. `compile()` ignores it, which
  hides a wrong rename.

The script exits 1 when a parameter is missing or mismatched. Unconsumed
tensors don't fail it, because a port can drop tensors on purpose (an MTP
head, KV-cache scales). Explain each one. See
[rename-weights.md](rename-weights.md) and
[state-dict-audit.md](state-dict-audit.md).

### Weights-format preflight

MAX loads only `.safetensors` or `.gguf` (`WeightsFormat` in
`max/graph/weights/format.py`). It can't load `.bin`. In
`<port_dir>/arch.py`, copy `default_weights_format` and `weight_adapters`
from your scaffold donor under
`max/pipelines/architectures/<donor>/arch.py`. See "Import and config API
traps" in
[pitfalls-config.md](pitfalls-config.md#import-and-config-api-traps).

Check the repo's file list:

```bash
pixi run python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('<HF_MODEL_ID>')
has_st = any(f.endswith('.safetensors') for f in files)
has_gguf = any(f.endswith('.gguf') for f in files)
has_bin = any(f.endswith('.bin') for f in files)
print(f'safetensors={has_st}  gguf={has_gguf}  bin_only_legacy={has_bin and not has_st}')
if has_bin and not has_st:
    print('STOP: convert to safetensors or pick a GGUF repo; MAX cannot load .bin')
"
```

## Sanity-check the HF reference first

Load the reference in float32 when it fits. A bfloat16 reference adds its own
rounding, so a correct bfloat16 port can drift from it within a few tokens.

```bash
pixi run python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = '<HF_MODEL_ID>'
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    mid, trust_remote_code=True, device_map='auto', dtype=torch.float32,
).eval()
ids = tok('<MODEL CARD EXAMPLE PROMPT>', return_tensors='pt').input_ids.to(m.device)
with torch.no_grad():
    out = m.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=False))
"
```

## Serve and probe

```bash
pixi run max serve --model-path <HF_MODEL_ID> \
  --custom-architectures <port_dir> \
  --quantization-encoding <default_encoding from arch.py>

curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<HF_MODEL_ID>", "prompt": "<MODEL CARD EXAMPLE PROMPT>", "max_tokens": 64}' \
  | pixi run python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

Use the model card's template for the prompt. The `model` field must match
`--model-path` (or `--served-model-name` if you set it).

## Encoder / embedding slugs

Use the same `--custom-architectures <port_dir>`, plus
`--task embeddings_generation`. Without it, `max serve` picks text
generation for an architecture name that more than one task registers
(`Qwen3ForCausalLM` is both), and the model fails to load. `scaffold.py`
prints the serve command with the flag. The endpoint is `/v1/embeddings`,
and it takes `input`:

```bash
pixi run max serve --model-path <HF_MODEL_ID> \
  --custom-architectures <port_dir> --task embeddings_generation

curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "<HF_MODEL_ID>", "input": ["first text", "second text"]}'
```

## Read the first serve result

- **Crash on load** → config, imports, or weight adapter.
- **Garbage tokens** → load [`debug-model`](../../debug-model/SKILL.md).
  The module may still implement donor math or a latent delta.
- **Plausible short output** → run `max_tokens=64+` before you trust it.

## Iterate on fixes

Each fix costs a serve and its compile: make one fix per serve, and stop the
server between runs with `pkill -f "max serve"`. When logits diverge or output
is garbage, stop iterating on scalar taps and load
[`debug-model`](../../debug-model/SKILL.md). Use
[layer-by-layer-debugging.md](layer-by-layer-debugging.md) for the quick
`compare_layers.py` probe only.
