# Rewrite donor docstrings after scaffold-by-copy

Copying a donor and renaming it doesn't update **comments and docstrings**.
The class names get renamed. The text that records *what the file claims to
do* (module docstrings, class docstrings, code comments) doesn't. This
happens in `scaffold.py --full-copy` mode and in any donor file you copy or
subclass-and-edit by hand. After a rename, `my_large_moe.py` can
still open with a docstring describing DeepSeek-V3. The new class can claim
"TP, TP + EP, and DP + EP" when the port only supports one of them.

The default `scaffold.py` mode writes fresh docstrings for its generated
files. Those docstrings describe the skeleton, and they go stale as you
implement deltas. Run the audit below in every mode.

No step in the test loop checks this text. ``max serve`` and
``compare_layers.py`` don't read docstrings. A later reader who trusts a
donor docstring gets wrong information about the port's modes and layers.

This reference defines:

1. A positive shape every module docstring should follow.
2. A mechanical check to run before declaring scaffold + implementation
   done.

## The module docstring

Every file in your port directory opens with a module docstring of this
shape:

1. **What this file is**, in your port's terms, not the donor's.
   ("``my_large_moe`` text-generation module for multi-GPU serve.")
2. **What it's derived from**, named explicitly. ("Structurally mirrors
   ``max.pipelines.architectures.deepseekV3_modulev3.deepseekV3`` for the
   mesh-sharded MoE skeleton.")
3. **Deltas applied**, as a bullet list of only what you implemented.
   ("Parallel decoder block. LayerNorm without bias (not RMSNorm).
   Interleaved RoPE. Sigmoid+renorm router. NoPE on full-attn layers via
   identity ``freqs_cis``.")

Name the donor as the source ("derived from deepseekV3_modulev3"). Don't
describe the port as the donor ("this is a DeepSeek-V3 model"). A rename
swaps names in the donor's text and keeps its claims, so rewrite the
docstring from scratch using this shape.

## Worked example: module docstring

### Donor text after renaming `deepseekV3_modulev3/deepseekV3.py`

```python
"""Build a DeepSeek-V3 model that supports single-GPU, TP, and DP + EP."""
```

The class in this file is now `MyLargeMoE`. It doesn't implement
DeepSeek-V3. The released BF16 weights don't fit on one GPU, so single-GPU is
not a supported mode. Both claims are false.

### Rewrite

```python
"""MyLargeMoE text-generation module for multi-GPU serve.

Structurally mirrors ``max.pipelines.architectures.deepseekV3_modulev3``
for the mesh-sharded skeleton (``DeviceMesh`` with a ``"tp"`` axis,
column- and row-parallel projections, ``EPBatchManager`` for the routed
experts). The BF16 checkpoint needs at least four GPUs.

Deltas layered on top of the deepseekV3_modulev3 skeleton:

- **Parallel decoder block** with a single ``input_layernorm`` and
  ``x' = x + attn(norm(x)) + ffn(norm(x))`` (no post-attention norm).
- **LayerNorm without bias** (not RMSNorm). Read ``layer_norm_eps`` from
  config when ``rms_norm_eps`` is absent.
- **Interleaved RoPE** with ``interleaved=True`` on the rotary embedding
  (verify against HF ``rotate_half``, see [divergences.md](divergences.md)).
- **NoPE on full-attention layers**: identity ``freqs_cis`` (cos=1,
  sin=0) on the layers HF skips rotation for (``config.layer_types[]``
  here).
- **Sigmoid + renorm-by-sum router** (softmax-then-top-k in the donor).
"""
```

## Worked example: class docstring

### Donor text after renaming

```python
class MyLargeMoE(Module[[Tensor, Tensor, Tensor], tuple[Tensor, ...]]):
    """Unified DeepSeek-V3 model supporting single-GPU, TP, and DP + EP inference."""
```

The name, GPU modes, and model family are all wrong.

### Rewrite

```python
class MyLargeMoE(Module[[Tensor, Tensor, Tensor], tuple[Tensor, ...]]):
    """MyLargeMoE causal-LM module for multi-GPU inference.

    Builds the embedding, :class:`MyLargeMoETransformerBlock` stack, final
    norm, and LM head. Weight placements and KV cache plumbing live in the
    block modules. This class wires layers and per-layer RoPE tables (real
    vs identity for NoPE).
    """
```

## Wrong code comments

```python
# Per-head RMSNorm for Q and K (Gemma3-specific)
self.q_norm = RMSNorm(...)
self.k_norm = RMSNorm(...)
```

If your model has no QK-norm, delete the dead code or explain why fields
remain (e.g. donor sharding plumbing). Don't leave the Gemma3-specific
comment.

## Wrong behavior claims that a name grep misses

- ``"""Supports single-GPU, multi-GPU TP, and DP+EP inference."""`` in a
  multi-GPU-only file
- ``# All-reduce after attention (TP mode only)`` when there is no non-TP mode
- ``# Apply post-attention layer norm`` in a parallel block with no
  post-attention norm

Grepping for donor names doesn't catch these. Read every docstring and
comment.

## Mandatory audit before declaring the implementation done

Run from ``<port_dir>/`` (the slug folder with ``arch.py``):

```bash
grep -rniE \
  'qwen|llama|mistral|cohere|gemma|phi|deepseek|olmo|granite|qwen3|mixtral|single-GPU|single GPU|RMSNorm|QK-norm' \
  .
```

Classify each match:

- **OK**: explicit lineage ("Structurally mirrors deepseekV3_modulev3 …").
  Leave it.
- **Wrong**: donor behavior that doesn't apply, such as "Gemma3-specific
  QK-norm" with no QK-norm, or "Supports single-GPU" in a multi-GPU-only
  file. Rewrite or delete.
- **Stale**: wrong line numbers, names, or paths. Update or remove.

Then read every module and class docstring without grepping.

## What "done" looks like

Attest explicitly when implementation is complete:

> Scaffolding + implementation complete. Module docstrings follow the
> pattern above (deepseekV3_modulev3 under "derived from", not
> claimed). ``grep -rni 'deepseek' <port_dir>`` returns N hits, all legitimate
> lineage references.

Don't report the implementation complete without that statement.

## Why not ban renames?

Donor file structure is useful to keep. The wrong text comes from skipping
*post-rename review*, so the audit targets that step. It produces concrete
artifacts: grep output and rewritten docstrings.
