# Read the layer-by-layer debug output

> **For parity/coherence failures** (server runs but output is wrong), load
> the [`debug-model`](../../debug-model/SKILL.md) skill
> first. That protocol mandates per-layer tensor-dump comparators and parallel
> investigation agents. This reference covers the quick `compare_layers.py`
> probe that `import-model` runs before handing off.

The divergence hunt has two layers of debugging:

1. **`compare_layers.py`**: automated logit probes (what the script runs).
2. **Per-layer tensor dumps**: HF vs MAX comparators (see
   `debug-model`). Prefer these over scalar output taps.

## Run `compare_layers.py`

From the skill root:

```bash
pixi run python scripts/compare_layers.py <HF_MODEL_ID> \
  --port 8000 \
  --prompt "The capital of France is"
```

Requires `pixi run max serve` with `--custom-architectures <port_dir>` (the slug
folder, not its parent). The probes read logprobs, which the overlap
scheduler rejects with HTTP 400. MAX enables that scheduler unless `arch.py`
sets `supports_overlap_scheduler=False`. For other ports, add both
`--no-enable-overlap-scheduler` and `--force` to the serve command: MAX 26.6
and earlier ignore the first flag without the second.

Flags:

| Flag                            | Purpose                                                            |
|---------------------------------|--------------------------------------------------------------------|
| `--dtype float32\|bfloat16`     | HF reference dtype (default `float32`)                             |
| `--served-model-name NAME`      | Model name the server answers to (default: the HF model ID)        |
| `--skip-hf-layers`              | Skip HF hidden-state stats (faster)                                |
| `--skip-multi`                  | Skip multi-position prefix probe                                   |
| `--long-prompt "..."`           | Prompt for multi-position probe (~110 tokens default)              |
| `--positions 1,5,20,50,100,200` | Prefix lengths to compare                                          |

Preflight before serve (guard + scaffold):

```bash
pixi run python scripts/check_walls.py <HF_MODEL_ID>
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>
```

After serve (during the divergence hunt):

```bash
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> \
  --port-dir <port_dir> --phase verify --port 8000
```

---

## What the script prints

### 1. HF hidden-state stats (HF-only)

```text
  layer        mean     max_abs        norm
    0      0.0001      0.4200       12.34
    ...
```

These rows are diagnostic only. MAX doesn't expose hidden states through
the completions API. Use them to confirm HF loads sanely (no NaNs, reasonable
norms). They don't compare against MAX.

### 2. Short-prompt top-1 logprob

```text
       check            hf           max    rel_diff  verdict
top1_logprob      -0.1234      -0.1250      0.0130  ok
```

| Verdict    | Meaning                                          |
|------------|--------------------------------------------------|
| `ok`       | Relative logprob difference &lt; 5%              |
| `DIVERGED` | Prefill logits disagree at the last prompt token |

If this row diverges at token 0, the bug is early: tokenizer, embedding,
norm order, or attention block 0.

### 3. Multi-position top-1 probe

```text
   pos     hf_id              max_text  verdict
     1      1234                  ' the'  ok
     5      5678                  ' fox'  ok
    50      9012                  ' lap'  DIVERGED
```

The script decodes HF's argmax token id and compares to MAX's greedy completion
text at each prefix length:

| Verdict    | Meaning                                                                |
|------------|------------------------------------------------------------------------|
| `ok`       | MAX's top-1 token is HF's top-1 token                                  |
| `near-tie` | MAX picked HF's second or later token, within 0.1 nats of HF's top-1   |
| `DIVERGED` | MAX's top-1 token isn't among HF's close candidates                    |

`near-tie` counts as a match: bfloat16 serving moves top logprobs by about
0.05 nats, enough to swap two close tokens in a correct model.
`run_oss_gates.py --phase verify` runs this probe as its `logits_multi` gate,
and the short-prompt check above as its `logits_prefill` gate. A wrong RoPE
layout can match the top-1 token at short prefixes, so the prefill gate is
the one that catches it early.

**First `DIVERGED` row** localizes position-dependent bugs:

| First divergence at | Likely cause                                  |
|---------------------|-----------------------------------------------|
| Position 1–5        | Tokenizer, embedding, early attention         |
| Position 5–20       | GQA repeat, QK-norm, head layout              |
| Position 50+        | RoPE style/theta, partial-RoPE padding        |
| Position 100+       | Decode-path position handling, sliding window |

See [divergences.md](divergences.md) for fixes.

---

## Per-layer tensor dumps (when logit probes aren't enough)

Use per-layer dumps when short and multi-position probes pass but output is
still wrong, or when you need to know *which sublayer* diverged. Follow
[`debug-model`](../../debug-model/SKILL.md): build HF and MAX dumpers, run the
comparator, and read
[`comparator-output-patterns.md`](../../debug-model/references/comparator-output-patterns.md).

Use extra `forward()` outputs as fine-grained sub-taps only *after* the lead
agent localizes the broken subsystem, not as the primary bisect loop.

### Interpret tap diffs

If the first divergence is at:

- **post-embed**: check `weight_adapters.py` for `embed_tokens.weight`, and
  check `tie_word_embeddings`.
- **post-attn (layer 0)**: suspect attention. Check Q/K/V naming, RoPE
  style, GQA, QK-norm, and the sliding-window mask.
- **post-mlp (layer 0)**: suspect the MLP. Check the activation variant
  (`gelu_new` vs `gelu_tanh`), gated layout, and bias terms.
- **post-block-norm** but attn and MLP ok: suspect block wiring. Check
  pre-norm vs parallel vs dual-norm, residual order, and MuP multipliers.

### Scaling vs direction bugs

- **Similar direction, different magnitude**: missing scale factor, wrong
  softmax scale, wrong norm formulation (`x * weight` vs `x * (1 + weight)`).
- **Similar magnitude, low cos_sim**: RoPE style mismatch, head permutation,
  sign error in `rotate_half`.

---

## Iteration loop

1. Run `compare_layers.py` (or `run_oss_gates.py --phase verify`).
2. Localize from the first `DIVERGED` row (logprob row or multi-position row).
3. Open the matching HF `forward()` for that layer/component.
4. Fix one operation in `<slug>.py` or `weight_adapters.py`.
5. **Restart `pixi run max serve`**. Stale compiled graphs hide fixes.
6. Re-run.

Each pass should push the first divergence later. If it doesn't, revert and
re-read HF source.

---

## Limitations

- The MAX completions API has no per-layer hidden-state export, so taps are
  manual.
- Top-1 text comparison assumes greedy decode, so use `temperature=0.0` on
  serve.
- HF reference loads with `trust_remote_code=True` and `device_map="auto"`.
- `--dtype` sets the HF reference's dtype. The MAX side runs at the serve
  encoding. A bfloat16 reference adds its own rounding: on SmolLM2-135M, a
  correct MAX model differs from a bfloat16 reference by 7.7% in top-1
  logprob and from a float32 reference by 1.6%. Keep the float32 default.

For `trust_remote_code` models that NaN with `.to("cuda")`, see
[pitfalls-serving.md](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan).
