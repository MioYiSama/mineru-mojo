# Is the bug in MAX or in my port?

When accuracy regresses, a feature path misbehaves, or a benchmark drops to
random, the bug can be in your port, in MAX serve, or in a shared kernel.
Run the same probe on a stock MAX-native model with **identical serve
flags** to decide which side to debug.

## When to run this

- Generation works, but loglikelihood (echo path) fails
- Accuracy passes at 0-shot but fails at few-shot
- A serve flag path (batched prefill, echo, KV eviction) misbehaves
- Chat-formatted or few-shot prompts break while plain prompts work

If the failure tracks a **serve flag or endpoint format**, suspect MAX first.
If it tracks **attention shape, MoE routing, or RoPE wiring**, debug the port
(see [layer-by-layer-debugging.md](layer-by-layer-debugging.md)).

## The control test

1. **Pick a small native arch** in the same family:
   - Dense decoder: `Qwen/Qwen3-1.7B`
   - MoE: `Qwen/Qwen3-30B-A3B`
   - MLA: `deepseek-ai/DeepSeek-V2-Lite`
2. **Serve with the same flags** as your port: `--devices`,
   `--max-batch-size`, `--max-length`, `--enable-echo`,
   `--quantization-encoding`, etc.
3. **Run the same probe** (same lm-eval task, `num_fewshot`, `curl`
   payload).
4. **Compare:**
   - Native arch **fails the same way** → MAX bug. File it upstream with a
     minimal repro. Stop tuning your port for this symptom.
   - Native arch **passes** → port bug. Continue locally.

## Worked example: few-shot loglikelihood collapse

A large MoE port looked fine on generation and GSM8K, and HellaSwag 0-shot
was reasonable. But 5-shot and 10-shot were near random (~0.25–0.31 on
4-choice).

Port-side suspects: wrong NoPE `freqs_cis`, sliding-window bug,
router wiring. Tweaking those moved scores slightly but didn't fix the drop.

Control: `Qwen/Qwen3-1.7B` with the **same** `--enable-echo` and
`--max-batch-size`:

| Task                      | Your port | Qwen3-1.7B (native) |
|---------------------------|-----------|---------------------|
| HellaSwag 0-shot acc_norm | 0.76      | 0.49                |
| HellaSwag 5-shot acc_norm | 0.31      | **0.27**            |

The native model showed the same 0-shot → 5-shot drop. Diagnosis: the bug
was in few-shot loglikelihood through `--enable-echo` /
`compute_log_probabilities`, not in the custom port. The control run took
about five minutes and ruled out the port.

## Bugs the control test can find

The control test finds bugs in features every model uses the same way, such
as echo, batched prefill, and some quantization formats. If the control
fails too, the bug is in MAX.

Bugs that depend on model size or shape are harder to separate. Pick a
control in the same class as your port, such as an MoE control for an MoE
port.

The control test can't find bugs in your port's attention, MoE routing, or
block wiring. Use [divergences.md](divergences.md) and layer taps for those.

## Known-good controls

| Your port's class              | Control                        |
|--------------------------------|--------------------------------|
| Dense decoder, RoPE, RMSNorm   | `Qwen/Qwen3-1.7B`              |
| Sliding window                 | `mistralai/Mistral-7B-v0.3`    |
| MoE, top-k routing             | `Qwen/Qwen3-30B-A3B`           |
| MLA + MoE                      | `deepseek-ai/DeepSeek-V2-Lite` |
| Multimodal wrapper (text-only) | `google/gemma-3-12b-it`        |

Default to `Qwen3-1.7B` when unsure. It has the fastest cold compile.

## Escalate after the control fails

1. Minimize repro (smallest flags + probe that still fails).
2. Capture `pixi run max --version`.
3. Report that the failure reproduces on **both** your port and a stock
   native arch. That makes it a MAX issue, not a bug in your port.
4. Note the gap in bring-up notes and ship what passes. Don't keep
   debugging the port for that symptom.
