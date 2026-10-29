# State-dict coverage audit at load

`compile(weights=...)` checks the state dict against the module's parameters
from the parameter side. Every parameter needs a tensor with a matching shape
and dtype, or compilation raises: `KeyError` for a missing tensor, and
`ValueError` for a mismatched one. It does not check the other direction. A
checkpoint tensor whose name matches no parameter is left out without a
warning.

Every weight adapter has this gap. Take a typo in a rename table, a stale
prefix from a previous port revision, or a checkpoint component the module
never built. In each case the model still compiles, still serves, and still
emits real-looking tokens. The error shows up only in accuracy benchmarks.

A typical failure: the checkpoint ships `self_attn.q_norm.weight` and
`self_attn.k_norm.weight` in every layer, but the port's attention module was
copied from a donor with no QK-norm. Compilation succeeds because the module
asks for no norm weights, the norm tensors are dropped, and generation looks
plausible until an accuracy benchmark exposes the gap.

This reference describes an audit that reports every unconsumed checkpoint
tensor at load time, in seconds.

## What the audit checks

The audit compares two name sets: the keys the adapter produced, and the
parameter names the root module yields from `parameters`. Every key with no
parameter is either a bug or a tensor the port drops on purpose. Dropped
tensors include a vision tower in a text-only port, or a
multi-token-prediction head the port doesn't serve. The audit raises on the
first kind and ignores the second by prefix.

## Where the parameter names come from

`parameters` walks the module as a tree and yields each tensor under its
dotted attribute path, so the names the audit compares are the same names
`load_state_dict()` and `compile(weights=...)` look for. Call the walk
directly when the audit needs the paths as a set:

```python
from max import tree

parameter_names = set(tree.paths(nn_model, leaf=Tensor))
```

`leaf=Tensor` stops the walk at tensors, which is the set the audit compares
against. Drop the argument and the same call also yields the module's
non-tensor attributes, such as a quantization config or an integer constant,
and those are not parameters.

## Generic implementation

Add this function to your slug's ``model.py``:

```python
from collections.abc import Iterable
from collections.abc import Set as AbstractSet


def _audit_unconsumed_keys(
    *,
    checkpoint_keys: AbstractSet[str],
    parameter_names: Iterable[str],
    dropped_prefixes: tuple[str, ...] = (),
) -> None:
    """Raises if a checkpoint tensor matches no parameter of the module.

    Keys under ``dropped_prefixes`` belong to components the port leaves out
    on purpose and are ignored.
    """
    unconsumed = sorted(
        key
        for key in checkpoint_keys - set(parameter_names)
        if not key.startswith(dropped_prefixes)
    )
    if unconsumed:
        sample = unconsumed[:8]
        raise RuntimeError(
            f"State-dict audit: {len(unconsumed)} checkpoint tensor(s) match "
            f"no module parameter. Likely cause: a stale rename in "
            f"weight_adapters.py or a component the module doesn't build. "
            f"First {len(sample)}: {sample}"
        )
```

## Where to wire it

`ModuleV3PipelineModelWithKVCache.load_model()` calls `_prepare_state_dict()`
with the adapted state dict, then builds the module in
`_instantiate_module()` under `F.lazy()`, then compiles. Record the keys in
the first hook and audit in the second:

```python
class MyModel(DonorModel):
    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        state_dict = super()._prepare_state_dict(state_dict, model_config)
        self._checkpoint_keys = frozenset(state_dict)
        return state_dict

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = MyTopLevelModule(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        _audit_unconsumed_keys(
            checkpoint_keys=self._checkpoint_keys,
            parameter_names=(name for name, _ in nn_model.parameters),
        )
        return nn_model
```

The module is built lazily, so listing its parameters allocates no weights.
The audit is set arithmetic over ~10K names and runs once per load.

## What it catches

- **Adapter rename typos.** ``mlp.routed.experts`` when the module's path is
  ``mlp.experts``: every routed expert tensor is reported unconsumed, and
  compilation would also fail on the missing ``mlp.experts`` parameters.
- **Stale prefix.** ``model.layers`` left unrenamed when the root module
  expects ``language_model.layers``: every layer tensor is reported.
- **Components the module never built.** QK-norm, attention biases, shared
  experts, or a post-feedforward norm that the checkpoint ships and the
  donor lacks. Compilation alone can't see these.
- **Wrong expert count.** A checkpoint with 128 experts loaded into a module
  built for 64: experts 64–127 are reported.

## What it doesn't catch

- **Renames to a wrong-but-existing path.** If the adapter sends
  ``layers.0.mlp.experts.{j}.gate_proj.weight`` to
  ``layers.0.mlp.shared_experts.gate_proj.weight``, and the shapes match,
  the tensor is consumed by the wrong parameter. Compare adapter output
  against HF modeling code for any rename you wrote by hand.
- **Tensor values.** The audit and the compile-time checks verify names,
  shapes, and dtypes. A transposed or mis-permuted tensor with the right
  shape loads without complaint. The comparator in
  [`debug-model`](../../debug-model/SKILL.md) finds those.

## Reporting style

When the audit passes, log it explicitly so the user sees it:

```python
logger.info(
    "State-dict audit: all %d checkpoint tensors consumed by the module.",
    len(self._checkpoint_keys),
)
```

When it fails, the ``RuntimeError`` message is enough to diagnose the bug.
The first 8 unconsumed keys usually pattern-match
a single class of bug ("every ``q_norm``/``k_norm`` key" → missing QK-norm,
"everything under ``model.layers``" → prefix rename).
