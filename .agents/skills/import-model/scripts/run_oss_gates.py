# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Preflight gates for the port workflow (walls, checkpoint metadata, arch registration).

Complements ``check_port.py``, which checks the port's parameters against
its adapted checkpoint.

Phases:

- ``preflight`` (default): wall scan + ``arch.py`` name/encoding vs Hub config.
- ``verify`` (requires ``--port``): against a running ``pixi run max serve``,
  compares HF's and MAX's prefill logprob at ``--prompt`` and the top-1 token
  at several prefix lengths.

Usage::

    pixi run python run_oss_gates.py <HF_ID> --port-dir <port_dir>
    pixi run python run_oss_gates.py <HF_ID> --port-dir <port_dir> \\
        --phase verify --port 8000
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from .check_walls import scan_config
    from .checkpoint_metadata import fetch_repo_tensors
    from .compare_layers import (
        PREFILL_REL_TOL,
        fetch_max_logprobs,
        hf_prefill_top1,
        prefill_rel_diff,
        probe_multi_position,
        server_error,
    )
    from .dtype_utils import canonical_native_dtype, encoding_from_config_dict
    from .hub_config import architecture_class, load_hub_config
    from .max_arch_paths import string_keyword, supported_architecture_calls
except ImportError:
    # Standalone invocation: `python /path/to/run_oss_gates.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_walls import scan_config  # type: ignore[no-redef]
    from checkpoint_metadata import fetch_repo_tensors  # type: ignore[no-redef]
    from compare_layers import (  # type: ignore[no-redef]
        PREFILL_REL_TOL,
        fetch_max_logprobs,
        hf_prefill_top1,
        prefill_rel_diff,
        probe_multi_position,
        server_error,
    )
    from dtype_utils import (  # type: ignore[no-redef]
        canonical_native_dtype,
        encoding_from_config_dict,
    )
    from hub_config import (  # type: ignore[no-redef]
        architecture_class,
        load_hub_config,
    )
    from max_arch_paths import (  # type: ignore[no-redef]
        string_keyword,
        supported_architecture_calls,
    )


@dataclass
class GateResult:
    gate: str
    status: Literal["PASS", "FAIL", "WARN", "SKIP"]
    detail: str


def _parse_arch_py(port_dir: Path) -> tuple[str | None, str | None]:
    """The ``name`` and string ``default_encoding`` the port registers.

    Reads the first ``SupportedArchitecture`` call in ``arch.py``. A full
    copy can set ``default_encoding`` from a config constant, which reads
    as ``None``.
    """
    arch = port_dir / "arch.py"
    if not arch.is_file():
        return None, None
    call = next(supported_architecture_calls(ast.parse(arch.read_text())), None)
    if call is None:
        return None, None
    return string_keyword(call, "name"), string_keyword(
        call, "default_encoding"
    )


def gate_checkpoint_metadata(hf_id: str, port_dir: Path) -> GateResult:
    """Verify the Hub repo exposes safetensors metadata and dtype matches arch."""
    try:
        summary = fetch_repo_tensors(hf_id)
    except Exception as exc:
        return GateResult("checkpoint_meta", "FAIL", str(exc))

    dominant = summary.dominant_dtype()
    _, enc = _parse_arch_py(port_dir)
    if enc and dominant and enc != dominant:
        return GateResult(
            "checkpoint_meta",
            "WARN",
            f"dominant checkpoint dtype {dominant!r} != arch default_encoding "
            f"{enc!r}. compile() rejects a {dominant} tensor for a {enc} "
            f"parameter, so cast in weight_adapters.py "
            f"(references/pitfalls-weights.md)",
        )
    detail = f"{len(summary.tensors)} tensors, dominant={dominant}"
    if summary.sharded:
        detail += ", sharded"
    return GateResult("checkpoint_meta", "PASS", detail)


def gate_walls(cfg: dict[str, Any]) -> GateResult:
    findings = scan_config(cfg)
    blocks = [f for f in findings if f.level == "block"]
    if blocks:
        return GateResult("walls", "FAIL", blocks[0].message)
    warns = [f for f in findings if f.level == "warn"]
    if warns:
        return GateResult("walls", "WARN", warns[0].message)
    return GateResult("walls", "PASS", "no wall signals")


def gate_arch_name(cfg: dict[str, Any], port_dir: Path) -> GateResult:
    expected = architecture_class(cfg)
    found, _ = _parse_arch_py(port_dir)
    if found is None:
        return GateResult(
            "arch_name", "FAIL", f"missing arch.py under {port_dir}"
        )
    if found != expected:
        return GateResult(
            "arch_name",
            "FAIL",
            f"arch.py name={found!r} != config architectures[0]={expected!r}",
        )
    return GateResult("arch_name", "PASS", found)


def gate_encoding(cfg: dict[str, Any], port_dir: Path) -> GateResult:
    hub_raw = encoding_from_config_dict(cfg)
    expected = canonical_native_dtype(hub_raw) if hub_raw else "bfloat16"
    _, found = _parse_arch_py(port_dir)
    if found is None:
        # A full copy keeps the donor's ``Config.DEFAULT_ENCODING`` reference.
        detail = (
            "arch.py sets no string default_encoding"
            if (port_dir / "arch.py").is_file()
            else "no arch.py"
        )
        return GateResult("encoding", "SKIP", detail)
    if expected is None:
        return GateResult(
            "encoding",
            "SKIP",
            f"Hub config dtype {hub_raw!r} names no concrete dtype; set "
            "default_encoding from the checkpoint's tensors",
        )
    if found != expected:
        return GateResult(
            "encoding",
            "WARN",
            f"default_encoding={found!r} != Hub-native {expected!r}",
        )
    return GateResult("encoding", "PASS", found)


def gate_verify_prefill(
    hf_id: str, model_name: str, port: int, dtype: str, prompt: str
) -> GateResult:
    """Compare HF's and MAX's top-1 logprob for the token after ``prompt``.

    Catches a wrong layout (RoPE, norm, attention scale) that still picks
    the right top-1 token at short prefixes.
    """
    try:
        mx = fetch_max_logprobs(port, model_name, prompt)
    except urllib.error.HTTPError as exc:
        return GateResult("logits_prefill", "FAIL", server_error(exc))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return GateResult(
            "logits_prefill", "FAIL", f"MAX server unreachable on {port}: {exc}"
        )
    try:
        hf_logprob, _ = hf_prefill_top1(hf_id, prompt, dtype)
    except Exception as exc:
        return GateResult("logits_prefill", "FAIL", f"HF side failed: {exc}")
    rel_diff = prefill_rel_diff(hf_logprob, mx["top1_logprob"])
    detail = (
        f"top-1 logprob hf={hf_logprob:.4f} max={mx['top1_logprob']:.4f} "
        f"rel_diff={rel_diff:.4f}"
    )
    if rel_diff < PREFILL_REL_TOL:
        return GateResult("logits_prefill", "PASS", detail)
    return GateResult("logits_prefill", "FAIL", detail)


def gate_verify_logprobs(
    hf_id: str,
    model_name: str,
    port: int,
    dtype: str,
) -> GateResult:
    try:
        diverged = probe_multi_position(
            hf_id,
            model_name,
            port,
            dtype=dtype,
            quiet=True,
        )
    except urllib.error.HTTPError as exc:
        return GateResult("logits_multi", "FAIL", server_error(exc))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return GateResult(
            "logits_multi", "FAIL", f"MAX server unreachable on {port}: {exc}"
        )
    except Exception as exc:
        # The HF reference load can fail on its own (network, OOM, config).
        return GateResult("logits_multi", "FAIL", f"HF side failed: {exc}")
    if diverged:
        pos, _hf, _mx = diverged[0]
        return GateResult(
            "logits_multi",
            "FAIL",
            f"first top-1 mismatch at prefix length {pos}",
        )
    return GateResult("logits_multi", "PASS", "all probed positions match")


def run_preflight(hf_id: str, port_dir: Path) -> list[GateResult]:
    try:
        cfg = load_hub_config(hf_id)
    except Exception as exc:
        return [
            GateResult("hub_config", "FAIL", f"Hub config load failed: {exc}")
        ]
    return [
        gate_walls(cfg),
        gate_checkpoint_metadata(hf_id, port_dir),
        gate_arch_name(cfg, port_dir),
        gate_encoding(cfg, port_dir),
    ]


def run_verify(
    hf_id: str,
    port_dir: Path,
    model_name: str,
    port: int,
    dtype: str,
    prompt: str,
) -> list[GateResult]:
    results = run_preflight(hf_id, port_dir)
    if any(r.status == "FAIL" for r in results):
        for gate in ("logits_prefill", "logits_multi"):
            results.append(GateResult(gate, "SKIP", "preflight failed"))
        return results
    results.append(gate_verify_prefill(hf_id, model_name, port, dtype, prompt))
    results.append(gate_verify_logprobs(hf_id, model_name, port, dtype))
    return results


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id")
    parser.add_argument(
        "--port-dir",
        type=Path,
        required=True,
        help="Slug directory containing arch.py (same path as --custom-architectures)",
    )
    parser.add_argument(
        "--phase", choices=("preflight", "verify"), default="preflight"
    )
    parser.add_argument(
        "--served-model-name",
        help="Model name the server answers to (verify phase, default: HF ID)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="HF reference dtype for the verify phase",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt for the verify phase's prefill logprob check. For an "
        "instruction-tuned model, pass text rendered with its chat template.",
    )
    parser.add_argument("--json-out", type=Path, help="Write results JSON")


def main(args: argparse.Namespace) -> int:
    port_dir = args.port_dir.resolve()
    if not port_dir.is_dir():
        sys.exit(f"port-dir not found: {port_dir}")

    model_name = args.served_model_name or args.hf_id
    if args.phase == "preflight":
        results = run_preflight(args.hf_id, port_dir)
    else:
        results = run_verify(
            args.hf_id, port_dir, model_name, args.port, args.dtype, args.prompt
        )

    first_fail = next((r.gate for r in results if r.status == "FAIL"), None)
    overall = "FAIL" if first_fail else "PASS"

    payload = {
        "hf_id": args.hf_id,
        "port_dir": str(port_dir),
        "phase": args.phase,
        "overall": overall,
        "first_failing_gate": first_fail,
        "gates": [r.__dict__ for r in results],
    }

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    for r in results:
        print(f"{r.gate:16} {r.status:4}  {r.detail}")
    print(f"\noverall: {overall}")

    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
