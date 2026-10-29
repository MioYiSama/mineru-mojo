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
"""Copy a donor package verbatim and rename it to the port.

Backs ``scaffold.py --full-copy``, for ports whose module tree has a
different shape from every donor.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    from .donor import Donor
    from .max_arch_paths import (
        ARCH_PACKAGE,
        source_offset,
        string_keyword,
        supported_architecture_calls,
    )
except ImportError:
    # Standalone invocation: `python /path/to/scaffold.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from donor import Donor  # type: ignore[no-redef]
    from max_arch_paths import (  # type: ignore[no-redef]
        ARCH_PACKAGE,
        source_offset,
        string_keyword,
        supported_architecture_calls,
    )


_MODULEV3_SLUG_SUFFIX = "_modulev3"
_COPY_SUFFIXES = {".py", ".md", ".yaml", ".json", ".toml"}
_RELATIVE_IMPORT_RE = re.compile(
    r"^(?P<lead>\s*from\s+)(?P<dots>\.+)(?P<rest>[\w.]*)(?P<tail>\s+import\b)",
    re.MULTILINE,
)


def _camel(slug: str) -> str:
    """``llama3`` -> ``Llama3``, ``deepseekV3`` -> ``DeepseekV3``."""
    return "".join(p[:1].upper() + p[1:] for p in slug.split("_"))


def rewrite_text(
    text: str, from_slug: str, to_slug: str, to_arch_name: str
) -> str:
    # References to other architectures (``max.pipelines.architectures.X``)
    # keep their slug: they still name the donor tree the copy imports from.
    text = re.sub(
        rf"(?<!{re.escape(ARCH_PACKAGE)}\.)\b{re.escape(from_slug)}\b",
        to_slug,
        text,
    )
    base = from_slug.removesuffix(_MODULEV3_SLUG_SUFFIX)
    text = re.sub(rf"\b{re.escape(_camel(base))}\b", to_arch_name, text)
    return text


def absolutize_escaping_imports(text: str, package: str) -> str:
    """Rewrite relative imports that climb out of the copied donor package.

    ModuleV3 donors reuse each other's code (``from ..llama3_modulev3.model
    import Llama3Model``). Copied into a custom-arch directory, a relative
    import that leaves the donor package resolves to nothing, so point it at
    ``max.pipelines.architectures`` by absolute name. ``package`` is the
    dotted package the file lived in inside the donor tree.
    """
    parts = package.split(".")
    donor_depth = len(ARCH_PACKAGE.split(".")) + 1

    def replace(match: re.Match[str]) -> str:
        level = len(match["dots"])
        target = parts[: len(parts) - (level - 1)]
        if len(target) >= donor_depth:
            return match[0]
        module = ".".join([*target, match["rest"]] if match["rest"] else target)
        return f"{match['lead']}{module}{match['tail']}"

    return _RELATIVE_IMPORT_RE.sub(replace, text)


def rename_relative_import(text: str, stem: str, slug: str, level: int) -> str:
    """Point ``from .<stem> import`` at ``<slug>``.

    ``level`` is the number of leading dots that reach the copied package's
    top level from the file being rewritten.
    """
    pattern = re.compile(
        rf"^(\s*from\s+){re.escape('.' * level)}{re.escape(stem)}(\s+import\b)",
        re.MULTILINE,
    )
    return pattern.sub(rf"\g<1>{'.' * level}{slug}\g<2>", text)


def _registration_name(call: ast.Call) -> str | None:
    """The ``name=`` string of a registration, direct or ``dataclasses.replace``.

    A donor can build one architecture from another with
    ``dataclasses.replace(arch, name=...)``, so both call shapes register.
    ``None`` for any other call.
    """
    func = call.func
    direct = isinstance(func, ast.Name) and func.id == "SupportedArchitecture"
    derived = (
        isinstance(func, ast.Attribute)
        and func.attr == "replace"
        and isinstance(func.value, ast.Name)
        and func.value.id == "dataclasses"
    )
    if not (direct or derived):
        return None
    return string_keyword(call, "name")


def export_architectures(dst: Path, arch_name: str) -> bool:
    """Add the ``ARCHITECTURES`` list ``--custom-architectures`` reads.

    Exports only the registration the port renamed to ``arch_name``.
    ``--custom-architectures`` registers every list entry with
    ``allow_override=True``, so a donor's unselected registration would
    shadow its built-in architecture. Returns whether ``__init__.py``
    changed.
    """
    init = dst / "__init__.py"
    arch = dst / "arch.py"
    if not arch.is_file():
        return False
    text = init.read_text() if init.is_file() else ""
    if "ARCHITECTURES" in text:
        return False
    names = [
        target.id
        for node in ast.parse(arch.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _registration_name(node.value) == arch_name
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    if not names:
        return False
    joined = ", ".join(names)
    init.write_text(
        f"{text.rstrip()}\n\nfrom .arch import {joined}\n\n"
        f"ARCHITECTURES = [{joined}]\n"
    )
    return True


def rename_registration(text: str, old_name: str, new_name: str) -> str:
    """Set ``name=`` of the ``SupportedArchitecture`` that registers ``old_name``.

    Rewrites the first call in the ``arch.py`` source ``text`` whose
    ``name=`` is the string ``old_name``. Other calls keep their names.
    """
    for call in supported_architecture_calls(ast.parse(text)):
        if string_keyword(call, "name") != old_name:
            continue
        value = next(kw.value for kw in call.keywords if kw.arg == "name")
        assert value.end_lineno is not None
        assert value.end_col_offset is not None
        start = source_offset(text, value.lineno, value.col_offset)
        end = source_offset(text, value.end_lineno, value.end_col_offset)
        return f'{text[:start]}"{new_name}"{text[end:]}'
    return text


def write_full_copy(
    *, donor: Donor, dst: Path, slug: str, arch_name: str
) -> list[Path]:
    """Copy the donor package into ``dst`` and rename it to the port.

    The file holding the donor's root module (``llama3.py`` for
    ``llama3_modulev3``) becomes ``<slug>.py``. Only the relative imports of
    that file change with it: the stem can be a common word (``graph`` for
    ``mpnet_modulev3``) that also names other modules and string values.

    Returns the paths written, relative to ``dst``.
    """
    src = donor.root / donor.slug
    module_stem = donor.module_stem
    stems = [donor.slug]
    if module_stem and module_stem != donor.slug:
        stems.append(module_stem)
    written = []
    for entry in sorted(src.rglob("*")):
        rel = entry.relative_to(src)
        if entry.is_dir() or "__pycache__" in rel.parts:
            continue
        if entry.suffix not in _COPY_SUFFIXES:
            continue
        new_name = entry.name
        if len(rel.parts) == 1 and entry.stem in stems:
            new_name = f"{slug}{entry.suffix}"
        target = dst / rel.parent / new_name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = entry.read_text()
        if entry.suffix == ".py":
            package = ".".join([ARCH_PACKAGE, donor.slug, *rel.parent.parts])
            text = absolutize_escaping_imports(text, package)
            if module_stem and module_stem != donor.slug:
                text = rename_relative_import(
                    text, module_stem, slug, len(rel.parent.parts) + 1
                )
        text = rewrite_text(text, donor.slug, slug, arch_name)
        if rel == Path("arch.py"):
            # The donor's name went through the same rewrite, so match the
            # rewritten form. A donor's ``_ModuleV3`` registry suffix drops
            # here: the port registers the plain HF architectures[0].
            renamed = rewrite_text(donor.name, donor.slug, slug, arch_name)
            text = rename_registration(text, renamed, arch_name)
        target.write_text(text)
        written.append(target.relative_to(dst))
    return written
