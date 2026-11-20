"""Binding-surface drift test for the `quantumforge` Python package.

Three checks, all of which are cheap and all of which catch real regressions:

1. Every public name the compiled extension offers is re-exported by the
   `python/quantumforge/__init__.py` shim. Without this, a new `#[pyfunction]`
   lands in the wheel but is invisible to `import quantumforge`.
2. Every name in `__all__` resolves on the package.
3. Every name in `__all__` is documented in `python/API.md`. Run from a
   checkout; skipped when that file is not present (an installed wheel does not
   carry the docs).

Run after installing a wheel:

    python python/test_exports.py

Exit code is non-zero with a readable list on any drift.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import quantumforge as qf
from quantumforge import quantumforge as _ext

failures: list[str] = []


def shim_imports() -> tuple[set[str], set[str]]:
    """(extension names imported, names bound by the shim).

    The two differ for the aliased classes: the extension offers `PyCircuit`,
    the shim binds it as `Circuit`. Both directions matter -- an extension name
    nobody imports is dead, and an `__all__` entry the shim does not bind is a
    broken re-export.
    """
    source = Path(qf.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "quantumforge":
            for alias in node.names:
                imported.add(alias.name)
                bound.add(alias.asname or alias.name)
    return imported, bound


def check_extension_fully_reexported() -> None:
    imported, _ = shim_imports()
    public = {name for name in dir(_ext) if not name.startswith("_")}
    missing = sorted(public - imported)
    if missing:
        failures.append(
            "compiled extension names not re-exported by the shim: " + ", ".join(missing)
        )
    stale = sorted(imported - public)
    if stale:
        failures.append(
            "shim re-exports names the extension no longer offers: " + ", ".join(stale)
        )


def check_all_resolves() -> None:
    unresolvable = [name for name in qf.__all__ if not hasattr(qf, name)]
    if unresolvable:
        failures.append("__all__ names that do not resolve: " + ", ".join(unresolvable))
    _, bound = shim_imports()
    listed = set(qf.__all__)
    unbound = sorted(listed - bound)
    if unbound:
        failures.append(
            "__all__ names the shim does not bind: " + ", ".join(unbound)
        )
    extra = sorted(bound - listed)
    if extra:
        failures.append(
            "shim exports names missing from __all__: " + ", ".join(extra)
        )


def check_documented_in_api_reference() -> None:
    # An explicit path is a hard requirement: CI passes the checkout path so a
    # missing reference page fails instead of silently skipping.
    if len(sys.argv) > 1:
        api = Path(sys.argv[1])
        if not api.is_file():
            failures.append(f"API reference not found at {api}")
            return
    else:
        api = Path("python/API.md")
        if not api.is_file():
            print("note: python/API.md not found; skipping documentation coverage check")
            return
    documented = api.read_text(encoding="utf-8")
    missing = [name for name in qf.__all__ if name not in documented]
    if missing:
        failures.append(
            "__all__ names missing from python/API.md: " + ", ".join(sorted(missing))
        )


def main() -> int:
    check_extension_fully_reexported()
    check_all_resolves()
    check_documented_in_api_reference()
    if failures:
        print("quantumforge binding surface drift:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"quantumforge binding surface OK "
        f"({len(qf.__all__)} names, all re-exported and resolvable)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
