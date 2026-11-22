#!/usr/bin/env python3
"""Which public modules have no test coverage at all?

Line coverage says how much of a module is exercised; this says whether a
module is exercised AT ALL. That distinction matters here: `stabilizer` had no
test file referencing it and shipped seven bugs, four of them silent wrong
answers rather than panics.

A module counts as referenced when some file under `tests/`, `examples/` or
`python/` mentions it by name (`quantumforge::<module>`, `crate::<module>` or
`<module>::`). This is a proxy, not a measurement: it cannot see doctests
inside the module itself, and a mention is not an assertion. It exists to catch
the zero case cheaply.

Usage:
    python scripts/module_test_coverage.py          # report, exit 0
    python scripts/module_test_coverage.py --fail   # exit 1 if any module is uncovered
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODULE_RE = re.compile(r"^\s*pub mod ([a-z_][a-z0-9_]*)\s*;", re.MULTILINE)
SEARCH_DIRS = ("tests", "examples", "python")

# Modules whose public surface is reached through another module's methods (or
# through the Python bindings) rather than a path of their own. Listed
# explicitly so the report says how they are covered instead of implying they
# are not.
REACHED_VIA = {
    "serialize": "Circuit::{to_json, from_json, to_json_bytes, from_json_bytes} "
    "(defined in serialize.rs; exercised by tests/module_coverage.rs)",
    "python": "the PyO3 bindings, exercised end-to-end by python/test_api.py and "
    "python/smoke_test.py against a built wheel (CI job python-bindings)",
}


def public_modules() -> list[str]:
    return MODULE_RE.findall((ROOT / "src" / "lib.rs").read_text(encoding="utf-8"))


def references(module: str) -> list[str]:
    """Files under tests/, examples/ or python/ that mention the module."""
    hits: list[str] = []
    for directory in SEARCH_DIRS:
        for path in sorted((ROOT / directory).glob("*.rs")) + sorted(
            (ROOT / directory).glob("*.py")
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(rf"(quantumforge|crate)::{module}\b", text) or re.search(
                rf"\b{module}::", text
            ):
                hits.append(f"{directory}/{path.name}")
    return hits


def main() -> int:
    modules = public_modules()
    uncovered = [(m, references(m)) for m in modules]
    missing = [m for m, hits in uncovered if not hits]
    aliased = [m for m in missing if m in REACHED_VIA]
    gaps = [m for m in missing if m not in REACHED_VIA]
    covered = len(modules) - len(gaps)

    print(f"public modules: {len(modules)}")
    print(f"referenced from tests/, examples/ or python/: {len(modules) - len(missing)}")
    print(f"covered through another path: {len(aliased)}")
    print(f"no coverage at all: {len(gaps)}")
    for name in aliased:
        print(f"  ~ {name}: {REACHED_VIA[name]}")
    if gaps:
        print("\nno test or example references:")
        for name in gaps:
            print(f"  {name}")
    else:
        print("\nevery public module is referenced or reached through a listed path")

    ratio = covered / len(modules) if modules else 1.0
    print(f"\nmodule coverage: {ratio:.0%}")
    if "--fail" in sys.argv and gaps:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
