"""INV-1 guard: the instrumentation layer must never import evaluation code.

One of the two load-bearing design invariants is that capture performs *zero*
analysis (docs/adr/0003). This test guards the boundary structurally: importing
the public instrument surface must not pull ``sentinel.eval`` into the process,
and no source file under ``src/sentinel/instrument`` may reference the eval
namespace at all.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import sentinel.instrument  # noqa: F401  (imported for its side effects)

_SRC = Path(__file__).resolve().parents[2] / "src"
_INSTRUMENT_DIR = _SRC / "sentinel" / "instrument"


def test_instrument_import_does_not_load_eval_modules() -> None:
    del sys.modules["sentinel"]
    del sys.modules["sentinel.instrument"]
    importlib.import_module("sentinel.instrument")
    eval_modules = [name for name in sys.modules if name.startswith("sentinel.eval")]
    assert eval_modules == []


def test_instrument_source_never_references_eval() -> None:
    offenders = [
        str(path.relative_to(_SRC))
        for path in sorted(_INSTRUMENT_DIR.rglob("*.py"))
        if "sentinel.eval" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
