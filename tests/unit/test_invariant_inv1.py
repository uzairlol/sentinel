"""INV-1 guard: the instrumentation layer must never import evaluation code.

One of the two load-bearing design invariants is that capture performs *zero*
analysis (docs/adr/0003). This test guards the boundary structurally: importing
the public instrument surface must not pull ``sentinel.eval`` into the process,
and no source file under ``src/sentinel/instrument`` may reference the eval
namespace at all.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_INSTRUMENT_DIR = _SRC / "sentinel" / "instrument"

_PROBE = textwrap.dedent(
    """
    import sys

    import sentinel.instrument  # noqa: F401  (imported for its side effects)

    print("\\n".join(name for name in sys.modules if name.startswith("sentinel.eval")))
    """
)


def test_instrument_import_does_not_load_eval_modules() -> None:
    """Checked in a cold interpreter: a warm ``sys.modules`` is not evidence.

    Other tests in this suite legitimately import the eval package, so asking
    this question inside the shared process would only measure test ordering.
    """
    done = subprocess.run(  # noqa: S603  # fixed argv, no shell, this is our own probe
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.split() == [], done.stdout


def test_instrument_source_never_references_eval() -> None:
    offenders = [
        str(path.relative_to(_SRC))
        for path in sorted(_INSTRUMENT_DIR.rglob("*.py"))
        if "sentinel.eval" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
