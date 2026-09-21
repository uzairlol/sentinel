"""Sentinel command-line interface.

Private module. The `sentinel` console script is installed from here, but this
module is not part of the public API surface (docs/adr/0009).
"""

from __future__ import annotations

import argparse
import sys

from sentinel import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Runtime safety instrumentation for autonomous LLM agents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sentinel-sdk {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `sentinel` command. Returns a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
