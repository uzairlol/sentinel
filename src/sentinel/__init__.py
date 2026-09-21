"""Sentinel — runtime safety instrumentation for autonomous LLM agents.

This is the public package surface. Everything exported here is stable and
semver-managed. Anything reachable only through private modules (named with a
leading underscore) may change without notice. See docs/adr/0009.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
