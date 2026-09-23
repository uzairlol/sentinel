"""Capture pipeline infrastructure (``S1-T12``/``S1-T13``).

The public entry point is :class:`~sentinel.capture.writer.BatchedWriter`.
"""

from __future__ import annotations

from sentinel.capture.writer import BatchedWriter, CapturePipelineError

__all__ = ["BatchedWriter", "CapturePipelineError"]
