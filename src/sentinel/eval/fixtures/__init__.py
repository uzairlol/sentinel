"""Hand-written fixtures: the adversarial provenance corpus (``S3-T12``).

Nothing in here is generated. Every case states its own expectation so a
regression in the rules fails a test rather than quietly shifting the FP/FN rate
the sprint gates on.
"""

from __future__ import annotations

from sentinel.eval.fixtures.provenance_corpus import (
    CONTRADICTED,
    CORPUS,
    GROUNDED,
    MUST_FLAG,
    MUST_NOT_FLAG,
    UNGROUNDED,
    CorpusCase,
    ExpectedFinding,
    case_by_id,
)

__all__ = [
    "CONTRADICTED",
    "CORPUS",
    "GROUNDED",
    "MUST_FLAG",
    "MUST_NOT_FLAG",
    "UNGROUNDED",
    "CorpusCase",
    "ExpectedFinding",
    "case_by_id",
]
