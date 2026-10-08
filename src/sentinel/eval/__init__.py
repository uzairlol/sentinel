"""Evaluation modules: read the log, write flags (``docs/adr/0004``).

``sentinel.eval`` is the only part of the SDK that judges an agent, and it is
deliberately one-directional:

* It never imports ``sentinel.instrument`` or ``sentinel.capture``. INV-1 makes
  capture a *producer* of events and evaluation a *consumer* of them; an
  evaluator that could reach back into the host process would couple a
  background detector to the latency of the agent it watches.
* It never writes events. The store's flag surface (``S3-T1``) is the only way
  out, so a detector cannot forge evidence and the log stays append-only.
* It is importable but not re-exported from ``sentinel`` (ADR-0009): the small
  public surface is for host applications, and an evaluator is an operator-side
  concern.

The pieces:

* :mod:`sentinel.eval.session` — one session's events plus its call graph.
* :mod:`sentinel.eval.worker` — triggering, idempotency, retry, checkpointing.
* :mod:`sentinel.eval.provenance_core` — the pure grounding rules, shared.
* :mod:`sentinel.eval.provenance` — the tool-use grounding module (``S3``).
* :mod:`sentinel.eval.embeddings` — embedding providers (``S4-T1``-``T3``).
* :mod:`sentinel.eval.memory_core` — the pure memory-integrity rules.
* :mod:`sentinel.eval.memory` — the memory-integrity module (``S4``).
* :mod:`sentinel.eval.harness` — the adversarial corpora and the FP/FN gate.
"""

from __future__ import annotations

from sentinel.eval.memory import (
    CATEGORY_COLLAPSE,
    CATEGORY_DRIFT,
    CATEGORY_MEMORY_UNGROUNDED,
    MemoryIntegrityConfig,
    MemoryIntegrityEvaluator,
)
from sentinel.eval.memory import (
    MODULE as MEMORY_MODULE,
)
from sentinel.eval.provenance import (
    CATEGORY_CONTRADICTED,
    CATEGORY_UNGROUNDED,
    ProvenanceEvaluator,
    ProvenanceResult,
)
from sentinel.eval.provenance import (
    MODULE as PROVENANCE_MODULE,
)
from sentinel.eval.session import SessionView
from sentinel.eval.worker import (
    CheckpointStore,
    EvaluatorWorker,
    InMemoryCheckpointStore,
    SqliteCheckpointStore,
    WorkerConfig,
    WorkerRun,
)

__all__ = [
    "CATEGORY_COLLAPSE",
    "CATEGORY_CONTRADICTED",
    "CATEGORY_DRIFT",
    "CATEGORY_MEMORY_UNGROUNDED",
    "CATEGORY_UNGROUNDED",
    "MEMORY_MODULE",
    "PROVENANCE_MODULE",
    "CheckpointStore",
    "EvaluatorWorker",
    "InMemoryCheckpointStore",
    "MemoryIntegrityConfig",
    "MemoryIntegrityEvaluator",
    "ProvenanceEvaluator",
    "ProvenanceResult",
    "SessionView",
    "SqliteCheckpointStore",
    "WorkerConfig",
    "WorkerRun",
]
