"""The evaluator worker framework (sprint ``S3-T2``/``S3-T3``, docs/adr/0004).

An evaluator is an independent worker: it reads finished sessions out of the
event store, decides whether anything is wrong, and writes
:class:`~sentinel.models.flags.Flag` rows back into the same store. This module
supplies everything a module should not have to re-implement:

* **Triggering** — a watermark over completed sessions (``session.end``
  bookends) plus an on-demand :meth:`EvaluatorWorker.evaluate_session` for the
  gate path (``S3-T3``).
* **Idempotency** — a checkpoint keyed by ``(module, module_version,
  session_id)``. A restart resumes where it stopped; a re-run of the same module
  version neither re-evaluates nor duplicates a flag.
* **Bounded work** — one session, capped events, batched flag writes.
* **Retry with backoff** — a transient store failure is retried; a deterministic
  module bug surfaces immediately rather than being retried forever.
* **Determinism** — :meth:`EvaluatorWorker.evaluate` is a pure function of the
  :class:`~sentinel.eval.session.SessionView`; the base class owns everything
  that would otherwise vary (ids, timestamps, ordering), which is what makes a
  published FP/FN rate reproducible (``S3-T4``).

Checkpoints live in their own store (``S3-T2``). The durable implementation is a
single-file SQLite table so a restarted worker never redoes finished work; the
in-memory implementation is for tests and one-shot CLI runs. Both are behind the
:class:`CheckpointStore` protocol, so moving the trigger onto a database queue
in ``S7`` does not touch a module.
"""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import aiosqlite
import structlog

from sentinel.eval.session import SessionView
from sentinel.models.flags import Flag
from sentinel.store.protocol import EventStore

log = structlog.get_logger("sentinel.eval.worker")

#: Status the watermark trigger requires before it evaluates a session.
SESSION_STATUS_COMPLETE = "ended"


class EvaluationError(RuntimeError):
    """Raised when a module fails; carries the session for the retry loop."""


class DeterministicEvaluationError(EvaluationError):
    """A module failure that will recur identically — never retried.

    Raised for malformed input a module cannot make sense of, and for bugs that
    would produce the same output again. Retrying those only burns budget and
    hides the defect behind a backoff.
    """


@dataclass(frozen=True)
class WorkerConfig:
    """Per-worker tuning. Every field has a safe default (``S2.10`` 12-factor)."""

    #: Module name recorded on every flag (``Flag.module``).
    module: str
    #: Module version, part of the idempotency key: bumping it re-evaluates.
    module_version: str
    #: Max events of one session handed to :meth:`EvaluatorWorker.evaluate`.
    max_events: int = 5_000
    #: Flags written per store round trip.
    flag_batch_size: int = 100
    #: Attempts per session before a failure is recorded and left behind.
    max_attempts: int = 3
    #: Exponential backoff base; attempt *n* waits ``base * 2**(n-1)`` + jitter.
    backoff_base_s: float = 0.05
    backoff_max_s: float = 5.0
    #: Only sessions with this store status are picked up by the watermark.
    complete_status: str = SESSION_STATUS_COMPLETE
    #: ``review_only`` is set on any flag at or below this confidence
    #: (``S3-T15``); those never gate, they queue for a human.
    review_confidence_threshold: float = 0.0

    @property
    def idempotency_namespace(self) -> str:
        """The ``module@version`` string a checkpoint is keyed under."""
        return f"{self.module}@{self.module_version}"


@dataclass
class WorkerRun:
    """The outcome of one :meth:`EvaluatorWorker.run_once` pass."""

    sessions_evaluated: int = 0
    sessions_skipped: int = 0
    sessions_failed: int = 0
    flags_written: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def total_sessions(self) -> int:
        """Every session the pass looked at, evaluated or not."""
        return self.sessions_evaluated + self.sessions_skipped


class CheckpointStore(Protocol):
    """Durable record of which sessions a module version has already finished."""

    async def is_done(self, namespace: str, session_id: str) -> bool:
        """Whether *session_id* already completed for *namespace*."""

    async def mark_done(
        self,
        namespace: str,
        session_id: str,
        *,
        flag_ids: Sequence[str] = (),
        evaluated_at: datetime | None = None,
    ) -> None:
        """Record a finished evaluation and the flags it produced."""

    async def flag_ids(self, namespace: str, session_id: str) -> tuple[str, ...]:
        """Flag ids recorded for a finished evaluation (empty if not done)."""

    async def record_failure(self, namespace: str, session_id: str, error: str) -> int:
        """Record a failed attempt and return the total attempts so far."""

    async def attempts(self, namespace: str, session_id: str) -> int:
        """How many times this session has failed for *namespace*."""

    async def clear(self, namespace: str | None = None) -> None:
        """Drop checkpoints (used when a module version is retired)."""

    async def close(self) -> None:
        """Release resources."""


class InMemoryCheckpointStore:
    """A checkpoint store that lives as long as the process (tests, one-shot runs)."""

    def __init__(self) -> None:
        """Create an empty checkpoint set."""
        self._done: dict[tuple[str, str], tuple[str, ...]] = {}
        self._failures: dict[tuple[str, str], int] = {}

    async def is_done(self, namespace: str, session_id: str) -> bool:
        """Whether *session_id* already completed for *namespace*."""
        return (namespace, session_id) in self._done

    async def mark_done(
        self,
        namespace: str,
        session_id: str,
        *,
        flag_ids: Sequence[str] = (),
        evaluated_at: datetime | None = None,
    ) -> None:
        """Record a finished evaluation and the flags it produced.

        Any recorded failure is cleared: the session is done, so its attempt
        count has nothing left to say.
        """
        del evaluated_at
        key = (namespace, session_id)
        self._done[key] = tuple(flag_ids)
        self._failures.pop(key, None)

    async def flag_ids(self, namespace: str, session_id: str) -> tuple[str, ...]:
        """Flag ids recorded for a finished evaluation (empty if not done)."""
        return self._done.get((namespace, session_id), ())

    async def record_failure(self, namespace: str, session_id: str, error: str) -> int:
        """Count a failed attempt and return the new total."""
        del error
        key = (namespace, session_id)
        self._failures[key] = self._failures.get(key, 0) + 1
        return self._failures[key]

    async def attempts(self, namespace: str, session_id: str) -> int:
        """How many times this session has failed for *namespace*."""
        return self._failures.get((namespace, session_id), 0)

    async def clear(self, namespace: str | None = None) -> None:
        """Drop checkpoints, or only those of one namespace."""
        if namespace is None:
            self._done.clear()
            self._failures.clear()
            return
        for key in [k for k in self._done if k[0] == namespace]:
            del self._done[key]
        for key in [k for k in self._failures if k[0] == namespace]:
            del self._failures[key]

    async def close(self) -> None:
        """No resources to release."""


_CHECKPOINT_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_checkpoints (
    namespace    TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    evaluated_at TEXT,
    flag_ids     TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (namespace, session_id)
);
CREATE TABLE IF NOT EXISTS eval_failures (
    namespace  TEXT NOT NULL,
    session_id TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, session_id)
);
"""


class SqliteCheckpointStore:
    """A durable, single-file checkpoint store backed by SQLite.

    Deliberately *not* part of the event store: worker progress is operational
    state, not evidence, and it must be writable by a credential that cannot
    touch the append-only log. Restarting the worker (or the box) with the same
    file resumes exactly where it stopped (``S3-T2``).
    """

    def __init__(self, path: str = "sentinel-eval-checkpoints.sqlite3") -> None:
        """Create a store backed by *path*; the file is opened on first use."""
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _connection(self) -> aiosqlite.Connection:
        """Open the database and create the schema exactly once."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._path)
            await self._conn.executescript(_CHECKPOINT_SCHEMA)
            await self._conn.commit()
        return self._conn

    async def is_done(self, namespace: str, session_id: str) -> bool:
        """Whether *session_id* already completed for *namespace*."""
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT 1 FROM eval_checkpoints WHERE namespace = ? AND session_id = ?",
            (namespace, session_id),
        )
        return await cursor.fetchone() is not None

    async def mark_done(
        self,
        namespace: str,
        session_id: str,
        *,
        flag_ids: Sequence[str] = (),
        evaluated_at: datetime | None = None,
    ) -> None:
        """Record a finished evaluation and the flags it produced."""
        conn = await self._connection()
        stamp = (evaluated_at or datetime.now(UTC)).isoformat()
        async with self._lock:
            await conn.execute(
                "INSERT INTO eval_checkpoints (namespace, session_id, evaluated_at, flag_ids) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (namespace, session_id) DO UPDATE SET "
                "evaluated_at = excluded.evaluated_at, flag_ids = excluded.flag_ids",
                (namespace, session_id, stamp, json.dumps(list(flag_ids))),
            )
            await conn.execute(
                "DELETE FROM eval_failures WHERE namespace = ? AND session_id = ?",
                (namespace, session_id),
            )
            await conn.commit()

    async def flag_ids(self, namespace: str, session_id: str) -> tuple[str, ...]:
        """Flag ids recorded for a finished evaluation (empty if not done)."""
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT flag_ids FROM eval_checkpoints WHERE namespace = ? AND session_id = ?",
            (namespace, session_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return ()
        return tuple(json.loads(row[0]))

    async def record_failure(self, namespace: str, session_id: str, error: str) -> int:
        """Count a failed attempt and return the new total."""
        conn = await self._connection()
        async with self._lock:
            await conn.execute(
                "INSERT INTO eval_failures "
                "(namespace, session_id, attempts, last_error, updated_at) "
                "VALUES (?, ?, 1, ?, ?) ON CONFLICT (namespace, session_id) DO UPDATE SET "
                "attempts = eval_failures.attempts + 1, last_error = excluded.last_error, "
                "updated_at = excluded.updated_at",
                (namespace, session_id, error, datetime.now(UTC).isoformat()),
            )
            await conn.commit()
            cursor = await conn.execute(
                "SELECT attempts FROM eval_failures WHERE namespace = ? AND session_id = ?",
                (namespace, session_id),
            )
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 1

    async def attempts(self, namespace: str, session_id: str) -> int:
        """How many times this session has failed for *namespace*."""
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT attempts FROM eval_failures WHERE namespace = ? AND session_id = ?",
            (namespace, session_id),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def clear(self, namespace: str | None = None) -> None:
        """Drop checkpoints, or only those of one namespace."""
        conn = await self._connection()
        async with self._lock:
            if namespace is None:
                await conn.execute("DELETE FROM eval_checkpoints")
                await conn.execute("DELETE FROM eval_failures")
            else:
                await conn.execute("DELETE FROM eval_checkpoints WHERE namespace = ?", (namespace,))
                await conn.execute("DELETE FROM eval_failures WHERE namespace = ?", (namespace,))
            await conn.commit()

    async def close(self) -> None:
        """Close the checkpoint file."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


class EvaluatorWorker(ABC):
    """Base class for every detection module (``S3-T2``).

    A subclass implements :meth:`evaluate` and nothing else. The base class
    supplies triggering, idempotency, bounded batching, retry/backoff, and the
    flag-writing contract.

    Example::

        class MyEvaluator(EvaluatorWorker):
            async def evaluate(self, session: SessionView) -> list[Flag]:
                return []

        worker = MyEvaluator(store, config=WorkerConfig("mine", "0.1.0"))
        run = await worker.run_once()
    """

    def __init__(
        self,
        store: EventStore,
        *,
        config: WorkerConfig,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        """Create a worker over *store*; ``checkpoints`` defaults to in-memory."""
        self._store = store
        self._config = config
        self._checkpoints: CheckpointStore = checkpoints or InMemoryCheckpointStore()
        self._owns_checkpoints = checkpoints is None

    # -- identity ---------------------------------------------------------

    @property
    def config(self) -> WorkerConfig:
        """The worker's immutable configuration."""
        return self._config

    @property
    def module(self) -> str:
        """The module name recorded on every flag this worker writes."""
        return self._config.module

    @property
    def module_version(self) -> str:
        """The module version, part of the idempotency key."""
        return self._config.module_version

    @property
    def namespace(self) -> str:
        """The ``module@version`` checkpoint namespace."""
        return self._config.idempotency_namespace

    @property
    def checkpoints(self) -> CheckpointStore:
        """The checkpoint store backing this worker's idempotency."""
        return self._checkpoints

    # -- the module's job -------------------------------------------------

    @abstractmethod
    async def evaluate(self, session: SessionView) -> list[Flag]:
        """Return the flags *session* deserves. Pure: same view, same flags.

        Implementations must not touch the store, the clock, or the network:
        everything they need is on the view, and every flag they build must
        derive its ``created_at`` from an event timestamp so two runs are
        byte-identical (``S3-T4``).
        """

    # -- on-demand trigger (``S3-T3``) ------------------------------------

    async def evaluate_session(
        self,
        session_id: str,
        *,
        force: bool = False,
    ) -> list[Flag]:
        """Evaluate *session_id* now and return the flags it produced.

        The gate path calls this directly at a checkpoint (``S3-T3``). With
        ``force=False`` an already-finished session is skipped and its recorded
        flag ids are returned as ``[]`` (the flags are already in the store);
        ``force=True`` re-runs the module, which is safe because flag writes are
        idempotent by ``flag_id``.
        """
        if not force and await self._checkpoints.is_done(self.namespace, session_id):
            log.debug("eval.skipped", module=self.module, session_id=session_id)
            return []
        view = await SessionView.load(self._store, session_id, max_events=self._config.max_events)
        flags = await self._run_with_retry(view)
        await self._write_flags(flags)
        await self._checkpoints.mark_done(
            self.namespace,
            session_id,
            flag_ids=[flag.flag_id for flag in flags],
        )
        return flags

    # -- watermark trigger (``S3-T3``) ------------------------------------

    async def pending_sessions(self, *, limit: int = 100, offset: int = 0) -> list[str]:
        """Session ids that look complete and have not been evaluated yet.

        "Complete" means the store recorded the session as ended, which happens
        when the ``session.end`` bookend is appended (``S1``). A session that is
        still active is never evaluated: its log can still grow. *limit* caps
        the work one pass may plan, and *offset* pages past the head of the
        list for a worker that is catching up.
        """
        rows = await self._store.list_sessions(limit=limit, offset=offset)
        pending: list[str] = []
        for row in rows:
            if row.status != self._config.complete_status:
                continue
            if not await self._checkpoints.is_done(self.namespace, row.session_id):
                pending.append(row.session_id)
        return pending

    async def run_once(self, *, limit: int = 50, offset: int = 0) -> WorkerRun:
        """Evaluate up to *limit* pending sessions; never raises on a failure.

        One bad session must not stall the module, so a failure is recorded,
        counted, and the pass continues. ``errors`` carries the messages.
        """
        started = datetime.now(UTC)
        run = WorkerRun()
        for session_id in await self.pending_sessions(limit=limit, offset=offset):
            try:
                flags = await self.evaluate_session(session_id)
            except DeterministicEvaluationError as exc:
                await self._checkpoints.record_failure(self.namespace, session_id, str(exc))
                run.sessions_failed += 1
                run.errors.append(f"{session_id}: {exc}")
                log.error("eval.failed", module=self.module, session_id=session_id, error=str(exc))
                continue
            except Exception as exc:
                run.sessions_failed += 1
                run.errors.append(f"{session_id}: {exc}")
                log.error("eval.failed", module=self.module, session_id=session_id, error=repr(exc))
                continue
            run.sessions_evaluated += 1
            run.flags_written += len(flags)
        run.duration_s = (datetime.now(UTC) - started).total_seconds()
        return run

    async def run_forever(
        self,
        *,
        interval_s: float = 5.0,
        stop: asyncio.Event | None = None,
        limit: int = 50,
    ) -> None:
        """Poll for pending sessions until *stop* is set.

        Sleeps between passes; a pass that finds nothing returns immediately, so
        an idle worker costs one list query per interval.
        """
        stop = stop or asyncio.Event()
        while not stop.is_set():
            await self.run_once(limit=limit)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except TimeoutError:
                continue

    async def close(self) -> None:
        """Release resources the worker owns (never the caller's store)."""
        if self._owns_checkpoints:
            await self._checkpoints.close()

    # -- internals --------------------------------------------------------

    async def _run_with_retry(self, view: SessionView) -> list[Flag]:
        """Run :meth:`evaluate` with bounded retries and exponential backoff."""
        last_error: BaseException | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await self.evaluate(view)
            except DeterministicEvaluationError:
                raise
            except Exception as exc:
                last_error = exc
                await self._checkpoints.record_failure(self.namespace, view.session_id, str(exc))
                if attempt >= self._config.max_attempts:
                    break
                delay = self._backoff_delay(attempt)
                log.warning(
                    "eval.retry",
                    module=self.module,
                    session_id=view.session_id,
                    attempt=attempt,
                    delay_s=delay,
                    error=repr(exc),
                )
                await asyncio.sleep(delay)
        raise EvaluationError(
            f"{self.module} failed on session {view.session_id} after "
            f"{self._config.max_attempts} attempts: {last_error!r}"
        ) from last_error

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped by the config."""
        base = self._config.backoff_base_s * (2 ** (attempt - 1))
        ceiling = min(base, self._config.backoff_max_s)
        return random.uniform(0.0, ceiling)  # noqa: S311  # nosec B311  # jitter, not cryptography

    async def _write_flags(self, flags: Sequence[Flag]) -> None:
        """Write *flags* in batches, routing low-confidence ones to review."""
        if not flags:
            return
        threshold = self._config.review_confidence_threshold
        for start in range(0, len(flags), self._config.flag_batch_size):
            batch = [
                flag
                if flag.confidence > threshold or flag.review_only
                else flag.model_copy(update={"review_only": True})
                for flag in flags[start : start + self._config.flag_batch_size]
            ]
            await self._store.put_flags(batch)


__all__ = [
    "SESSION_STATUS_COMPLETE",
    "CheckpointStore",
    "DeterministicEvaluationError",
    "EvaluationError",
    "EvaluatorWorker",
    "InMemoryCheckpointStore",
    "SqliteCheckpointStore",
    "WorkerConfig",
    "WorkerRun",
]
