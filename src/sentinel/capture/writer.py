"""The asynchronous, fail-open capture pipeline (``S1-T12``, ``S1-T13``).

:class:`BatchedWriter` sits between the host agent and the event store. It:

* accepts events on a *bounded* queue — submission never blocks the host
  (INV-6). If the queue is full the event is counted as dropped and a
  ``capture.dropped`` marker is appended when room allows (``S1-T12``).
* flushes in batches, running the redaction hook (``S1-T14``) over each event
  before it reaches the store.
* fails open by default: an exception while persisting is logged, counted, and
  swallowed so the agent keeps running (``S1-T13``). With ``fail_open=False``
  the first persistence error is recorded, the writer stops, and the next
  :meth:`BatchedWriter.submit` raises it.

Events are appended strictly in submission order (one worker, one store), which
preserves per-session ``seq`` order and keeps ``refs`` resolvable at append
time (INV-3).

Shutdown uses a stop flag: the worker waits for the next event with a timeout
equal to the flush interval, so closing a quiet writer settles within one
interval without a typed sentinel value polluting the queue.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import structlog

from sentinel.config import get_config
from sentinel.models.events import CAPTURE_DROPPED, Event, make_event
from sentinel.redact import redact_payload
from sentinel.store.protocol import EventStore

log = structlog.get_logger("sentinel.capture")


class CapturePipelineError(RuntimeError):
    """Raised by :meth:`BatchedWriter.submit` after a fail-closed fatal error."""


class BatchedWriter:
    """Batch-append events from a bounded queue, failing open by default."""

    def __init__(
        self,
        store: EventStore,
        *,
        queue_max_size: int = 10_000,
        batch_max_size: int = 100,
        flush_interval_ms: float = 250.0,
        fail_open: bool = True,
        redactor: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Create a writer that persists to *store* through a bounded queue.

        ``queue_max_size`` bounds how many events may wait before the host is
        forced to back-pressure (see :meth:`submit`, INV-6); ``batch_max_size``
        and ``flush_interval_ms`` control how many events ride each batched
        flush; ``fail_open`` toggles the ``S1-T13`` failure boundary; a custom
        ``redactor`` replaces the default ``S1-T14`` policy.
        """
        self._store = store
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_max_size)
        self._batch_max = max(1, batch_max_size)
        self._flush_interval = max(0.0, flush_interval_ms / 1000.0)
        self._fail_open = fail_open
        self._redactor = redactor or _default_redactor
        self._task: asyncio.Task[None] | None = None
        self._fatal: BaseException | None = None
        self._stopping = False
        self._wake = asyncio.Event()
        self._submitted = 0
        self._handled = 0

        #: Events dropped on a full queue (never persisted).
        self.dropped = 0
        #: Events that failed to persist (capture failures, fail-open only).
        self.failed = 0
        #: The most recent persistence error (or ``None``).
        self.last_error: BaseException | None = None

    @property
    def fatal_error(self) -> BaseException | None:
        """The error that stopped a fail-closed writer, if any."""
        return self._fatal

    @property
    def queued(self) -> int:
        """Number of events waiting in the queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Spawn the background worker. Idempotent."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="sentinel-capture-writer")

    async def submit(self, event: Event) -> None:
        """Queue *event* for async persistence; never blocks the host."""
        if self._fatal is not None:
            raise CapturePipelineError(
                "capture pipeline stopped (fail-closed); event not captured"
            ) from self._fatal
        await self.start()
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            log.warning("capture.queue_full", event_id=event.event_id, session_id=event.session_id)
            self._queue_marker(event, reason="queue_full")
            return
        self._submitted += 1
        self._wake.set()

    async def flush(self) -> None:
        """Wait until every submitted event has been persisted (or failed)."""
        while self._fatal is None and self._handled < self._submitted:
            await asyncio.sleep(0)

    async def close(self) -> None:
        """Drain pending events, then stop the worker."""
        if self._task is None:
            return
        if self._task.done():
            self._task = None
            return
        while self._fatal is None and self._handled < self._submitted:
            await asyncio.sleep(0)
        self._stopping = True
        self._wake.set()
        await self._task
        self._task = None

    def _queue_marker(self, dropped: Event, *, reason: str) -> None:
        """Queue a ``capture.dropped`` marker when room allows.

        Carries no ``refs``: the dropped event was never persisted, so a
        structural link would be unresolvable (INV-3). Its id rides in the
        payload instead.
        """
        marker = make_event(
            session_id=dropped.session_id,
            seq=dropped.seq,
            type=CAPTURE_DROPPED,
            payload={
                "reason": reason,
                "count": 1,
                "dropped_event_id": dropped.event_id,
            },
        )
        try:
            self._queue.put_nowait(marker)
            self._submitted += 1
            self._wake.set()
        except asyncio.QueueFull:
            log.debug("capture.marker_dropped", session_id=dropped.session_id)

    async def _run(self) -> None:
        while True:
            if self._stopping and self._queue.empty():
                return
            if self._queue.empty():
                self._wake.clear()
                if self._stopping:
                    return
                await self._wake.wait()
                continue

            self._wake.clear()
            try:
                first = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
            self._queue.task_done()

            batch: list[Event] = [first]
            while len(batch) < self._batch_max:
                try:
                    nxt = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._queue.task_done()
                batch.append(nxt)

            for event in batch:
                try:
                    await self._append(event)
                except BaseException as exc:
                    self.last_error = exc
                    if not self._fail_open:
                        self._fatal = exc
                        log.error("capture.fatal", error=repr(exc))
                        return
                    self.failed += 1
                    log.warning("capture.persist_failed", error=repr(exc))
                    if event.type != CAPTURE_DROPPED:
                        self._queue_marker(event, reason="store_error")
                finally:
                    self._handled += 1

            if self._flush_interval > 0:
                await asyncio.sleep(self._flush_interval)

    async def _append(self, event: Event) -> None:
        """Redact then persist one event."""
        redacted = self._redactor(event.payload)
        await self._store.append(event.model_copy(update={"payload": redacted}))


def _default_redactor(payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = get_config()
    if not settings.redaction_enabled:
        return dict(payload)
    return redact_payload(payload, patterns=settings.redaction_patterns)
