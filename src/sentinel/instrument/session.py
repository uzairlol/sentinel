"""Capture session management (INV-1: capture, never analysis).

``session()`` opens a bounded capture context: it allocates a ``session_id``,
assigns strictly monotonic ``seq`` numbers, and appends ``session.start`` /
``session.end`` bookends. Call sites capture boundary crossings through
:meth:`SessionContext.capture`; the enumeration lives in
:class:`~sentinel.instrument.session.SessionContext`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from sentinel.capture.writer import BatchedWriter
from sentinel.models.events import (
    SESSION_END,
    SESSION_START,
    Event,
    RefLink,
    make_event,
    new_event_id,
)
from sentinel.store.protocol import EventStore


class SessionClosedError(RuntimeError):
    """Raised when a :class:`SessionContext` receives a capture after ``end()``."""


#: The session :func:`session` has opened in the current task, if any. Set on
#: entering the context and reset on exit, so nested sessions shadow correctly.
_CURRENT_SESSION: ContextVar[SessionContext | None] = ContextVar(
    "sentinel.current_session", default=None
)


def current_session() -> SessionContext | None:
    """Return the :class:`SessionContext` active in this task, if any."""
    return _CURRENT_SESSION.get()


class SessionContext:
    """One capture session: owns a ``session_id`` and a monotonic ``seq``."""

    def __init__(self, store: EventStore, writer: BatchedWriter | None = None) -> None:
        """Open an unbounded capture session against *store* or *writer*."""
        self._store = store
        self._writer = writer
        self._seq = -1
        self._closed = False
        self.session_id = new_event_id()

    async def _emit(
        self,
        *,
        type: str,  # noqa: A002
        payload: Mapping[str, Any],
        refs: Sequence[RefLink | str],
    ) -> Event:
        event = make_event(
            session_id=self.session_id,
            seq=self._seq + 1,
            type=type,
            payload=dict(payload),
            refs=refs,
        )
        if self._writer is not None:
            await self._writer.submit(event)
        else:
            await self._store.append(event)
        self._seq = event.seq
        return event

    async def start(self) -> Event:
        """Open the session by recording ``session.start``."""
        return await self._emit(type=SESSION_START, payload={}, refs=())

    async def capture(
        self,
        *,
        type: str,  # noqa: A002
        payload: Mapping[str, Any] | None = None,
        refs: Sequence[RefLink | str] = (),
    ) -> Event:
        """Append one event to the session and return its persisted form."""
        if self._closed:
            raise SessionClosedError(
                f"session {self.session_id} is closed; no further captures accepted"
            )
        return await self._emit(type=type, payload=payload or {}, refs=tuple(refs))

    async def end(self) -> Event | None:
        """Close the session by recording ``session.end``. Idempotent."""
        if self._closed:
            return None
        self._closed = True
        return await self._emit(type=SESSION_END, payload={}, refs=())

    @property
    def seq(self) -> int:
        """The highest ``seq`` assigned so far (``-1`` before ``start()``)."""
        return self._seq


@asynccontextmanager
async def session(
    store: EventStore, writer: BatchedWriter | None = None
) -> AsyncIterator[SessionContext]:
    """Open a capture session against *store* (or the *writer* pipeline).

    Records ``session.start`` on entry and ``session.end`` on exit, even when
    the inner body raises, so a session is always bracketed by its bookends.
    When a :class:`~sentinel.capture.writer.BatchedWriter` is supplied, events
    are routed through the async, fail-open pipeline and flushed on exit;
    otherwise they are appended directly to *store*.

    Example::

        async with session(store) as ctx:
            await ctx.capture(type="llm.request", payload={...})
    """
    ctx = SessionContext(store, writer=writer)
    started = False
    token = _CURRENT_SESSION.set(ctx)
    try:
        await ctx.start()
        started = True
        yield ctx
    finally:
        if started:
            await ctx.end()
        if writer is not None:
            await writer.flush()
        _CURRENT_SESSION.reset(token)
