"""Generic function/tool tracer (``S1-T10``).

``sentinel.instrument.trace`` decorates a custom function or tool so each
call becomes a pair of ``agent.step`` events (``status: started`` /
``status: ended``) in the session currently open on the calling task, with
the bound arguments, the return value, and the measured latency:

.. code-block:: python

    from sentinel.instrument import trace

    @trace(kind="retrieval")
    async def search(query: str) -> list[str]: ...

    async with session(store) as ctx:
        results = await search("documents")   # recorded as agent.step

The decorator is a pure capture hook (INV-1): it never changes the
function's behavior or return value, and it fails open (INV-6) — when no
session is active, the session is closed, or capture itself raises, the
call proceeds untraced. A traced call is linked as the ``parent`` of
nested traced calls, so a multi-step custom workflow produces the same
navigable tree shape as the framework instrumentors (``S1-T6``).

The active session is the one entered by :func:`sentinel.session` and
tracked through a task-local context variable (``current_session``;
capture targets asyncio agents). Synchronous callables are passed through
untraced, mirroring the sync-inside-the-blind agent-loop limitation
documented for the LangChain handler.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from typing import Any, TypeVar, cast

import structlog

from sentinel.instrument.langchain import _capped, _clock, _latency_ms
from sentinel.instrument.session import SessionContext, current_session
from sentinel.models.events import AGENT_STEP, ERROR, Event, RefKind, RefLink

log = structlog.get_logger("sentinel.generic")

#: Event ids of traced runs currently on the async stack, for nesting only.
_STACK: ContextVar[tuple[str, ...]] = ContextVar("sentinel.trace.stack", default=())

F = TypeVar("F", bound=Callable[..., Any])


def _parent_refs(stack: tuple[str, ...]) -> list[RefLink]:
    if not stack:
        return []
    return [RefLink(event_id=stack[-1], kind=RefKind.PARENT)]


async def _capture(
    ctx: SessionContext,
    *,
    type: str,  # noqa: A002
    payload: dict[str, object],
    refs: Sequence[RefLink] = (),
) -> Event | None:
    """Record one event, failing open (INV-6): capture never raises to the host."""
    try:
        return await ctx.capture(type=type, payload=payload, refs=tuple(refs))
    except Exception:
        log.warning("generic.capture.failed", exc_info=True)
        return None


async def _run_async(
    func: Callable[..., Awaitable[object]],
    *,
    kind: str,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 -- private bridge; result passes through untouched
    ctx = current_session()
    if ctx is None:
        return await func(*args, **kwargs)

    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs: dict[str, object] = dict(bound.arguments)
    except (TypeError, ValueError):  # signature not bindable; cap raw args
        inputs = {"args": _capped(args), "kwargs": _capped(kwargs)}

    stack = _STACK.get()
    started_event = await _capture(
        ctx,
        type=AGENT_STEP,
        payload={
            "provider": "generic",
            "step": kind,
            "function": name,
            "status": "started",
            "input": _capped(inputs),
        },
        refs=_parent_refs(stack),
    )
    if started_event is None:
        return await func(*args, **kwargs)

    started = _clock()
    token = _STACK.set((*stack, started_event.event_id))
    try:
        result = await func(*args, **kwargs)
    except BaseException as exc:
        await _capture(
            ctx,
            type=ERROR,
            payload={
                "provider": "generic",
                "scope": "function",
                "step": kind,
                "function": name,
                "message": _capped(str(exc)),
            },
            refs=[RefLink(event_id=started_event.event_id, kind=RefKind.PARENT)],
        )
        raise
    else:
        await _capture(
            ctx,
            type=AGENT_STEP,
            payload={
                "provider": "generic",
                "step": kind,
                "function": name,
                "status": "ended",
                "latency_ms": _latency_ms(started),
                "output": _capped(result),
            },
            refs=[RefLink(event_id=started_event.event_id, kind=RefKind.PARENT)],
        )
        return result
    finally:
        _STACK.reset(token)


def _decorate(func: Callable[..., Any], *, kind: str) -> Callable[..., Any]:
    name = func.__qualname__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            result = _run_async(func, kind=kind, name=name, args=args, kwargs=kwargs)
            return await result

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        log.debug("generic.sync.passthrough", function=name)
        return func(*args, **kwargs)

    return sync_wrapper


def trace(*, kind: str = "function") -> Callable[[F], F]:
    """Decorate a custom function/tool so each call is captured in the session.

    Each traced call records an ``agent.step`` pair in the
    :func:`sentinel.session` active on the calling task — ``status:
    started`` with the bound arguments, ``status: ended`` with the return
    value and ``latency_ms``, or an ``error`` event when it raises (the
    original exception then propagates unchanged). ``kind`` labels the step
    and defaults to ``"function"`` (e.g. ``"retrieval"``, ``"web_search"``).

    Decorated functions are reference-transparent: results, exceptions, and
    side effects are untouched, and capture never raises into the host
    (INV-6). Sync callables pass through untraced (asyncio-only capture).
    """
    if not isinstance(kind, str) or not kind.strip():
        raise TypeError("trace(kind=...) requires a non-empty string")

    def decorator(func: F) -> F:
        return cast(F, _decorate(func, kind=kind))

    return decorator
