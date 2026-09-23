"""Capture LangChain LLM calls, tool invocations, and chain steps (``S1-T7``).

Sentinel does not wrap LangChain runnables; instead it plugs a callback handler
into LangChain's own callback system (``callbacks=[handler]``), so the agent's
execution flow is untouched and capture never intercepts the model's responses.

The handler maps each boundary crossing to an event:

* LLM / chat-model calls become ``llm.request`` and ``llm.response`` events
  (``on_llm_start`` / ``on_llm_end``; chat models fall through to these on the
  framework's fallback path).
* Tool invocations become ``tool.call`` and ``tool.result`` events.
* Chain execution becomes ``agent.step`` events with ``status: started`` /
  ``status: ended``.

Each response and result links back to its originating request with a
``caused_by`` reference, and every inner event carries a ``parent`` reference to
its enclosing chain step, so links resolve to already-persisted events at
append time (INV-3).

LangChain remains an optional dependency: this module imports nothing from
``langchain-core`` until :meth:`LangChainInstrumentor.handler` is called, so the
base distribution stays lightweight and ``import sentinel`` never fails when the
extra is missing.

``S1-T7`` builds on INV-1 (capture only, never analysis) and INV-6 (capture is
best-effort and must never raise into the host): a closed session, an
unavailable store, or a non-serializable payload is logged and skipped.

.. note::
   Capture targets asyncio agents. The handler methods are ``async def``, so
   LangChain awaits them directly during ``ainvoke``/``astream`` runs and event
   order matches execution order. A *synchronous* run inside a running event
   loop is dispatched on LangChain's executor thread; if the store is bound to
   the agent's loop the capture is skipped (fail-open, INV-6) rather than
   corrupting the session.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from sentinel.instrument.registry import BaseInstrumentor
from sentinel.instrument.session import SessionContext
from sentinel.models.events import (
    AGENT_STEP,
    ERROR,
    LLM_REQUEST,
    LLM_RESPONSE,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
    RefLink,
)

log = structlog.get_logger("sentinel.langchain")

#: Cap for any captured transcript fragment (string payloads are truncated at
#: this length and marked). Full streaming caps are a ``S2`` deliverable
#: (``S1-T16``); per-field capping keeps payloads bounded in the interim.
MAX_CAPTURED_TEXT = 65_536

#: The handler class, built lazily once ``langchain-core`` is importable.
_HANDLER_CLASS: type[Any] | None = None


class LangChainUnavailableError(ImportError):
    """Raised when :meth:`LangChainInstrumentor.handler` needs ``langchain-core``.

    Install it with ``pip install "sentinel-sdk[langchain]"``.
    """


@dataclass(frozen=True)
class _Run:
    """One in-flight LangChain run keyed by its ``run_id``."""

    event_id: str
    name: str
    started: float


def _clock() -> float:
    return time.perf_counter()


def _latency_ms(started: float | None) -> float:
    """Monotonic elapsed milliseconds since *started* (``0.0`` when missing)."""
    if started is None:
        return 0.0
    return round((_clock() - started) * 1000.0, 3)


def _capped(value: object) -> object:
    """Truncate transcript fragments to ``MAX_CAPTURED_TEXT``.

    Strings are cut and marked ``…[truncated]``; dicts and lists are capped
    recursively; any other object is stringified so payloads stay JSON-safe.
    """
    if isinstance(value, str):
        if len(value) <= MAX_CAPTURED_TEXT:
            return value
        return value[:MAX_CAPTURED_TEXT] + "\n…[truncated]"
    if isinstance(value, dict):
        return {str(key): _capped(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_capped(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _activity_name(serialized: dict[str, Any] | None) -> str:
    """Best-effort name for a runnable from LangChain's ``serialized`` dict."""
    snapshot = serialized if isinstance(serialized, dict) else {}
    name = snapshot.get("name")
    if isinstance(name, str) and name:
        return name
    id_path = snapshot.get("id")
    if isinstance(id_path, list) and id_path and isinstance(id_path[-1], str):
        return id_path[-1]
    lc_type = snapshot.get("lc")
    return f"langchain:{lc_type}" if lc_type is not None else "unknown"


def _handler_class() -> type[Any]:
    """Build (once) the LangChain ``BaseCallbackHandler`` subclass.

    The class is defined lazily because subclassing ``BaseCallbackHandler``
    requires ``langchain-core`` to be importable; the SDK core never imports it.
    """
    global _HANDLER_CLASS
    if _HANDLER_CLASS is None:
        try:
            from langchain_core.callbacks import BaseCallbackHandler
            from langchain_core.outputs import LLMResult
        except ImportError as exc:
            raise LangChainUnavailableError(
                "sentinel.instrument.langchain needs the optional 'langchain' "
                "extra: pip install 'sentinel-sdk[langchain]'"
            ) from exc

        class SentinelLangChainHandler(BaseCallbackHandler):
            """Emit Sentinel events from LangChain callbacks (``S1-T7``)."""

            #: Never let a capture failure propagate into the agent (INV-6).
            raise_error = False

            def __init__(self, ctx: SessionContext, *, is_active: Callable[[], bool]) -> None:
                """Capture into *ctx* while *is_active* returns ``True``."""
                self._ctx = ctx
                self._active = is_active
                # run_id -> in-flight run, for pairing and causal refs.
                self._llm: dict[str, _Run] = {}
                self._tool: dict[str, _Run] = {}
                self._chain: dict[str, _Run] = {}

            async def _capture(
                self,
                *,
                type: str,  # noqa: A002
                payload: Mapping[str, object],
                refs: Sequence[RefLink],
            ) -> Event | None:
                """Record an event, failing open (INV-6)."""
                if not self._active():
                    return None
                try:
                    return await self._ctx.capture(type=type, payload=payload, refs=refs)
                except Exception as exc:
                    log.warning("langchain.capture_failed", event_type=type, error=repr(exc))
                    return None

            def _refs(
                self, *, parent_run_id: UUID | None = None, caused_by: str | None = None
            ) -> list[RefLink]:
                """Links to an enclosing chain step and/or the causal request."""
                refs: list[RefLink] = []
                if caused_by is not None:
                    refs.append(RefLink(event_id=caused_by, kind=RefKind.CAUSED_BY))
                if parent_run_id is not None:
                    parent = self._chain.get(str(parent_run_id))
                    if parent is not None:
                        refs.append(RefLink(event_id=parent.event_id, kind=RefKind.PARENT))
                return refs

            async def on_chain_start(
                self,
                serialized: dict[str, Any],
                inputs: dict[str, Any],
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: object,
            ) -> None:
                """Open an ``agent.step``; later events inherit it as parent."""
                name = _activity_name(serialized)
                event = await self._capture(
                    type=AGENT_STEP,
                    payload={
                        "provider": "langchain",
                        "step": "chain",
                        "name": name,
                        "status": "started",
                        "inputs": _capped(inputs),
                    },
                    refs=self._refs(parent_run_id=parent_run_id),
                )
                if event is not None:
                    self._chain[str(run_id)] = _Run(event.event_id, name, _clock())

            async def on_chain_end(
                self,
                outputs: dict[str, Any],
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: object,
            ) -> None:
                """Close the ``agent.step`` opened by :meth:`on_chain_start`."""
                run = self._chain.pop(str(run_id), None)
                refs: list[RefLink] = []
                if run is not None:
                    refs.append(RefLink(event_id=run.event_id, kind=RefKind.PARENT))
                else:
                    refs = self._refs(parent_run_id=parent_run_id)
                await self._capture(
                    type=AGENT_STEP,
                    payload={
                        "provider": "langchain",
                        "step": "chain",
                        "name": run.name if run else "unknown",
                        "status": "ended",
                        "latency_ms": _latency_ms(run.started if run else None),
                        "outputs": _capped(outputs),
                    },
                    refs=refs,
                )

            async def on_chain_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: object,
            ) -> None:
                """Capture a failed chain step as an ``error`` event."""
                run = self._chain.pop(str(run_id), None)
                refs: list[RefLink] = []
                if run is not None:
                    refs.append(RefLink(event_id=run.event_id, kind=RefKind.PARENT))
                else:
                    refs = self._refs(parent_run_id=parent_run_id)
                await self._capture(
                    type=ERROR,
                    payload={
                        "provider": "langchain",
                        "scope": "chain",
                        "name": run.name if run else "unknown",
                        "message": _capped(str(error)),
                    },
                    refs=refs,
                )

            async def on_llm_start(
                self,
                serialized: dict[str, Any],
                prompts: list[str],
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: object,
            ) -> None:
                """Capture an ``llm.request`` for an LLM/chat-model call."""
                name = _activity_name(serialized)
                event = await self._capture(
                    type=LLM_REQUEST,
                    payload={
                        "provider": "langchain",
                        "model": name,
                        "prompts": _capped(prompts),
                    },
                    refs=self._refs(parent_run_id=parent_run_id),
                )
                if event is not None:
                    self._llm[str(run_id)] = _Run(event.event_id, name, _clock())

            async def on_llm_end(
                self,
                response: LLMResult,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: object,
            ) -> None:
                """Capture the ``llm.response`` linked to its request."""
                run = self._llm.pop(str(run_id), None)
                texts = [generation[0].text for generation in response.generations if generation]
                await self._capture(
                    type=LLM_RESPONSE,
                    payload={
                        "provider": "langchain",
                        "model": run.name if run else "unknown",
                        "latency_ms": _latency_ms(run.started if run else None),
                        "generations": _capped(texts),
                    },
                    refs=self._refs(
                        parent_run_id=parent_run_id, caused_by=run.event_id if run else None
                    ),
                )

            async def on_llm_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: object,
            ) -> None:
                """Capture a failed LLM call as an ``error`` event."""
                run = self._llm.pop(str(run_id), None)
                await self._capture(
                    type=ERROR,
                    payload={
                        "provider": "langchain",
                        "scope": "llm",
                        "model": run.name if run else "unknown",
                        "message": _capped(str(error)),
                    },
                    refs=self._refs(
                        parent_run_id=parent_run_id, caused_by=run.event_id if run else None
                    ),
                )

            async def on_tool_start(
                self,
                serialized: dict[str, Any],
                input_str: str,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                inputs: dict[str, Any] | None = None,
                **kwargs: object,
            ) -> None:
                """Capture a ``tool.call``."""
                name = str(_activity_name(serialized))
                event = await self._capture(
                    type=TOOL_CALL,
                    payload={
                        "provider": "langchain",
                        "tool": name,
                        "input": _capped(input_str),
                    },
                    refs=self._refs(parent_run_id=parent_run_id),
                )
                if event is not None:
                    self._tool[str(run_id)] = _Run(event.event_id, name, _clock())

            async def on_tool_end(
                self,
                output: object,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: object,
            ) -> None:
                """Capture the ``tool.result`` linked to its call."""
                run = self._tool.pop(str(run_id), None)
                await self._capture(
                    type=TOOL_RESULT,
                    payload={
                        "provider": "langchain",
                        "tool": run.name if run else "unknown",
                        "latency_ms": _latency_ms(run.started if run else None),
                        "output": _capped(output),
                    },
                    refs=self._refs(
                        parent_run_id=parent_run_id, caused_by=run.event_id if run else None
                    ),
                )

            async def on_tool_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: object,
            ) -> None:
                """Capture a failed tool invocation as an ``error`` event."""
                run = self._tool.pop(str(run_id), None)
                await self._capture(
                    type=ERROR,
                    payload={
                        "provider": "langchain",
                        "scope": "tool",
                        "tool": run.name if run else "unknown",
                        "message": _capped(str(error)),
                    },
                    refs=self._refs(
                        parent_run_id=parent_run_id, caused_by=run.event_id if run else None
                    ),
                )

        _HANDLER_CLASS = SentinelLangChainHandler
    return _HANDLER_CLASS


class LangChainInstrumentor(BaseInstrumentor):
    """Capture LangChain activity through its callback system (``S1-T7``).

    Attach :meth:`handler` to the LangChain ``callbacks`` list of a runnable::

        from sentinel import SQLiteEventStore, session
        from sentinel.instrument.langchain import LangChainInstrumentor

        store = SQLiteEventStore("agent.db")
        async with session(store) as ctx:
            instrumentor = LangChainInstrumentor(ctx)
            await my_chain.ainvoke(
                {"question": "..."},
                config={"callbacks": [instrumentor.handler()]},
            )

    LLM calls become ``llm.request``/``llm.response``, tools ``tool.call``/
    ``tool.result``, and chains ``agent.step``. When :meth:`disable` is called
    the handler stops emitting (``enable``/``disable`` are idempotent).
    """

    name = "langchain"
    event_types: frozenset[str] = frozenset(
        {LLM_REQUEST, LLM_RESPONSE, TOOL_CALL, TOOL_RESULT, AGENT_STEP, ERROR}
    )

    def __init__(self, ctx: SessionContext, *, enabled: bool = True) -> None:
        """Wrap *ctx*; start emitting unless ``enabled=False``."""
        self._ctx = ctx
        self._handler: type[Any] | None = None
        self.enabled = enabled

    def handler(self) -> Any:  # noqa: ANN401 -- the LangChain handler is type-agnostic; langchain-core stays optional
        """Return the live LangChain callback handler, building it once.

        Requires ``langchain-core``; raises :class:`LangChainUnavailableError`
        when the ``langchain`` extra is not installed.
        """
        if self._handler is None:
            self._handler = _handler_class()(self._ctx, is_active=lambda: self.enabled)
        return self._handler
