"""Capture LangGraph node execution as ``agent.step`` events (``S1-T8``).

LangGraph compiles onto LangChain's callback system, so Sentinel plugs the
same callback-handler approach in: attach :meth:`LangGraphInstrumentor.handler`
to the graph invocation (``config={"callbacks": [handler]}``) and capture runs
alongside normal agent execution.

LangGraph decorates every node run's callback ``metadata`` with
``langgraph_node``, ``langgraph_step`` and the node's checkpoint namespace.
This handler detects those runs and records each node entry/exit as an
``agent.step`` event:

* ``payload["step"] == "step"`` (vs ``"chain"`` for the enclosing graph run).
* ``payload["node"]`` is the node name; ``payload["step_index"]`` is the
  per-run step number.
* ``payload["input"]``/``payload["output"]`` hold the bounded state deltas at
  node entry/exit.
* ``payload["status"]`` is ``started`` or ``ended``; ``end`` links back to
  ``start`` of the same node via a ``parent`` reference.

The graph's own run (no ``langgraph_node`` key) falls through to the
:mod:`sentinel.instrument.langchain` handler, becoming the enclosing
``agent.step`` (``step == "chain"``) that every node inherits as its parent.
LLM calls and tool invocations inside a node are emitted by the inherited
handler too and inherit the node step as their ``parent`` — so the sequence of
events for one graph run is a navigable execution tree:

``graph -> nodes -> llm/tool events``

Only ``langchain-core`` is required at runtime (node identity arrives through
its callback metadata); the separate ``langgraph`` extra is only needed by the
user's agent. Capture is best-effort (INV-6) and never blocks the host.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

import structlog

from sentinel.instrument.langchain import (
    _capped,
    _clock,
    _handler_class,
    _latency_ms,
    _Run,
)
from sentinel.instrument.registry import BaseInstrumentor
from sentinel.instrument.session import SessionContext
from sentinel.models.events import (
    AGENT_STEP,
    ERROR,
    LLM_REQUEST,
    LLM_RESPONSE,
    TOOL_CALL,
    TOOL_RESULT,
    RefKind,
    RefLink,
)

log = structlog.get_logger("sentinel.langgraph")

#: The node-tracking callback handler, built lazily once langchain-core is
#: available (LangGraph itself is not imported here).
_HANDLER_CLASS: type[Any] | None = None


def _graph_node(metadata: Mapping[str, object] | None) -> str | None:
    """Return the LangGraph node name for *metadata*, if this is a node run."""
    snapshot = metadata if isinstance(metadata, Mapping) else {}
    node = snapshot.get("langgraph_node")
    return str(node) if isinstance(node, str) and node else None


def _graph_int(metadata: Mapping[str, object] | None, key: str) -> int | None:
    """Extract an int metadata field (LangGraph step counters) or ``None``."""
    snapshot = metadata if isinstance(metadata, Mapping) else {}
    value = snapshot.get(key)
    return value if isinstance(value, int) else None


def _handler_class_langgraph() -> type[Any]:
    """Build (once) the node-tracking ``agent.step`` handler.

    Subclasses the S1-T7 LangChain handler so LLM/tool capture and parent-link
    bookkeeping are reused; only the chain boundary is specialized to detect
    LangGraph node runs.
    """
    global _HANDLER_CLASS
    if _HANDLER_CLASS is None:
        base = _handler_class()

        class SentinelLangGraphHandler(base):  # type: ignore[valid-type, misc]
            """Emit ``agent.step`` events for LangGraph node runs (``S1-T8``)."""

            raise_error = False

            def __init__(
                self,
                ctx: SessionContext,
                *,
                is_active: Callable[[], bool],
            ) -> None:
                """Track in-flight node runs on top of the inherited handler."""
                super().__init__(ctx, is_active=is_active)
                # run_id -> per-run step index, for node end events.
                self._node_steps: dict[str, int | None] = {}

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
                """Record a node entry as ``agent.step``; else inherit chain capture."""
                node = _graph_node(metadata)
                if node is None:
                    await super().on_chain_start(
                        serialized,
                        inputs,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        tags=tags,
                        metadata=metadata,
                        **kwargs,
                    )
                    return
                event = await self._capture(
                    type=AGENT_STEP,
                    payload={
                        "provider": "langgraph",
                        "step": "step",
                        "node": node,
                        "status": "started",
                        "step_index": _graph_int(metadata, "langgraph_step"),
                        "input": _capped(inputs),
                    },
                    refs=self._refs(parent_run_id=parent_run_id),
                )
                if event is not None:
                    self._chain[str(run_id)] = _Run(event.event_id, node, _clock())
                    self._node_steps[str(run_id)] = _graph_int(metadata, "langgraph_step")

            async def on_chain_end(
                self,
                outputs: dict[str, Any],
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: object,
            ) -> None:
                """Close the node step; else inherit chain capture."""
                if str(run_id) not in self._node_steps:
                    await super().on_chain_end(
                        outputs,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        **kwargs,
                    )
                    return
                run = self._chain.pop(str(run_id), None)
                step_index = self._node_steps.pop(str(run_id), None)
                refs: list[RefLink] = []
                if run is not None:
                    refs.append(RefLink(event_id=run.event_id, kind=RefKind.PARENT))
                else:
                    refs = self._refs(parent_run_id=parent_run_id)
                await self._capture(
                    type=AGENT_STEP,
                    payload={
                        "provider": "langgraph",
                        "step": "step",
                        "node": run.name if run else "unknown",
                        "status": "ended",
                        "step_index": step_index,
                        "latency_ms": _latency_ms(run.started if run else None),
                        "output": _capped(outputs),
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
                """Capture a failed node run; else inherit chain error handling."""
                if str(run_id) not in self._node_steps:
                    await super().on_chain_error(
                        error,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        **kwargs,
                    )
                    return
                run = self._chain.pop(str(run_id), None)
                step_index = self._node_steps.pop(str(run_id), None)
                refs: list[RefLink] = []
                if run is not None:
                    refs.append(RefLink(event_id=run.event_id, kind=RefKind.PARENT))
                else:
                    refs = self._refs(parent_run_id=parent_run_id)
                await self._capture(
                    type=ERROR,
                    payload={
                        "provider": "langgraph",
                        "scope": "node",
                        "node": run.name if run else "unknown",
                        "step_index": step_index,
                        "message": _capped(str(error)),
                    },
                    refs=refs,
                )

        _HANDLER_CLASS = SentinelLangGraphHandler
    return _HANDLER_CLASS


class LangGraphInstrumentor(BaseInstrumentor):
    """Capture LangGraph node steps through its callback system (``S1-T8``).

    Attach :meth:`handler` to the graph's callbacks::

        from sentinel import SQLiteEventStore, session
        from sentinel.instrument.langgraph import LangGraphInstrumentor

        store = SQLiteEventStore("agent.db")
        async with session(store) as ctx:
            instrumentor = LangGraphInstrumentor(ctx)
            await app.ainvoke(
                {"question": "..."},
                config={"callbacks": [instrumentor.handler()]},
            )

    Every node boundary is recorded as an ``agent.step`` with ``step: "step"``
    and bounded input/output state deltas; the graph's own run becomes the
    enclosing ``agent.step`` (``step: "chain"``) linked from each node, and LLM
    calls and tool invocations inside a node inherit the node step as their
    parent — a navigable execution tree (``S1-T6``). Requires the ``langchain``
    extra; ``langgraph`` itself is needed to run the agent, not to capture it.
    """

    name = "langgraph"
    event_types: frozenset[str] = frozenset(
        {LLM_REQUEST, LLM_RESPONSE, TOOL_CALL, TOOL_RESULT, AGENT_STEP, ERROR}
    )

    def __init__(self, ctx: SessionContext, *, enabled: bool = True) -> None:
        """Wrap *ctx*; start emitting unless ``enabled=False``."""
        self._ctx = ctx
        self._handler: type[Any] | None = None
        self.enabled = enabled

    def handler(self) -> Any:  # noqa: ANN401 -- the LangGraph handler is type-agnostic
        """Return the live handler, building it once (lazily imports).

        Raises :class:`LangChainUnavailableError` when the ``langchain`` extra
        is not installed.
        """
        if self._handler is None:
            self._handler = _handler_class_langgraph()(self._ctx, is_active=lambda: self.enabled)
        return self._handler
