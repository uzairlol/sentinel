"""Sentinel — runtime safety instrumentation for autonomous LLM agents.

This is the public package surface. Everything exported here is stable and
semver-managed. Anything reachable only through private modules (named with a
leading underscore) may change without notice. See docs/adr/0009.

The frozen public API (``S1-T1``):

* ``sentinel.session()`` — open a capture session against an event store.
* ``sentinel.configure(...)`` / ``sentinel.get_config()`` — runtime settings.
* ``sentinel.instrument.*`` — capture instrumentation (ollama, registry, and
  the framework instrumentors shipped in ``S1``).
"""

from __future__ import annotations

from sentinel.config import configure, get_config
from sentinel.instrument.ollama import (
    DEFAULT_BASE_URL,
    OllamaChatError,
    instrument_ollama_call,
)
from sentinel.instrument.registry import (
    BaseInstrumentor,
    Instrumentor,
    InstrumentorNotFoundError,
    InstrumentorRegistry,
)
from sentinel.instrument.session import SessionClosedError, SessionContext, session
from sentinel.models.events import Event, new_event_id
from sentinel.query import CallGraph, Edge, get_call_graph
from sentinel.store.sqlite import SQLiteEventStore

__version__ = "0.0.2"

__all__ = [
    "DEFAULT_BASE_URL",
    "BaseInstrumentor",
    "CallGraph",
    "Edge",
    "Event",
    "Instrumentor",
    "InstrumentorNotFoundError",
    "InstrumentorRegistry",
    "OllamaChatError",
    "SQLiteEventStore",
    "SessionClosedError",
    "SessionContext",
    "__version__",
    "configure",
    "get_call_graph",
    "get_config",
    "instrument_ollama_call",
    "new_event_id",
    "session",
]
