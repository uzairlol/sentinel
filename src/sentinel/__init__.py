"""Sentinel — runtime safety instrumentation for autonomous LLM agents.

This is the public package surface. Everything exported here is stable and
semver-managed. Anything reachable only through private modules (named with a
leading underscore) may change without notice. See docs/adr/0009.
"""

from __future__ import annotations

from sentinel.instrument.ollama import (
    DEFAULT_BASE_URL,
    OllamaChatError,
    instrument_ollama_call,
)
from sentinel.instrument.session import SessionClosedError, SessionContext, session
from sentinel.models.events import Event, new_event_id
from sentinel.store.sqlite import SQLiteEventStore

__version__ = "0.0.2"

__all__ = [
    "DEFAULT_BASE_URL",
    "Event",
    "OllamaChatError",
    "SQLiteEventStore",
    "SessionClosedError",
    "SessionContext",
    "__version__",
    "instrument_ollama_call",
    "new_event_id",
    "session",
]
