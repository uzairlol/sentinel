"""Public instrumentation surface (INV-1: capture only, no evaluation)."""

from __future__ import annotations

from sentinel.instrument.ollama import (
    DEFAULT_BASE_URL,
    OllamaChatError,
    instrument_ollama_call,
)
from sentinel.instrument.session import (
    SessionClosedError,
    SessionContext,
    session,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "OllamaChatError",
    "SessionClosedError",
    "SessionContext",
    "instrument_ollama_call",
    "session",
]
