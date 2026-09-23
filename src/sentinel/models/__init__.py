"""Event and flag schemas. Public surface: ``Event`` and its factory."""

from __future__ import annotations

from sentinel.models.events import (
    EVENT_TYPES,
    LLM_REQUEST,
    LLM_RESPONSE,
    SCHEMA_VERSION,
    SESSION_END,
    SESSION_START,
    Event,
    new_event_id,
)

__all__ = [
    "EVENT_TYPES",
    "LLM_REQUEST",
    "LLM_RESPONSE",
    "SCHEMA_VERSION",
    "SESSION_END",
    "SESSION_START",
    "Event",
    "new_event_id",
]
