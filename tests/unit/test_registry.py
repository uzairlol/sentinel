"""Unit tests for the instrumentor registry (``S1-T2``)."""

from __future__ import annotations

import pytest

from sentinel.instrument.registry import (
    BaseInstrumentor,
    InstrumentorNotFoundError,
    InstrumentorRegistry,
)
from sentinel.models.events import LLM_REQUEST


def _instrumentor(name: str = "raw") -> BaseInstrumentor:
    instrumentor = BaseInstrumentor()
    instrumentor.name = name
    instrumentor.event_types = frozenset({LLM_REQUEST})
    return instrumentor


def test_register_validate_event_types() -> None:
    registry = InstrumentorRegistry()
    registry.register(_instrumentor())
    assert "raw" in registry
    assert registry.event_types("raw") == frozenset({LLM_REQUEST})


def test_register_rejects_undeclared_event_type() -> None:
    registry = InstrumentorRegistry()
    instrumentor = _instrumentor()
    instrumentor.event_types = frozenset({"made.up.type"})
    with pytest.raises(ValueError, match="unknown event types"):
        registry.register(instrumentor)


def test_enable_and_disable_are_idempotent() -> None:
    registry = InstrumentorRegistry()
    registry.register(_instrumentor())
    registry.enable("raw")
    registry.enable("raw")
    assert registry.is_enabled("raw") is True
    registry.disable("raw")
    registry.disable("raw")
    assert registry.is_enabled("raw") is False


def test_enable_all_then_disable_all() -> None:
    registry = InstrumentorRegistry()
    instrumentor = _instrumentor("langchain")
    registry.register(_instrumentor("raw"))
    registry.register(instrumentor)
    registry.enable_all()
    assert all(registry.is_enabled(name) for name in registry.names())
    registry.disable_all()
    assert not any(registry.is_enabled(name) for name in registry.names())


def test_unknown_instrumentor_raises() -> None:
    registry = InstrumentorRegistry()
    with pytest.raises(InstrumentorNotFoundError):
        registry.enable("nope")
    with pytest.raises(InstrumentorNotFoundError):
        registry.is_enabled("nope")


def test_names_in_registration_order() -> None:
    registry = InstrumentorRegistry()
    registry.register(_instrumentor())
    registry.register(_instrumentor("langgraph"))
    assert registry.names() == ["raw", "langgraph"]
