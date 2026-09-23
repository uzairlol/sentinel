"""Instrumentor registry (``S1-T2``).

An *instrumentor* captures one boundary (an LLM call, a tool invocation, a
memory access) for its host framework. The registry keeps them discoverable,
idempotently enable-able, and honest about the event types they emit: every
declared event type must belong to the taxonomy (``S1-T4``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sentinel.models.events import EVENT_TYPES


class InstrumentorNotFoundError(KeyError):
    """Raised when a registry call names an unregistered instrumentor."""


@runtime_checkable
class Instrumentor(Protocol):
    """The contract every instrumentor implements."""

    name: str
    event_types: frozenset[str]
    enabled: bool

    def enable(self) -> None:
        """Start capturing. Must be safe to call more than once."""

    def disable(self) -> None:
        """Stop capturing. Must be safe to call more than once."""


class BaseInstrumentor:
    """A default instrumentor implementation: name, declared types, toggle."""

    name: str = "base"
    event_types: frozenset[str] = frozenset()
    enabled: bool = False

    def enable(self) -> None:
        """Mark the instrumentor enabled; idempotent."""
        self.enabled = True

    def disable(self) -> None:
        """Mark the instrumentor disabled; idempotent."""
        self.enabled = False


class InstrumentorRegistry:
    """Register and toggle instrumentors by name."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self._instrumentors: dict[str, Instrumentor] = {}

    def register(self, instrumentor: Instrumentor) -> None:
        """Register *instrumentor*, validating its declared event types."""
        undeclared = instrumentor.event_types - EVENT_TYPES
        if undeclared:
            raise ValueError(
                f"instrumentor {instrumentor.name!r} declares unknown event "
                f"types: {sorted(undeclared)}"
            )
        self._instrumentors[instrumentor.name] = instrumentor

    def names(self) -> list[str]:
        """Return instrumentor names in registration order."""
        return list(self._instrumentors)

    def enable(self, name: str) -> None:
        """Enable *name*, idempotently; unknown names raise."""
        self._require(name).enable()

    def disable(self, name: str) -> None:
        """Disable *name*, idempotently; unknown names raise."""
        self._require(name).disable()

    def enable_all(self) -> None:
        """Enable every registered instrumentor."""
        for instrumentor in self._instrumentors.values():
            instrumentor.enable()

    def disable_all(self) -> None:
        """Disable every registered instrumentor."""
        for instrumentor in self._instrumentors.values():
            instrumentor.disable()

    def is_enabled(self, name: str) -> bool:
        """Return whether *name* is currently enabled."""
        return self._require(name).enabled

    def event_types(self, name: str) -> frozenset[str]:
        """Return the taxonomy event types *name* emits."""
        return self._require(name).event_types

    def __contains__(self, name: str) -> bool:
        """Return whether *name* is registered."""
        return name in self._instrumentors

    def _require(self, name: str) -> Instrumentor:
        try:
            return self._instrumentors[name]
        except KeyError as exc:
            raise InstrumentorNotFoundError(name) from exc
