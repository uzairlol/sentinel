"""Store-side domain errors (``S1-T5``).

Referential integrity (INV-3) is enforced at append time: every
:class:`~sentinel.models.events.RefLink` must point at an event that already
exists in the same session. Violations raise :class:`RefIntegrityError`
instead of silently persisting a dangling link.
"""

from __future__ import annotations


class RefIntegrityError(ValueError):
    """Raised when an event references an event that is absent from its session."""
