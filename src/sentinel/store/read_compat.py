"""Validation-on-read for older event schemas (``S2-T6``, docs/adr/0007).

Schema ``0.1`` (``S0``) stored ``refs`` as a flat list of parent event id
strings; schema ``0.2`` (``S1``) introduced typed :class:`RefLink` objects.
Persistence is append-only (INV-2), so a store may still be replaying ``0.1``
rows years from now. ``materialize_refs`` normalises either shape into the
current :class:`RefLink` list so the rest of the SDK never sees the old form.
Unknown shapes raise :class:`ValueError` instead of being silently coerced.
"""

from __future__ import annotations

from collections.abc import Iterable

from sentinel.models.events import RefKind, RefLink


def materialize_refs(raw: Iterable[object] | None) -> list[RefLink]:
    """Normalise a persisted refs list into current :class:`RefLink` objects.

    Accepts the ``0.2`` dict form (``{"event_id": ..., "kind": ...}``) and the
    ``0.1`` flat-string form (each entry a bare parent event id).
    """
    if raw is None:
        return []
    links: list[RefLink] = []
    for entry in raw:
        if isinstance(entry, str):
            links.append(RefLink(event_id=entry, kind=RefKind.PARENT))
        elif isinstance(entry, dict):
            kind = entry.get("kind", RefKind.PARENT.value)
            event_id = entry.get("event_id")
            if not isinstance(event_id, str):
                raise ValueError(f"malformed ref entry (missing string event_id): {entry!r}")
            links.append(RefLink(event_id=event_id, kind=RefKind(kind)))
        else:
            raise ValueError(f"malformed ref entry: {entry!r}")
    return links
