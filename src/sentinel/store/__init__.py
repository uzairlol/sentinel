"""Event store backends.

``SQLiteEventStore`` is the dev/test store; Postgres (``S2``) is the reference
production store.
"""

from __future__ import annotations

from sentinel.store.protocol import EventStore
from sentinel.store.sqlite import SQLiteEventStore

__all__ = ["EventStore", "SQLiteEventStore"]
