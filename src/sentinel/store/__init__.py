"""Event store backends.

``SQLiteEventStore`` is the dev/test store; ``PostgresEventStore`` is the
reference production store (docs/adr/0002, Sprint ``S2``). Both backend
implementations pass the same parity contract suite.
"""

from __future__ import annotations

from sentinel.store.errors import RefIntegrityError
from sentinel.store.gaps import SeqGap, assert_no_gaps, seq_gaps
from sentinel.store.postgres import PostgresEventStore, PostgresUnavailableError
from sentinel.store.protocol import EventStore
from sentinel.store.read_compat import materialize_refs
from sentinel.store.reporting import CallEdge, SessionSummary, StoreHealth
from sentinel.store.retention import PruneReport, RetentionPolicy, RetentionRule
from sentinel.store.sqlite import SQLiteEventStore

__all__ = [
    "CallEdge",
    "EventStore",
    "PostgresEventStore",
    "PostgresUnavailableError",
    "PruneReport",
    "RefIntegrityError",
    "RetentionPolicy",
    "RetentionRule",
    "SQLiteEventStore",
    "SeqGap",
    "SessionSummary",
    "StoreHealth",
    "assert_no_gaps",
    "materialize_refs",
    "seq_gaps",
]
