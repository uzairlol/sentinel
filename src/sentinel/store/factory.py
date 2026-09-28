"""Store construction from configuration (Sprint ``S2``).

A thin factory so CLI tools and deployment glue can build the right
:class:`~sentinel.store.protocol.EventStore` from a DSN without importing
backends directly. ``sqlite://`` selects the dev store; anything else is
treated as a Postgres DSN (``postgresql://``, ``postgresql+asyncpg://`` or
``postgres://`` -- see :func:`~sentinel.store.postgres._normalize_dsn`).
"""

from __future__ import annotations

from sentinel.config import get_config
from sentinel.store import sqlite
from sentinel.store.protocol import EventStore


def build_store(dsn: str | None = None) -> EventStore:
    """Construct a store for *dsn*, falling back to configured/embedded SQLite.

    ``dsn`` wins over configuration. With Postgres, the pool and retry knobs
    (``S2-T14``) come from the active settings so deployments tune them in one
    place.
    """
    settings = get_config()
    target = dsn or settings.store_dsn
    if (
        target is None
        or target == "sqlite://"
        or target.startswith("sqlite:")
        # a bare file path (no URI scheme) is always the local dev store
        or "://" not in target
    ):
        return sqlite.SQLiteEventStore(_sqlite_path(target))
    try:
        from sentinel.store.postgres import PostgresEventStore
    except ImportError as exc:  # pragma: no cover - exercised by integration CI
        raise ImportError(
            "Postgres store selected but the sentinel-sdk[postgres] extra is not installed"
        ) from exc
    return PostgresEventStore(
        target,
        pool_size=settings.store_pool_size,
        max_overflow=settings.store_max_overflow,
        pool_timeout=settings.store_pool_timeout_s,
        retry_attempts=settings.store_retry_attempts,
        retry_jitter_ms=settings.store_retry_jitter_ms,
    )


def _sqlite_path(dsn: str | None) -> str:
    """Extract a file path from a ``sqlite:`` DSN (or a bare path, as given).

    ``sqlite:///abs/path`` is the URI spelling of the absolute path
    ``/abs/path``: removing the scheme already leaves the path intact, and the
    *single* slash the URI form contributes is part of the path. Stripping
    slashes here instead turned a bare absolute POSIX path into a relative one
    whose parent directory does not exist -- which is why this only ever
    surfaced on Linux CI, where ``tmp_path`` starts with a slash.
    """
    if not dsn or dsn == "sqlite://" or dsn.endswith(":memory:"):
        return ":memory:"
    if not dsn.startswith("sqlite:"):
        # a bare file path: hand it to the store unchanged
        return dsn
    path = dsn.removeprefix("sqlite://")
    # `sqlite:///` and friends carry no path at all
    return path if path.strip("/") else ":memory:"
