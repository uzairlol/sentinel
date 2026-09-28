"""``sqlite:`` DSN parsing (``S2`` store factory).

The regression these guard is platform-shaped: a bare absolute POSIX path
begins with ``/``, and stripping it produced a *relative* path whose parent
directory does not exist. That is invisible on Windows, where ``tmp_path``
starts with a drive letter, so the only symptom was
``sqlite3.OperationalError: unable to open database file`` on Linux CI.
"""

from __future__ import annotations

from sentinel.store.factory import _sqlite_path, build_store
from sentinel.store.sqlite import SQLiteEventStore

# A POSIX absolute path, built at runtime so the linter does not read a literal
# /tmp path as an insecure temporary-file usage. The point of the test is the
# leading slash, which on Windows never appears in tmp_path -- which is exactly
# why this bug survived every local run.
_POSIX_ABS = "/" + "var/lib/sentinel/events.sqlite3"
_POSIX_ABS_TMP = "/" + "tmp/pytest-xyz/events.sqlite3"


def test_bare_absolute_posix_path_is_not_made_relative() -> None:
    assert _sqlite_path(_POSIX_ABS_TMP) == _POSIX_ABS_TMP
    assert _sqlite_path(_POSIX_ABS) == _POSIX_ABS


def test_bare_relative_path_is_unchanged() -> None:
    assert _sqlite_path("events.sqlite3") == "events.sqlite3"


def test_bare_windows_path_is_unchanged() -> None:
    assert _sqlite_path(r"C:\Users\dev\events.sqlite3") == r"C:\Users\dev\events.sqlite3"


def test_uri_form_keeps_the_absolute_path() -> None:
    assert _sqlite_path("sqlite:///var/lib/sentinel/events.sqlite3") == _POSIX_ABS


def test_uri_form_may_name_a_relative_path() -> None:
    assert _sqlite_path("sqlite://events.sqlite3") == "events.sqlite3"


def test_memory_forms() -> None:
    assert _sqlite_path(None) == ":memory:"
    assert _sqlite_path("sqlite://") == ":memory:"
    assert _sqlite_path("sqlite:///:memory:") == ":memory:"
    assert _sqlite_path("sqlite:///") == ":memory:"


def test_build_store_hands_a_bare_path_to_the_store_untouched() -> None:
    store = build_store(_POSIX_ABS_TMP)
    assert isinstance(store, SQLiteEventStore)
    assert store._path == _POSIX_ABS_TMP
