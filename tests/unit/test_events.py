"""Unit tests for the S0 event envelope (``S0-T1``).

Envelope rules: ULID ``event_id``, ``seq >= 0``, timezone-aware UTC ``ts``,
ULID ``refs``, frozen ``schema_version``, and unknown fields rejected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from sentinel.models.events import SCHEMA_VERSION, Event, new_event_id


def _valid_event(**overrides: Any) -> Event:
    fields: dict[str, Any] = {
        "event_id": new_event_id(),
        "session_id": new_event_id(),
        "seq": 0,
        "ts": datetime.now(UTC),
        "type": "test.type",
    }
    fields.update(overrides)
    return Event(**fields)


def test_new_event_id_is_an_ulid() -> None:
    assert len(new_event_id()) == 26
    assert new_event_id() != new_event_id()


def test_envelope_defaults_schema_version_and_empty_payload() -> None:
    event = _valid_event()
    assert event.schema_version == SCHEMA_VERSION
    assert event.payload == {}
    assert event.refs == []


def test_envelope_stores_payload_and_refs() -> None:
    ref = new_event_id()
    event = _valid_event(payload={"model": "llama3.2"}, refs=[ref])
    assert event.payload == {"model": "llama3.2"}
    assert event.refs == [ref]


def test_envelope_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Event(  # type: ignore[call-arg]
            event_id=new_event_id(), session_id=new_event_id(), seq=0, type="x"
        )


def test_envelope_rejects_negative_seq() -> None:
    with pytest.raises(ValidationError):
        _valid_event(seq=-1)


def test_envelope_rejects_invalid_event_id() -> None:
    with pytest.raises(ValidationError):
        _valid_event(event_id="not-an-ulid")


def test_envelope_rejects_invalid_ref() -> None:
    with pytest.raises(ValidationError):
        _valid_event(refs=["junk"])


def test_envelope_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        _valid_event(ts=datetime(2026, 1, 1))


def test_envelope_rejects_non_utc_timestamp() -> None:
    shifted = datetime(2026, 1, 1, tzinfo=UTC).astimezone(timezone(timedelta(hours=5)))
    with pytest.raises(ValidationError):
        _valid_event(ts=shifted)


def test_envelope_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _valid_event(extra_field="boom")
