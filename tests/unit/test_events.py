"""Unit tests for the S0/S1 event envelope (``S0-T1``, ``S1-T4``, ``S1-T5``).

Envelope rules: ULID ``event_id``, ``seq >= 0``, timezone-aware UTC ``ts``,
``type`` constrained to the taxone taxonomy, typed ``RefLink`` refs, frozen
``schema_version``, and unknown fields rejected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from sentinel.models.events import (
    EVENT_TYPES,
    LLM_REQUEST,
    SCHEMA_VERSION,
    Event,
    RefKind,
    RefLink,
    new_event_id,
)


def _valid_event(**overrides: Any) -> Event:
    fields: dict[str, Any] = {
        "event_id": new_event_id(),
        "session_id": new_event_id(),
        "seq": 0,
        "ts": datetime.now(UTC),
        "type": LLM_REQUEST,
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


def test_envelope_stores_payload_and_typed_refs() -> None:
    ref = new_event_id()
    link = RefLink(event_id=ref, kind=RefKind.GROUNDS)
    event = _valid_event(payload={"model": "llama3.2"}, refs=[link])
    assert event.payload == {"model": "llama3.2"}
    assert event.refs == [link]


def test_ref_link_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        RefLink(event_id=new_event_id(), kind="because")  # type: ignore[arg-type]


def test_ref_link_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RefLink(event_id=new_event_id(), kind=RefKind.PARENT, why="extra")  # type: ignore[call-arg]


def test_event_rejects_unknown_event_type() -> None:
    with pytest.raises(ValidationError):
        _valid_event(type="totally.made.up")


def test_event_accepts_every_taxonomy_type() -> None:
    for event_type in EVENT_TYPES:
        assert _valid_event(type=event_type).type == event_type


def test_event_rejects_ref_to_itself() -> None:
    event = _valid_event()
    fields = event.model_dump(mode="json")
    fields.pop("refs")
    with pytest.raises(ValidationError):
        Event(**fields, refs=[RefLink(event_id=event.event_id, kind=RefKind.PARENT)])


def test_envelope_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Event(  # type: ignore[call-arg]
            event_id=new_event_id(), session_id=new_event_id(), seq=0, type=LLM_REQUEST
        )


def test_envelope_rejects_negative_seq() -> None:
    with pytest.raises(ValidationError):
        _valid_event(seq=-1)


def test_envelope_rejects_invalid_event_id() -> None:
    with pytest.raises(ValidationError):
        _valid_event(event_id="not-an-ulid")


def test_envelope_rejects_invalid_ref_event_id() -> None:
    with pytest.raises(ValidationError):
        RefLink(event_id="junk", kind=RefKind.PARENT)


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
