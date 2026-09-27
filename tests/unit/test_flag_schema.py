"""The universal ``Flag`` schema (``S3-T1``, docs/adr/0012).

Covers the three load-bearing properties of the flag schema: structural
evidence (INV-3), deterministic identity, and deterministic timestamps.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from sentinel.models.events import new_event_id
from sentinel.models.flags import (
    FLAG_SCHEMA_VERSION,
    MAX_EVIDENCE,
    MAX_SUMMARY_CHARS,
    Adjudication,
    EvidenceRef,
    EvidenceRole,
    Flag,
    Severity,
    evidence_digest,
    flag_identity,
    known_categories,
    normalize_utc,
    register_category,
)


def _evidence(role: EvidenceRole = EvidenceRole.CLAIM) -> list[EvidenceRef]:
    return [EvidenceRef(event_id=new_event_id(), role=role, seq=7)]


def _flag(**overrides: object) -> Flag:
    base: dict[str, object] = {
        "session_id": new_event_id(),
        "module": "provenance",
        "module_version": "0.1.0",
        "category": "ungrounded_claim",
        "confidence": 0.9,
        "summary": "claim cites a tool that was never called",
        "evidence": _evidence(),
        "created_at": datetime.now(UTC),
        "dedupe_key": "abc",
    }
    base.update(overrides)
    return Flag.create(**base)  # type: ignore[arg-type]


# -- construction -----------------------------------------------------------


def test_create_populates_deterministic_id_and_defaults() -> None:
    session_id = new_event_id()
    evidence = _evidence()
    created_at = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    flag = _flag(session_id=session_id, evidence=evidence, created_at=created_at)

    assert flag.flag_id == flag_identity(
        session_id=session_id,
        module="provenance",
        module_version="0.1.0",
        category="ungrounded_claim",
        dedupe_key="abc",
    )
    assert flag.severity is Severity.MEDIUM
    assert flag.adjudication is Adjudication.PENDING
    assert flag.review_only is False
    assert flag.auto_resolved is None
    assert flag.schema_version == FLAG_SCHEMA_VERSION
    assert flag.evidence_event_ids == [evidence[0].event_id]
    assert flag.has_role(EvidenceRole.CLAIM)
    assert not flag.has_role(EvidenceRole.EVIDENCE)


def test_identity_is_pure_function_of_its_inputs() -> None:
    session_id = new_event_id()
    first = _flag(session_id=session_id, dedupe_key="k1")
    again = _flag(session_id=session_id, dedupe_key="k1")
    other_key = _flag(session_id=session_id, dedupe_key="k2")
    other_version = _flag(session_id=session_id, dedupe_key="k1", module_version="0.2.0")
    other_category = _flag(session_id=session_id, dedupe_key="k1", category="contradicted_claim")
    other_session = _flag(session_id=new_event_id(), dedupe_key="k1")

    assert first.flag_id == again.flag_id
    for other in (other_key, other_version, other_category, other_session):
        assert other.flag_id != first.flag_id


def test_identity_is_a_valid_ulid() -> None:
    from ulid import ULID

    flag = _flag()
    assert ULID.from_str(flag.flag_id) == ULID.from_str(flag.flag_id)


def test_severity_rank_is_ordered() -> None:
    assert (
        Severity.INFO.rank
        < Severity.LOW.rank
        < Severity.MEDIUM.rank
        < Severity.HIGH.rank
        < Severity.CRITICAL.rank
    )


# -- validation -------------------------------------------------------------


@pytest.mark.parametrize("field", ["flag_id", "event_id"])
def test_non_ulid_ids_are_rejected(field: str) -> None:
    payload = _flag().model_dump()
    if field == "event_id":
        payload["evidence"] = [
            {"event_id": "not-a-ulid", "role": "claim", "seq": None, "note": None}
        ]
    else:
        payload["flag_id"] = "not-a-ulid"
    with pytest.raises(ValidationError, match=field):
        Flag(**payload)


def test_evidence_must_not_be_empty() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        Flag(**{**_flag().model_dump(), "evidence": []})


def test_evidence_is_capped() -> None:
    evidence = [EvidenceRef(event_id=new_event_id(), role=EvidenceRole.CONTEXT) for _ in range(40)]
    with pytest.raises(ValidationError):
        Flag(**{**_flag().model_dump(), "evidence": evidence})
    assert MAX_EVIDENCE >= 2


def test_duplicate_evidence_entry_is_rejected() -> None:
    ref = EvidenceRef(event_id=new_event_id(), role=EvidenceRole.CLAIM)
    with pytest.raises(ValidationError, match="duplicate evidence"):
        Flag(**{**_flag().model_dump(), "evidence": [ref, ref.model_copy()]})


def test_same_event_may_play_two_roles() -> None:
    ref = EvidenceRef(event_id=new_event_id(), role=EvidenceRole.CLAIM)
    counter = ref.model_copy(update={"role": EvidenceRole.COUNTERVAILANCE})
    flag = Flag(**{**_flag().model_dump(), "evidence": [ref, counter]})
    assert len(flag.evidence) == 2


def test_blank_summary_is_rejected() -> None:
    with pytest.raises(ValidationError, match="summary"):
        Flag(**{**_flag().model_dump(), "summary": "   "})


def test_summary_length_is_capped() -> None:
    with pytest.raises(ValidationError):
        Flag(**{**_flag().model_dump(), "summary": "x" * (MAX_SUMMARY_CHARS + 1)})


def test_confidence_must_be_a_probability() -> None:
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError, match="confidence"):
            Flag(**{**_flag().model_dump(), "confidence": bad})


def test_naive_created_at_is_rejected() -> None:
    with pytest.raises(ValidationError, match="created_at"):
        Flag(**{**_flag().model_dump(), "created_at": datetime(2026, 9, 27, 12, 0)})


def test_non_utc_created_at_is_rejected() -> None:
    offset = timezone(timedelta(hours=5))
    with pytest.raises(ValidationError, match="created_at"):
        Flag(**{**_flag().model_dump(), "created_at": datetime(2026, 9, 27, 12, 0, tzinfo=offset)})


def test_category_must_be_a_slug() -> None:
    for bad in ("Ungrounded", "with space", "", "1leading_digit"):
        with pytest.raises(ValidationError, match="category"):
            Flag(**{**_flag().model_dump(), "category": bad})


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Flag(**{**_flag().model_dump(), "nope": 1})


def test_adjudication_enum_is_exhaustive() -> None:
    assert {member.value for member in Adjudication} == {"pending", "confirmed", "rejected"}
    assert Flag(**{**_flag().model_dump(), "adjudication": "confirmed"}).adjudication is (
        Adjudication.CONFIRMED
    )


# -- helpers ----------------------------------------------------------------


def test_evidence_rows_round_trip() -> None:
    ref = EvidenceRef(event_id=new_event_id(), role=EvidenceRole.COUNTERVAILANCE, seq=3, note="n")
    assert EvidenceRef.from_row(ref.to_row()) == ref


def test_evidence_digest_is_stable_and_role_sensitive() -> None:
    event_id = new_event_id()
    first = EvidenceRef(event_id=event_id, role=EvidenceRole.CLAIM, seq=1)
    same = EvidenceRef(event_id=event_id, role=EvidenceRole.CLAIM, seq=1)
    other_role = EvidenceRef(event_id=event_id, role=EvidenceRole.EVIDENCE, seq=1)

    assert evidence_digest([first]) == evidence_digest([same])
    assert evidence_digest([first]) != evidence_digest([other_role])
    assert len(evidence_digest([first])) == 16


def test_category_registry_is_discoverable() -> None:
    register_category("unit_test_category")
    assert "unit_test_category" in known_categories()
    with pytest.raises(ValueError, match="lower_snake_case"):
        register_category("Not A Slug")


def test_normalize_utc_handles_naive_and_offset() -> None:
    naive = datetime(2026, 9, 27, 12, 0)
    assert normalize_utc(naive) == datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    shifted = datetime(2026, 9, 27, 12, 0, tzinfo=timezone(timedelta(hours=5)))
    assert normalize_utc(shifted) == datetime(2026, 9, 27, 7, 0, tzinfo=UTC)
    assert normalize_utc(shifted).utcoffset() == timedelta(0)


def test_for_review_follows_review_only() -> None:
    assert _flag().for_review() is False
    assert _flag(review_only=True).for_review() is True
