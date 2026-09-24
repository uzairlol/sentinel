"""Unit tests for payload truncation (``S1-T16``)."""

from __future__ import annotations

import hashlib

import pytest

from sentinel.truncate import (
    TRUNCATED,
    TRUNCATED_COUNT,
    TRUNCATED_FIELDS,
    TRUNCATED_HASH,
    is_truncated,
    payload_bytes,
    truncate_payload,
)


def test_under_limit_passes_through_unmodified() -> None:
    payload = {"prompt": "hi", "meta": {"n": 1}}
    assert truncate_payload(payload, max_bytes=10_000) == payload
    assert not is_truncated(truncate_payload(payload, max_bytes=10_000))


def test_oversized_payload_is_marked_and_digested() -> None:
    payload = {"message": "z" * 50_000}
    result = truncate_payload(payload, max_bytes=1_000)
    assert result[TRUNCATED] is True
    assert is_truncated(result)
    assert len(result[TRUNCATED_HASH]) == 64
    assert result[TRUNCATED_HASH] == hashlib.sha256(payload_bytes(payload)).hexdigest()
    assert result[TRUNCATED_COUNT] >= 1
    assert result[TRUNCATED_FIELDS] == ["message"]


def test_result_fits_within_budget_including_markers() -> None:
    payload = {"message": "z" * 50_000, "nested": {"a": ["w" * 10_000]}}
    result = truncate_payload(payload, max_bytes=1_000)
    assert len(payload_bytes(result)) <= 2_000
    assert "…[truncated" in result["message"]


def test_structure_and_non_string_values_preserved() -> None:
    payload = {"message": "z" * 50_000, "count": 7, "ok": True, "items": [1, 2, 3]}
    result = truncate_payload(payload, max_bytes=1_000)
    assert result["count"] == 7
    assert result["ok"] is True
    assert result["items"] == [1, 2, 3]
    assert set(result.keys()) == {
        "count",
        "items",
        "message",
        "ok",
        TRUNCATED,
        TRUNCATED_COUNT,
        TRUNCATED_FIELDS,
        TRUNCATED_HASH,
    }


def test_deterministic_cut_for_same_payload() -> None:
    payload = {"message": "z" * 50_000}
    first = truncate_payload(payload, max_bytes=1_000)
    second = truncate_payload(payload, max_bytes=1_000)
    assert payload_bytes(first) == payload_bytes(second)


def test_max_bytes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        truncate_payload({"a": "b"}, max_bytes=0)
