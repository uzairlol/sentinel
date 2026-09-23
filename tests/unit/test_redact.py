"""Unit tests for the redaction hook (``S1-T14``)."""

from __future__ import annotations

from sentinel.config import DEFAULT_REDACTION_PATTERNS
from sentinel.redact import REDACTED, redact_payload


def test_redacts_bearer_token_in_messages() -> None:
    payload = {
        "url": "http://example.com",
        "request": {
            "messages": [
                {"role": "user", "content": "Bearer abcDEF0123_-ghi"},
                {"role": "assistant", "content": "fine"},
            ]
        },
    }
    redacted = redact_payload(payload, patterns=DEFAULT_REDACTION_PATTERNS)
    assert REDACTED in redacted["request"]["messages"][0]["content"]
    assert "abcDEF0123_-ghi" not in str(redacted)
    assert redacted["request"]["messages"][1] == {"role": "assistant", "content": "fine"}


def test_redacts_api_key_assignment_inside_strings() -> None:
    payload = {"env": "API_KEY=sk-ant-secret12345"}
    redacted = redact_payload(payload, patterns=DEFAULT_REDACTION_PATTERNS)
    assert "sk-ant-secret12345" not in redacted["env"]


def test_preserves_structure_and_non_string_values() -> None:
    payload = {"count": 3, "flag": True, "nothing": None, "nested": {"items": [1, "ok"]}}
    redacted = redact_payload(payload, patterns=DEFAULT_REDACTION_PATTERNS)
    assert redacted == payload


def test_custom_placeholder_and_patterns() -> None:
    payload = {"note": "my pin is 1234"}
    redacted = redact_payload(payload, patterns=[r"\d{4}"], placeholder="<hidden>")
    assert redacted["note"] == "my pin is <hidden>"


def test_redacts_openai_style_key() -> None:
    payload = {"api": "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJ"}
    redacted = redact_payload(payload, patterns=DEFAULT_REDACTION_PATTERNS)
    assert "sk-abcdefghijklmnopqrst" not in redacted["api"]


def test_default_policy_masks_authorization_headers() -> None:
    payload = {"headers": {"Authorization": "Bearer ghp_jklMNOpqr123456"}}
    redacted = redact_payload(payload, patterns=DEFAULT_REDACTION_PATTERNS)
    assert "ghp_jklMNOpqr123456" not in redacted["headers"]["Authorization"]
