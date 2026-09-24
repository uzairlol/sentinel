"""Unit tests for runtime configuration (``S1-T3``)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.config import (
    DEFAULT_REDACTION_PATTERNS,
    SentinelSettings,
    configure,
    get_config,
)


def test_defaults_are_safe_for_unconfigured_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "SENTINEL_CAPTURE_ENABLED",
        "SENTINEL_FAIL_OPEN",
        "SENTINEL_REDACTION_ENABLED",
        "SENTINEL_STORE_DSN",
        "SENTINEL_BATCH_MAX_SIZE",
        "SENTINEL_QUEUE_MAX_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = SentinelSettings()
    assert settings.capture_enabled is True
    assert settings.fail_open is True
    assert settings.redaction_enabled is True
    assert settings.store_dsn.startswith("sqlite:///")
    assert settings.batch_max_size >= 1
    assert settings.queue_max_size >= settings.batch_max_size


def test_configure_applies_overrides() -> None:
    settings = configure(fail_open=False, batch_max_size=5)
    assert settings.fail_open is False
    assert settings.batch_max_size == 5
    assert get_config() is settings


def test_configure_without_args_resets_overrides() -> None:
    configure(fail_open=False)
    reset = configure()
    assert reset.fail_open is True


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentinelSettings(no_such_knob=1)  # type: ignore[call-arg]


def test_default_redaction_patterns_are_secret_shaped() -> None:
    assert len(DEFAULT_REDACTION_PATTERNS) >= 3
    assert any("authorization" in pattern for pattern in DEFAULT_REDACTION_PATTERNS)
