"""Sentinel runtime configuration (``S1-T3``).

``configure()`` builds a :class:`SentinelSettings` from environment variables
(``SENTINEL_*``) plus explicit overrides and exposes it process-wide. Capture
pipeline knobs (batching, queue bounds, fail-open) are read by the batching
writer (``S1-T12``), the redaction policy is applied before persistence
(``S1-T14``), and host-agent protection defaults to fail-open (INV-6).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default DSN; SQLite keeps single-process dev and tests dependency-free.
DEFAULT_STORE_DSN = "sqlite:///:memory:"

#: Common secret-shaped markers used by the default redaction policy.
DEFAULT_REDACTION_PATTERNS = (
    r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[=:]\s*[^\s,;}\]]+",
    r"(?i)bearer\s+[a-z0-9._-]{12,}",
    r"(?i)(sk|ghp|gho|AKIA)[a-z0-9_-]{6,}",
)


class SentinelSettings(BaseSettings):
    """Settings controlling capture behaviour across the SDK.

    Environment variables use the ``SENTINEL_`` prefix; nested keys are split on
    ``__``. Every field has a safe default so an uninstrumented process works
    without any configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    store_dsn: str = DEFAULT_STORE_DSN
    capture_enabled: bool = True

    fail_open: bool = True

    redaction_enabled: bool = True
    redaction_patterns: tuple[str, ...] = DEFAULT_REDACTION_PATTERNS

    batch_max_size: int = Field(default=100, ge=1)
    batch_flush_interval_ms: int = Field(default=250, ge=1)
    queue_max_size: int = Field(default=10_000, ge=1)

    #: Planned sampling support (``S1-T15``, deferred to ``S2``); accepted in
    #: config so deployments can set it before the behaviour lands.
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


_settings: SentinelSettings | None = None


def configure(**overrides: object) -> SentinelSettings:
    """Resolve and install the process-wide Sentinel settings.

    Values come from ``SENTINEL_*`` environment variables first; ``overrides``
    take precedence. Calling ``configure()`` with no arguments re-reads the
    environment and resets any earlier overrides.
    """
    global _settings
    # pydantic coerces the arbitrary knob kwargs, so the spread is unchecked.
    settings = (
        SentinelSettings(**overrides)  # type: ignore[arg-type]
        if overrides
        else SentinelSettings()
    )
    _settings = settings
    return settings


def get_config() -> SentinelSettings:
    """Return the installed settings, configuring defaults on first use."""
    global _settings
    if _settings is None:
        _settings = SentinelSettings()
    if _settings is None:  # pragma: no cover - defensive; _settings always set above
        raise RuntimeError("Sentinel settings failed to initialize")
    return _settings
