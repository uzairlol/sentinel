"""Redaction hook applied before persistence (``S1-T14``).

The batching writer runs ``redact_payload`` over every event before it reaches
the store. The default policy replaces secret-shaped spans in string values with
a placeholder; the pattern set comes from :class:`~sentinel.config
.SentinelSettings.redaction_patterns` and is configurable per deployment.

Deliberately conservative: it targets *values*, never structure — an event keeps
every key and nesting so replay stays lossless.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

REDACTED = "[REDACTED]"


def redact_payload(
    payload: Mapping[str, Any],
    *,
    patterns: Sequence[str],
    placeholder: str = REDACTED,
) -> dict[str, Any]:
    """Return *payload* with secret-shaped string values masked.

    ``patterns`` are regular expressions applied to every string value found by
    a deep walk of the payload (nested dicts and lists included). Keys and
    types are left untouched; non-string values pass through unchanged.
    """
    compiled = [re.compile(pattern) for pattern in patterns]
    result = _walk(payload, compiled=compiled, placeholder=placeholder)
    return cast(dict[str, Any], result)


def _walk(
    value: object,
    *,
    compiled: Sequence[re.Pattern[str]],
    placeholder: str,
) -> object:
    if isinstance(value, str):
        return _redact_string(value, compiled=compiled, placeholder=placeholder)
    if isinstance(value, Mapping):
        return {
            key: _walk(item, compiled=compiled, placeholder=placeholder)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_walk(item, compiled=compiled, placeholder=placeholder) for item in value]
    return value


def _redact_string(value: str, *, compiled: Sequence[re.Pattern[str]], placeholder: str) -> str:
    result = value
    for pattern in compiled:
        result = pattern.sub(placeholder, result)
    return result
