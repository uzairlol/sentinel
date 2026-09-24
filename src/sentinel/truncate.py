"""Streaming-safe serialization caps (``S1-T16``, deferred from ``S1``).

A hostile or simply verbose host can hand capture payloads of arbitrary size
-- a tool that returns an entire file, or an LLM that echoes back a huge
prompt. Persisting that blob verbatim banks it forever (events are append-only,
INV-2), so the pipeline caps per-event size *before* the store sees it.

Truncation is lossless in provenance sense: the event is marked ``_truncated``
and carries a SHA-256 over the *original* serialized payload, so a reader can
(a) know data was cut, (b) tell *which* original produced this event, and
(c) distinguish marker-injection from genuine truncation. Structure is
preserved: keys and nesting survive; only oversized string *values* are cut.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

#: Alias so the JSON-walk helpers can stay dynamic yet ANN401-clean.
_Any = Any

#: Marker keys added to a truncated payload (never valid capture keys).
TRUNCATED = "_truncated"
TRUNCATED_HASH = "_truncated_hash"
TRUNCATED_COUNT = "_truncated_count"
TRUNCATED_FIELDS = "_truncated_fields"

#: Suffix appended to a cut string value, keeping the original length visible.
_CUT_MARK = "…[truncated %s chars]"


def payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic UTF-8 encoding of a payload (sort_keys, separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def truncate_payload(
    payload: Mapping[str, Any],
    *,
    max_bytes: int = 65_536,
) -> dict[str, Any]:
    """Return *payload*, truncated to fit within ``max_bytes``.

    When the serialized payload already fits, the mapping is returned
    unchanged (no marker keys added). Otherwise oversized string values are
    repeatedly halved until the result fits, then the marker keys
    ``_truncated``, ``_truncated_hash``, ``_truncated_count`` and
    ``_truncated_fields`` are added.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    original_bytes = payload_bytes(payload)
    if len(original_bytes) <= max_bytes:
        return dict(payload)

    marker_budget = len(
        payload_bytes(
            {TRUNCATED: True, TRUNCATED_HASH: "0" * 64, TRUNCATED_COUNT: 1, TRUNCATED_FIELDS: []}
        )
    )
    work = json.loads(original_bytes.decode("utf-8"))
    paths = _string_paths(work)
    truncated: list[str] = []

    while len(payload_bytes(work)) > max_bytes - marker_budget:
        # Candidates are strings long enough to still halve.
        candidates = [(path, len(_get(work, path))) for path in paths if len(_get(work, path)) > 24]
        if not candidates:
            break
        path, length = max(candidates, key=lambda c: c[1])
        keep = length // 2
        cut = _CUT_MARK % length
        kept_part = _get(work, path)[: max(keep - len(cut), 8)]
        _set(work, path, kept_part + cut)
        where = ".".join(str(p) for p in path)
        if where not in truncated:
            truncated.append(where)

    result: dict[str, Any] = json.loads(payload_bytes(work).decode("utf-8"))
    result[TRUNCATED] = True
    result[TRUNCATED_HASH] = hashlib.sha256(original_bytes).hexdigest()
    result[TRUNCATED_COUNT] = len(truncated)
    result[TRUNCATED_FIELDS] = sorted(truncated)
    return result


def is_truncated(payload: Mapping[str, Any]) -> bool:
    """Whether *payload* carries the truncation marker this module adds."""
    return bool(payload.get(TRUNCATED)) and TRUNCATED_HASH in payload


def _string_paths(value: object, prefix: Sequence[str | int] = ()) -> list[tuple[str | int, ...]]:
    paths: list[tuple[str | int, ...]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            sub = (*prefix, key)
            if isinstance(item, str):
                paths.append(sub)
            else:
                paths.extend(_string_paths(item, sub))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            sub = (*prefix, index)
            if isinstance(item, str):
                paths.append(sub)
            else:
                paths.extend(_string_paths(item, sub))
    return paths


def _get(container: _Any, path: Sequence[str | int]) -> _Any:
    value: _Any = container
    for step in path:
        value = value[step]
    return value


def _set(container: _Any, path: Sequence[str | int], value: _Any) -> None:
    node = container
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
