"""Sentinel command-line interface.

Private module. The `sentinel` console script is installed from here, but this
module is not part of the public API surface (docs/adr/0009).

Sprint ``S0`` adds ``sentinel replay <session_id>``; Sprint ``S2`` (``S2-T10``)
adds ``--json``/``--pretty`` output, cross-store ``--dsn`` selection, the
``sessions list`` query surface (``S2-T9``) and ``store health`` (``S2-T11``).
Sprint ``S3`` adds ``sentinel eval-fixtures --module provenance`` (``S3-T13``),
which runs a module over its adversarial corpus and prints the confusion matrix
with the gate verdict. Exit code 1 means the gate failed, so CI can just read it.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import structlog

from sentinel import __version__
from sentinel.store.factory import build_store
from sentinel.store.protocol import EventStore
from sentinel.store.sqlite import SQLiteEventStore

#: CLI default: a persistent dev file, not the config sample's ``:memory:``.
_DEFAULT_SQLITE_PATH = "sentinel.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Runtime safety instrumentation for autonomous LLM agents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sentinel-sdk {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    replay = subparsers.add_parser("replay", help="print a captured session from the event store")
    replay.add_argument("session_id", help="the session ID to replay")
    _add_store_arg(replay)
    _add_output_arg(replay)
    replay.set_defaults(func=_cmd_replay)

    sessions = subparsers.add_parser("sessions", help="query captured sessions")
    sessions_sub = sessions.add_subparsers(dest="sub", required=True)
    listing = sessions_sub.add_parser("list", help="list sessions (newest first)")
    listing.add_argument("--agent", help="only sessions for this agent_id")
    listing.add_argument("--since", help="only sessions started at/after ISO timestamp")
    listing.add_argument("--until", help="only sessions started at/before ISO timestamp")
    listing.add_argument("--flagged", action="store_true", help="only sessions with flags")
    listing.add_argument("--limit", type=int, default=100, help="max rows (default: 100)")
    listing.add_argument("--offset", type=int, default=0, help="first row to return")
    _add_store_arg(listing)
    _add_output_arg(listing)
    listing.set_defaults(func=_cmd_sessions_list)

    health = subparsers.add_parser("health", help="report store health (S2-T11)")
    _add_store_arg(health)
    _add_output_arg(health)
    health.set_defaults(func=_cmd_health)

    fixtures = subparsers.add_parser(
        "eval-fixtures",
        help="run an evaluation module over its adversarial corpus (S3-T13)",
    )
    fixtures.add_argument(
        "--module",
        required=True,
        help="module short name: provenance (S3), memory (S4), faithfulness (S5), spec (S6)",
    )
    _add_output_arg(fixtures)
    fixtures.set_defaults(func=_cmd_eval_fixtures)

    return parser


def _add_store_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dsn",
        help=(
            "store DSN (postgresql://..., sqlite:///...); defaults to the dev "
            f"SQLite file `{_DEFAULT_SQLITE_PATH}`"
        ),
    )


def _add_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit JSON (one doc per entity)")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="JSON pretty-printed with two-space indent (implies --json)",
    )


async def _open_store(dsn: str | None) -> EventStore:
    if dsn is None:
        return SQLiteEventStore(_DEFAULT_SQLITE_PATH)
    return build_store(dsn)


def _emit_json(args: argparse.Namespace, value: object) -> None:
    print(
        json.dumps(_jsonable(value), indent=2 if args.pretty else None, default=str, sort_keys=True)
    )


def _jsonable(value: object) -> object:
    """Walk dataclasses (result types) and containers to JSON-able objects."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


async def _run_replay(args: argparse.Namespace) -> int:
    store = await _open_store(args.dsn)
    try:
        events = await store.get_session(args.session_id)
    finally:
        await store.close()
    if not events:
        if args.json or args.pretty:
            _emit_json(args, {"error": f"session {args.session_id} not found"})
        else:
            print(f"No events found for session {args.session_id}", file=sys.stderr)
        return 1
    if args.json or args.pretty:
        _emit_json(args, [e.model_dump(mode="json") for e in events])
        return 0
    print(f"Session {args.session_id} - {len(events)} events")
    for event in events:
        summary = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
        if len(summary) > 120:
            summary = summary[:117] + "..."
        print(
            f"  #{event.seq} {event.ts.isoformat()} {event.type} refs={len(event.refs)} {summary}"
        )
    return 0


def _cmd_sessions_list(args: argparse.Namespace) -> int:
    return int(asyncio.run(_run_sessions_list(args)))


def _cmd_health(args: argparse.Namespace) -> int:
    return int(asyncio.run(_run_health(args)))


def _cmd_replay(args: argparse.Namespace) -> int:
    return int(asyncio.run(_run_replay(args)))


@contextmanager
def _quiet_logs() -> Iterator[None]:
    """Silence structlog for the duration of a corpus run.

    A corpus run's *only* product is the report. A single ``log.debug`` from a
    module under measurement lands in the middle of the ``--json`` document and
    every downstream parser fails on it, so logs are suppressed here rather than
    in each runner.

    Scoped, and restored in a ``finally``, deliberately: re-configuring structlog
    process-wide and leaving it that way makes the behaviour of every later call
    depend on whether a corpus ran first. An earlier version of this did exactly
    that — it re-bound structlog to the current ``sys.stderr`` with
    ``cache_logger_on_first_use=False`` — and under pytest's capture that left a
    log call writing to an already-closed stream, which raised inside the capture
    writer thread and hung the run.

    The previous configuration is snapshotted and put back verbatim rather than
    assumed: the restorer's job is to leave the process exactly as it found it,
    and re-deriving structlog's defaults is not the same as restoring what was
    there.
    """
    previous = structlog.get_config().copy()
    structlog.configure(
        logger_factory=structlog.ReturnLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    try:
        yield
    finally:
        structlog.configure(**previous)


def _cmd_eval_fixtures(args: argparse.Namespace) -> int:
    """``sentinel eval-fixtures --module provenance|memory`` (``S3-T13``/``S4-T12``).

    No production store is involved: the corpus *is* the event sequence, so this
    runs anywhere, including in CI without a database. The memory runner builds
    its own throwaway SQLite store because its evaluator goes through the real
    worker path, but nothing durable is touched.
    """
    from sentinel.eval.harness import CORPUS_RUNNERS, run_module_corpus

    if args.module not in CORPUS_RUNNERS:
        print(
            f"Unknown module {args.module!r}; known modules: {', '.join(sorted(CORPUS_RUNNERS))}",
            file=sys.stderr,
        )
        return 2
    with _quiet_logs():
        report = run_module_corpus(args.module)
    if args.json or args.pretty:
        _emit_json(args, report.to_dict())
    else:
        print(report.render())
    return 0 if report.passed else 1


async def _run_sessions_list(args: argparse.Namespace) -> int:
    since = _parse_ts(args.since)
    until = _parse_ts(args.until)
    if (args.since and since is None) or (args.until and until is None):
        print("--since/--until must be ISO timestamps", file=sys.stderr)
        return 2
    store = await _open_store(args.dsn)
    try:
        rows = await store.list_sessions(
            agent_id=args.agent,
            since=since,
            until=until,
            has_flags=args.flagged,
            limit=args.limit,
            offset=args.offset,
        )
    finally:
        await store.close()
    _emit_json(args, [_jsonable(r) for r in rows])
    print(f"{len(rows)} session(s)")
    for row in rows:
        print(
            f"  {row.session_id}  agent={row.agent_id or '-'}  {row.status}  "
            f"started={row.started_at.isoformat()}  events={row.event_count}  "
            f"flags={'yes' if row.has_flags else 'no'}"
        )
    return 0


async def _run_health(args: argparse.Namespace) -> int:
    store = await _open_store(args.dsn)
    try:
        health = await store.health()
    finally:
        await store.close()
    _emit_json(args, _jsonable(health))
    print(
        f"sessions={health.sessions} events={health.events} flags={health.flags} "
        f"tombstones={health.tombstones}"
    )
    if health.oldest_event is not None and health.newest_event is not None:
        print(f"span={health.oldest_event.isoformat()} -> {health.newest_event.isoformat()}")
    print(f"gaps: {health.gap_count} missing seqs across {health.sessions_with_gaps} session(s)")
    return 0


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `sentinel` command. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "func", None)
    if command is None:
        parser.print_help()
        return 0
    return int(command(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
