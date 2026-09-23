"""Sentinel command-line interface.

Private module. The `sentinel` console script is installed from here, but this
module is not part of the public API surface (docs/adr/0009).

Sprint ``S0`` adds ``sentinel replay <session_id>`` which prints a human-readable
trace of a captured session from the event store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sentinel import __version__
from sentinel.store.sqlite import SQLiteEventStore


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

    replay = subparsers.add_parser(
        "replay", help="print a captured session as a human-readable trace"
    )
    replay.add_argument("session_id", help="the session ID to replay")
    replay.add_argument(
        "--store",
        default="sentinel.sqlite3",
        help="path to the SQLite event store (default: sentinel.sqlite3)",
    )
    replay.set_defaults(func=_cmd_replay)

    return parser


async def _replay_events(store_path: str, session_id: str) -> int:
    store = SQLiteEventStore(store_path)
    try:
        events = await store.get_session(session_id)
    finally:
        await store.close()
    if not events:
        print(f"No events found for session {session_id} in {store_path}", file=sys.stderr)
        return 1
    print(f"Session {session_id} — {len(events)} events")
    for event in events:
        summary = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
        if len(summary) > 120:
            summary = summary[:117] + "..."
        print(
            f"  #{event.seq} {event.ts.isoformat()} {event.type} refs={len(event.refs)} {summary}"
        )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    return asyncio.run(_replay_events(args.store, args.session_id))


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
