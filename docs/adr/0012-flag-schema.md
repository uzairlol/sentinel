# 0012 — The universal flag row

Date: 2026-09-27

## Status
Accepted (Sprint `S3`)

## Context

Every evaluator module — provenance now, memory integrity in `S4`, reasoning
faithfulness in `S5` — has to answer the same four questions for a finding: what
was claimed, which events bear on it, how bad is it, and has a human looked yet.
`S7` then has to build a gate and a review queue on top of those rows, and an
operator has to be able to open one and understand it in a minute.

That creates a pull in two directions. A universal schema (`S3-T1`) is what makes
a cross-module queue, a cross-module gate and a credible FP/FN story possible.
But a schema owned by the first module tends to encode that module's private
details: a `provenance`-shaped column here, a review-tool URL there, a
confidence meaning that differs per module. The second module then either
misuses the column or forces a migration, and by `S6` the schema is unrecognizable.

Two constraints make the answer harder. Flags must be **deterministic** (`S3-T4`)
— the same log and the same module version must produce byte-identical rows, or
the measured FP/FN numbers describe a different module run than the one
deployed. And flags must respect the append-only store (`ADR-0011`): a flag is
written once and is thereafter *adjudicated*, not edited, or the audit trail
becomes a record of what someone last thought rather than what was found.

## Decision

`Flag` (`src/sentinel/models/flags.py`, schema version `0.1`) carries:

| Field | Purpose |
|---|---|
| `flag_id` | ULID derived from SHA-256 of `(session_id, module, module_version, category, dedupe_key)`. Pure function of its inputs: a re-run rewrites the same row. |
| `session_id`, `event_id` | Where to look. `event_id` is the primary trigger when there is one. |
| `module`, `module_version` | Who is complaining, and which version of the rules. The version is in the identity on purpose: a rules change is a new finding, never an overwrite. |
| `category` | `lower_snake_case` slug, open taxonomy via `register_category()`. `provenance` registers `ungrounded_claim`, `contradicted_claim` and `unsourced_citation`. |
| `severity` | `info`/`low`/`medium`/`high`/`critical`, default `medium`, ordered and comparable. |
| `confidence` | `[0, 1]`. Meanings differ per module, so the definition is the module's, not the schema's. |
| `summary` | One human-readable line, ≤ 2000 chars. Flags are read one at a time. |
| `evidence` | 1–32 `EvidenceRef`s, each an `event_id` + `role` + optional `seq`/`note`. **Non-empty by construction.** |
| `created_at` | Supplied by the caller, normally the evidence event's timestamp. Never a clock read. |
| `review_only` | Routes to the review queue instead of the gate (`S3-T15`). |
| `details` | Module-specific JSON. The escape hatch, and the only place a module may put its own vocabulary. |
| `adjudication`, `adjudicated_by`, `adjudicated_at` | Human-review state (`S7`). First write wins. |
| `auto_resolved` | Set by the module, not a human, when a session is superseded or a finding no longer holds. |
| `schema_version` | `0.1`, so a reader can tell what it is looking at. |

Three decisions inside that are worth stating outright:

**Evidence is non-empty and role-typed.** A flag with no evidence is a shrug, and
a reviewer who cannot get from a flag to the events that justify it will not
review it. The role (`claim`, `evidence`, `context`, `countervailance`) is part of
the contract because the order carries the argument: the claim first, then the
sources, then whatever context was kept, with the refuting value marked.

**The review link lives in `details`, not in a column.** A URL is
deployment-shaped: it needs a host, a route, and a template that knows the
session id. A column would be either nullable-and-mostly-null or a guess about
someone's review tool. The provenance module writes
`details["review_url"]` when the deployment supplies a template and is silent
otherwise.

**Adjudication is a state transition, not an edit.** `adjudicate_flag()` accepts
only a pending flag and refuses to overwrite a decision, so the row still says
what the module found even after a human disagrees with it. A decision also
cannot be recorded without its author and instant, and those cannot be set
without a decision: the row has to answer "who ruled, and when" on its own,
because the events it points at are eventually pruned. This is what keeps
`review_only` + adjudication safe to build a gate on in `S7`.

Confidence is deliberately *not* in the identity. A finding's confidence changing
while its inputs stay the same should update the row, not fork it; if a change is
big enough to change the verdict, `module_version` is the right lever.

## Consequences

- Positive: one gate and one review queue in `S7` can span every module, and a
  session-level rollup ("which sessions have any flag at or above `high`?") is a
  single query.
- Positive: re-running a worker is free. `put_flag` is an upsert on a
  deterministic id, so retries and restarts cannot double-flag.
- Positive: FP/FN numbers are reproducible — nothing in a flag depends on when
  the evaluator ran.
- Positive: `sentinel_reviewer` may `UPDATE` flags while `sentinel_writer` stays
  append-only, so the review workflow does not weaken the audit boundary
  (`ADR-0011`).
- Negative: `details` is untyped JSON. A consumer that wants `observed_value`
  must know it is talking to the provenance module; the alternative was a
  migration per module per field.
- Negative: `auto_resolved` is a second, weaker lifecycle path next to
  adjudication. It is nullable for exactly this reason and is expected to stay
  rare.
- Neutral: `confidence` semantics are per-module. Cross-module confidence
  comparison is a `S7` concern and should be handled by severity first.

## Alternatives considered

- **One flag row per event, no `event_id` column** — rejected: the natural
  question is "which event raised this?", and answering it by scanning 32 JSONB
  entries is a query nobody will write correctly.
- **A `review_url` column** — rejected: see above. Deployment-shaped data does not
  belong in the universal schema; it is the module's business to render.
- **Mutable flags (re-running overwrites severity/confidence in place, no
  identity hash)** — rejected: it destroys the audit trail and makes "did the
  rules change the answer?" unanswerable after the fact.
- **`flag_id` from `uuid4` per write** — rejected: retries become duplicate flags
  and idempotency is lost, which is the whole of `S3-T4`.
- **Per-module flag tables** — rejected: no cross-module queue, no single gate
  query, and a migration per module.
