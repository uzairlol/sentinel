# Sentinel Flag Schema

A **flag** is one module's finding about one session: what the agent claimed,
which events bear on it, how serious it is, and whether a human has ruled on it.
This is the contract `S7` builds a gate and a review queue on, and the contract
the FP/FN numbers in [`docs/modules/provenance.md`](modules/provenance.md) are
measured against. The reasoning is in
[`docs/adr/0012-flag-schema.md`](adr/0012-flag-schema.md).

Events are what happened ([`event-schema.md`](event-schema.md)). Flags are what
a module concluded about it. Events are immutable and append-only; a flag is
written once and thereafter adjudicated.

## 1. Fields

`sentinel.models.flags.Flag`, schema version `0.1`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `flag_id` | `str` (ULID) | derived | Deterministic identity. See §2. |
| `session_id` | `str` (ULID) | required | The session the finding is about. |
| `event_id` | `str \| None` | `None` | Primary triggering event, when there is one. |
| `module` | `str` | required | Who is complaining, e.g. `sentinel.tool_grounding`. |
| `module_version` | `str` | required | Which version of the rules. Part of the identity. |
| `category` | `str` slug | required | `lower_snake_case`, open taxonomy. |
| `severity` | `Severity` | `medium` | `info` / `low` / `medium` / `high` / `critical`. |
| `confidence` | `float` `[0, 1]` | required | The module's certainty. Meaning is per-module. |
| `summary` | `str` (≤ 2000 chars) | required | One human-readable line. |
| `evidence` | `list[EvidenceRef]` (1–32) | required | The events the finding rests on. **Never empty.** |
| `created_at` | `datetime` (UTC) | required | Supplied by the caller. Never a clock read. |
| `review_only` | `bool` | `False` | Queue for review; never gate. |
| `details` | `dict[str, Any]` | `{}` | Module-specific payload. |
| `adjudication` | `Adjudication` | `pending` | `pending` / `confirmed` / `rejected`. |
| `adjudicated_by` | `str \| None` | `None` | Who ruled. |
| `adjudicated_at` | `datetime \| None` | `None` | When they ruled. |
| `auto_resolved` | `bool \| None` | `None` | Module-side resolution, distinct from a human decision. |
| `schema_version` | `str` | `"0.1"` | Flag schema version (`S3-T1`). |

`Flag` is `extra="forbid"`: a typo in a field name is an error, not a silently
dropped column.

## 2. Identity

```
flag_id = ULID(sha256(session_id ␟ module ␟ module_version ␟ category ␟ dedupe_key)[:16])
```

(␟ = `\x1f`, a separator that cannot appear in any part.) The result is a valid,
sortable-looking 26-character ULID that is *purely a function of its inputs*.

Two consequences, both required:

- **Re-running a worker is free.** `put_flag` is an upsert on `flag_id`, so
  retries, restarts and forced re-evaluation cannot double-flag.
- **A rules change is a new finding.** `module_version` is in the identity, so
  shipping new rules produces new rows instead of silently rewriting what the
  old rules found. History stays readable.

`dedupe_key` is the module's identity for a finding *within* one
session/module/category — the provenance module uses
`f"{event_id}:{claim_id}"`, so the same claim in the same response is one flag
however many times it is evaluated.

Confidence is deliberately **not** in the identity: a confidence change with
unchanged inputs should update the row, not fork it.

## 3. Evidence

Every flag points at real events. This is not decoration — a flag whose evidence
cannot be opened is a shrug, and a reviewer who cannot check a finding will stop
reviewing them.

```python
EvidenceRef(event_id=..., role=..., seq=..., note=...)
```

| Field | Type | Meaning |
|---|---|---|
| `event_id` | `str` (ULID) | Must name a persisted event in the same session. |
| `role` | `EvidenceRole` | Why this event is in the list. |
| `seq` | `int ≥ 0 \| None` | The event's `seq` at evaluation time, so it is fast to find. |
| `note` | `str \| None` | A short quote of the relevant span. |

| Role | Meaning |
|---|---|
| `claim` | The event carrying the assertion that triggered the flag. |
| `evidence` | The event the claim was, or should have been, grounded in. |
| `context` | Surrounding context kept for the reviewer. |
| `countervailance` | The observed value that *contradicts* the claim. |

Order carries the argument, so producers put the claim first: read top to bottom
and you get the claim, its sources, and — for a contradiction — the line that
refutes it. `MAX_EVIDENCE` is 32: enough for a claim plus its sources, small
enough that the flag stays a thing a person opens.

## 4. Categories

Open taxonomy, `lower_snake_case`. Modules register what they ship via
`register_category()`, so tooling and docs cannot drift from the code:

| Category | Module | Meaning |
|---|---|---|
| `ungrounded_claim` | `sentinel.tool_grounding` | A specific checkable assertion no tool result supports — including a count that narrows a larger set. |
| `contradicted_claim` | `sentinel.tool_grounding` | An assertion the cited evidence refutes. |
| `unsourced_citation` | `sentinel.tool_grounding` | An assertion naming a source the session never cited. Carries `details["claimed_source"]`. |

Registration is advisory: a category validates as a slug regardless, so a module
can ship a category before the docs catch up.

## 5. Lifecycle

```
        evaluate
            │
            ▼
      ┌───────────┐  confidence ≤ threshold   ┌──────────────┐
      │  pending  │ ───────────────────────►  │ review_only  │ ─┐
      └─────┬─────┘                          └──────────────┘  │
            │ no threshold hit                                │
            ▼                                                 ▼
     gate candidate                                    review queue (S7)
            │
            │ adjudicate_flag()  (first write wins)
            ▼
      confirmed / rejected
```

- **`review_only=True`** means "never gate, always queue" (`S3-T15`). It is a
  property of the finding's certainty, not of the reviewer, and adjudication
  does **not** clear it: a human confirming a queued finding is not the same
  claim as the module being sure of it, and `S7`'s gate reads adjudication
  rather than `review_only` when it wants a human-backed decision.
- **`adjudicate_flag(flag_id, adjudication, adjudicated_by=…)`** accepts only a
  pending flag and refuses to overwrite a decision — a second reviewer's ruling
  does not replace the first — so the row keeps recording what the module found
  even after a human disagrees. It returns `False` when nothing changed, whether
  the flag is unknown or already decided; re-read to tell the two apart.
- **`auto_resolved`** is the module's own lifecycle path (a session superseded,
  a finding that no longer holds). It is nullable for exactly that reason and
  should stay rare; a human decision belongs in `adjudication`.

## 6. Storage

| Backend | Table | Notes |
|---|---|---|
| SQLite | `flags` (`src/sentinel/store/sqlite.py`) | `evidence` as a JSON column. |
| Postgres | `flags` (`src/sentinel/store/models.py`) | `evidence` as `JSONB`, indexed. Migration `0002_flags_s3`. |

Indexed for the queue: `(session_id)`, `(module, module_version)`,
`(adjudication, review_only)`, `(severity)`, `(created_at)`.

Protocol (`EventStore`):

```python
async def put_flag(self, flag: Flag) -> bool              # True = inserted, False = already there
async def put_flags(self, flags: Sequence[Flag]) -> int   # count newly inserted
async def get_flags(self, *, session_id=None, module=None, category=None,
                    min_severity=None, min_confidence=None, adjudication=None,
                    review_only=None, limit=100, offset=0) -> list[Flag]
async def adjudicate_flag(self, flag_id, adjudication, *, adjudicated_by,
                          at=None) -> bool                # False = unknown flag
```

`get_flags` orders newest first and is the review queue's read path
(`review_only=True, adjudication=Adjudication.PENDING`).

## 7. Roles

| Role | On `flags` | Why |
|---|---|---|
| `sentinel_writer` | `INSERT`, `SELECT` | Evaluator workers write findings; append-only (ADR-0011). |
| `sentinel_reviewer` | `SELECT`, `UPDATE` on `adjudication`, `adjudicated_by`, `adjudicated_at`, `auto_resolved` **only** | Reviewers adjudicate, and that is the whole of their power. No `INSERT` — a reviewer cannot manufacture a finding — and no `DELETE`, so a rejected finding stays on the record. |
| `sentinel_reader` | `SELECT` | Replay, audit, reporting. |

The reviewer's `UPDATE` is granted at **column scope**, not table scope, so
Postgres rejects any attempt to rewrite `summary`, `confidence`, `severity`, or
`evidence`. Adjudicating a finding therefore cannot alter what the finding says.
`tests/integration/test_db_roles.py::test_reviewer_may_only_adjudicate_flags`
connects as the role and asserts both halves: all four adjudication columns can
be written, while `INSERT` into `flags`, `DELETE FROM flags`, any write to
`events`, and an `UPDATE` of `flags.summary` are all rejected with
`permission denied`.

`sentinel_writer` holds `INSERT, SELECT` and has `UPDATE`/`DELETE`/`TRUNCATE`/
`REFERENCES`/`TRIGGER` revoked outright, so a live writer credential cannot
rewrite or destroy evidence — see `S2-T3`.

## 8. Invariants

Enforced by the model and covered by `tests/unit/test_flag_schema.py` and
`tests/contract/test_store_parity.py`:

1. `flag_id` is a pure function of the identity tuple.
2. `evidence` has at least one entry; every `event_id` is a valid ULID; no
   `(event_id, role)` pair repeats.
3. `confidence` is within `[0, 1]`; `summary` is non-empty and bounded.
4. `created_at` is timezone-aware and UTC-normalized.
5. A decided flag carries `adjudicated_by` and `adjudicated_at`; a pending one
   carries neither.
6. Two stores, one behaviour: write, filter, order and adjudicate identically —
   including refusing to adjudicate twice.
