# Sentinel — Technical Design Document & Development Roadmap

> **Working title:** Sentinel
> **Subtitle:** A Unified Runtime Safety Instrumentation Layer for Autonomous LLM Agents
> **Document version:** v1.0 — Master Engineering Plan
> **Supersedes:** `safety_sdk.tex` (v0.1 draft, conceptual only)
> **Status:** Living document. Every sprint below updates this file.
> **Audience:** The sole developer (primary), future collaborators, prospective design partners, technical due-diligence readers.
> **Execution model:** Solo-first, team-optional. Every sprint lists a solo estimate; a "team acceleration" note says how a 2–4 person team compresses it.

---

## 0. How To Use This Document

This file is written to be **machine-readable by an agent** as well as human-readable. If you are an AI coding agent picking this up, follow these rules:

1. **Find your place.** The current sprint state is recorded in [`§3.4 Roadmap Status Board`](#34-roadmap-status-board). Do not start work that belongs to a later sprint; do not leave unchecked items from the current sprint.
2. **One work item = one commit unit.** Every task has a stable ID of the form `S<N>-T<M>` (e.g. `S3-T4`). Reference the ID in commit messages and PR titles.
3. **Exit criteria are gates, not suggestions.** A sprint is *not* complete until every row of its **Exit Criteria** table is satisfied and the **GO / NO-GO** checklist is all-green. If a gate fails, the sprint stays open (see [`§3.3 Gate Failure Protocol`](#33-gate-failure-protocol)).
4. **Update this file in the same PR** that completes a work item: tick the checkbox, and if a design decision changed, add/update the ADR under `docs/adr/` and reference it here.
5. **Never break `main`.** `main` must always be installable, green, and releasable. All work flows through short-lived branches and PRs, even solo.
6. **Preserve history.** This is an event-sourced product; treat its own repository the same way — append, don't rewrite, design decisions.

### 0.1 Notation

| Symbol | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Done and verified |
| `S-1`, `S0`, `S1` … | Sprint identifiers (`S-1` = pre-work sprint, maturity `-1 → 0`) |
| `ADR-NNNN` | Architecture Decision Record number |
| `DoD` | Definition of Done (global, see [`§2.4`](#24-global-definition-of-done)) |
| `P0/P1/P2` | Priority: P0 = blocking, P1 = required for sprint, P2 = nice-to-have |
| Maturity `N` | Project completion percentage on the `-1 → 101` scale |

---

## 1. Vision, Scope, and the `-1 → 101` Maturity Scale

### 1.1 The One-Sentence Vision

> Build the **deployable, self-hosted, engineering-grade runtime safety instrumentation layer** that gives any team shipping an autonomous LLM agent the ability to *see, prove, and gate* the specific failure modes that current AI-safety research has isolated — reasoning unfaithfulness, tool-provenance fabrication, evaluation-aware sandbagging, memory corruption, and specification gaming — without forcing them to adopt a new orchestration framework.

### 1.2 What "Done at 101" Means (The North Star)

The project reaches maturity **101** when all of the following are simultaneously true:

1. A stranger can `pip install sentinel-sdk` (or pull a container), point it at a LangChain / LangGraph / raw-Ollama agent, and get working event capture in **under 15 minutes** following the quickstart.
2. All **five detection modules** and the **gating layer** are implemented, tested against adversarial fixtures, and have **published false-positive / false-negative rates** on a reproducible test suite.
3. The full system runs **self-hosted** with a single documented deployment (one Postgres + N workers + policy service), and a restore-from-backup drill has been executed and documented.
4. There is a **public docs site**, a **versioned release** with a changelog, an **SBOM**, and **signed artifacts**.
5. A **design partner has run it against real production traffic** and the reviewer-time-per-flag metric is measured and acceptable.
6. A **compliance mapping** exists (EU AI Act / NIST AI RMF) and an **audit export** can be produced on demand.
7. The product is **distributable**: license, pricing/packaging, terms, security policy, and a support model exist — it can be sold.

Everything between `-1` and `101` is the sequenced plan in [`§4 Sprint Work Packages`](#4-sprint-work-packages) that gets from (1) a PDF to (7) a sellable product.

### 1.3 Maturity Anchor Points

| Maturity | Named milestone | What exists at this point |
|---|---|---|
| `-1` | **Ground zero (today)** | A conceptual `.tex` design document. No code, no repo. |
| `0` | **Foundations** | Repository, tooling, CI, ADRs, contribution rules. |
| `5` | **Vertical slice** | A trivial end-to-end path: instrument one Ollama call → store it → replay it. |
| `25` | **MVP** | Instrumentation + hardened event store + **2 provenance modules** catching real adverse sessions. |
| `50` | **Feature-complete beta** | All 5 modules + gating + review UI; end-to-end flagged demo. |
| `75` | **Validated** | Full adversarial suite green; security & supply-chain hardened; FP/FN published internally. |
| `90` | **Release-ready** | Performance/scale proven; docs site; packaged install; DR drilled. |
| `100` | **GA** | Public release, compliance pack, design-partner production run, support model. |
| `101` | **Post-GA / sustainable** | Traction, enterprise readiness, recurring distribution. |

### 1.4 In Scope / Out of Scope

**In scope:** runtime capture, evaluators for the five named failure modes, a policy/gating layer, self-hosted deployment, the SDK itself, its docs, its compliance artifacts.

**Explicitly out of scope (v1):** general-purpose observability dashboards, model training/fine-tuning, a new agent orchestration framework, cloud-hosted SaaS multi-tenancy, and adjudicating model safety in the abstract. Sentinel *instruments and gates*; it does not *replace* the agent.

---

## 2. Global Engineering Standards (Apply To Every Sprint)

These are non-negotiable and are checked in the per-sprint **Exit Criteria**. A task is only "done" if it satisfies the global DoD in [`§2.4`](#24-global-definition-of-done).

### 2.1 Recommended Stack (ADR-backed)

The following stack is **recommended and locked** unless an ADR supersedes it. It is chosen for (a) ecosystem fit, (b) solo-developer velocity, (c) self-hostability, and (d) production durability.

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.12+** | The agent ecosystem (LangChain, LangGraph, most tooling) is Python-first. Solo velocity beats raw performance here. |
| Async model | **asyncio-first**, sync shims where unavoidable | Capture must never block the host agent. |
| Packaging / env | **`uv`** + **`hatchling`** build backend, `pyproject.toml` | Fast, reproducible, lockfile committed (`uv.lock`). |
| Data validation | **Pydantic v2** at boundaries; `msgspec`/dataclasses for hot-path events | Validation at edges, speed in the hot loop. |
| Reference store | **PostgreSQL 16+** via `SQLAlchemy 2.0 (asyncio)` + `asyncpg` + **Alembic** | Query patterns are relational (joins across sessions/calls/flags). |
| Dev/test store | **SQLite (aiosqlite)** — dev and tests only, never prod | Zero-friction local dev; prod parity is Postgres. |
| Control plane API | **FastAPI** + Uvicorn | Policy API + review UI backend, typed and fast to build. |
| Review UI | **Static-first** (HTML + minimal vanilla JS, no JS build chain initially) | Ships without a frontend toolchain; upgradeable later. |
| Embeddings | **Local via Ollama embedding models** (pluggable interface) | Data residency; no third-party egress by default. |
| Judge model | **Local via Ollama by default** (pluggable) | Same reason; guarantees self-hosted core path. |
| Observability | **OpenTelemetry SDK** + Prometheus metrics + `structlog` | Vendor-neutral; exportable to self-hosted collectors. |
| Testing | **pytest**, `pytest-asyncio`, **Hypothesis**, `factory_boy`, `respx`, `freezegun` | Unit→property→adversarial→e2e. |
| Static analysis | **ruff** (lint + format), **mypy** (strict), **bandit** | Fast, strict, security-aware. |
| CI/CD | **GitHub Actions** + `python-semantic-release` | Conventional-commit-driven releases. |
| Docs | **MkDocs Material** + `mkdocstrings` | Docs-as-code, API docs from docstrings. |
| Supply chain | `uv.lock`, **pip-audit**, **CycloneDX SBOM**, PyPI **trusted publishing** | Reproducible, signed, auditable releases. |

> **Why not Rust/Go for the core?** The instrumented surface is Python (LangChain/LangGraph). A cross-language core adds FFI complexity and halves solo velocity for a first product whose bottleneck is detector quality, not CPU. Revisit in an ADR if hot-path profiling ever justifies it.

### 2.2 Architecture Recap (from the design doc, now binding)

Three layers, kept strictly independent:

```
        ┌──────────────────────────────────────────────────────────────┐
        │  HOST AGENT  (LangChain / LangGraph / raw Ollama / custom)    │
        └───────────────┬──────────────────────────────────────────────┘
                        │  wraps: LLM calls, tool calls+returns, memory I/O
        ┌───────────────▼──────────────────────────────────────────────┐
        │  (1) INSTRUMENTATION LAYER — capture only, no evaluation      │
        │      thin async wrappers → serialize events → event store     │
        └───────────────┬──────────────────────────────────────────────┘
                        │  append-only, seq-numbered, foreign-keyed
        ┌───────────────▼──────────────────────────────────────────────┐
        │  (2) EVENT STORE — Postgres (ref) / SQLite (dev)              │
        │      sessions · events · call-graph links · flags             │
        └───────────────┬──────────────────────────────────────────────┘
                        │  subscribe to stream
        ┌───────────────▼──────────────────────────────────────────────┐
        │  (3a) EVALUATION LAYER — one worker per module (5 modules)    │
        │       writes flags back into the same store                   │
        ├──────────────────────────────────────────────────────────────┤
        │  (3b) POLICY & GATING LAYER — thin rules engine               │
        │       proceed / hold-for-review / block → review API + UI     │
        └──────────────────────────────────────────────────────────────┘
```

**Binding design invariants (violating any of these requires an ADR):**

- **INV-1 — Capture ⊥ Evaluation.** The instrumentation layer performs *zero* analysis. It only serializes and writes. Evaluators never run inside the capture path.
- **INV-2 — Append-only.** Events are immutable. Corrections are new events; deletions are tombstones with audit records.
- **INV-3 — Provenance is structural.** Every claim/flag links by foreign key to the exact event(s) that triggered it. Never "ask the model if it's lying."
- **INV-4 — Gating is auditable in an afternoon.** The policy engine must be small enough for a security team to read completely.
- **INV-5 — Self-hosted is the core path.** No mandatory third-party network dependency anywhere in capture → evaluate → gate.
- **INV-6 — Fail-open on capture, fail-closed on gate (configurable).** Capture failures must never crash the host agent; gate failures must hold, not silently approve. Both are configurable per deployment.

### 2.3 Version Control & Collaboration Standard

- **Branching:** trunk-based development. `main` is protected and always releasable. Work happens on short-lived branches: `feat/s3-t4-provenance-diff`, `fix/…`, `chore/…`, `docs/…`.
- **Commits:** Conventional Commits — `feat(scope): …`, `fix(scope): …`, `refactor: …`, `perf: …`, `test: …`, `docs: …`, `build: …`, `ci: …`, `chore: …`. Scope = module (`instrument`, `store`, `eval.provenance`, `gate`, …). Task ID included in body.
- **PRs even solo:** every change gets a PR with the template below. Self-review is a real step. Squash-merge into `main`.
- **PR template (checklist):** linked task ID · what changed · why · tests added/updated · docs updated · ADR updated (if decision) · migration included (if schema) · breaking change? (semver note) · security impact · rollback method.
- **Tags:** annotated, GPG-signed. `vX.Y.Z` produced automatically by `python-semantic-release`.
- **SemVer policy:** public API stability is sacred. Breaking public API = major bump + deprecation cycle (see [`§2.6`](#26-public-api--stability-policy)).

### 2.4 Global Definition of Done

A work item is **Done** only when all hold:

1. Code merged to `main` via PR; CI fully green.
2. `ruff` lint + format clean; `mypy --strict` clean on changed modules.
3. Tests added: at least one happy-path and one failure/adversarial path. Coverage on `src/sentinel/` ≥ 90% (diff coverage enforced on PRs).
4. Public functions have type annotations and docstrings; new public API documented.
5. Structured logs/OTel spans added where the item crosses a boundary.
6. Security review done for anything touching inputs, secrets, SQL, or the network.
7. Docs and/or ADR updated; this roadmap's checkbox ticked.
8. No `TODO` without a linked issue; no commented-out code; no secrets in the tree.
9. Releasable: `main` can be packaged and installed at any commit.

### 2.5 Testing Standard (The Pyramid, Enforced)

| Layer | Tooling | What it proves | Gate |
|---|---|---|---|
| Unit | pytest | Each function/evaluator component in isolation | Coverage ≥ 90% core |
| Property | Hypothesis | Invariants (provenance diff symmetry, replay idempotency, ordering) | ≥ 1 property per core algorithm |
| Integration | pytest + real Postgres (testcontainers or service) + stubbed Ollama | Store, migrations, instrumentors, worker wiring | Green on every PR |
| Contract | Schema snapshot tests | Event/flag schema backward compatibility | Fails on unversioned breaking change |
| Adversarial | Hand-built known-good/known-bad sessions | Each evaluator catches its failure mode | Per-module fixtures |
| End-to-end | Toy agent, fully instrumented | Capture→evaluate→flag→gate round trip | Green before sprint ends |
| Performance | Locust/k6 + custom harness | Latency/throughput budgets | Per [`§2.7`](#27-performance-budgets) |
| Security | bandit, pip-audit, secret scan, fuzz (Hypothesis + `atheris` later) | No known vulns, no secret leakage | Per sprint gate |

**Rule:** every evaluator ships with a **published FP/FN rate against its adversary fixture set**. A detector with no measured error rate is not "done", it is a prototype.

### 2.6 Public API & Stability Policy

- **Public surface is deliberately small** and lives under `sentinel/__init__.py` and `sentinel/instrument/`. Everything else is private (`_`-prefixed modules, not exported).
- Public API is fully typed and documented. Additions are minor bumps. Removals/renames require a deprecation shim that emits `DeprecationWarning` for at least one minor cycle.
- Event schemas carry an explicit `schema_version`. Readers must support the previous minor schema. A breaking schema change requires a migration + ADR.

### 2.7 Performance Budgets

| Path | Budget | Enforcement sprint |
|---|---|---|
| Instrumentation overhead per event | p99 < 5 ms added, non-blocking | S2 (measure), S10 (optimize) |
| Capture losslessness at target load | 0 dropped events at 200 events/s | S2 |
| Event write (local Postgres) | p99 < 10 ms — **measured 2026-09-24: 1.45 ms batch / 6.54 ms single** (`perf/write-benchmark.md`) | S2 |
| Claim extraction | p95 < 500 ms per output | S3 |
| Embedding drift eval | p95 < 1 s per memory write | S4 |
| Flag availability after eval | < 60 s end-to-end | S7 |
| Gate decision | < 200 ms added at checkpoint | S7 |

### 2.8 Security & Privacy Standard

- **Secrets:** environment via `pydantic-settings`; never logged; never committed; `.env` git-ignored; secret scanning in CI. Rotation documented in `docs/runbooks/`.
- **Redaction by default:** a configurable redaction pass runs before persistence; auth tokens/API keys rejected; PII redaction strategies per-field (`none`/`mask`/`hash`/`drop`) are configuration, defaulting to the safest option that still yields useful flags.
- **Data residency:** all core processing is local. Any external model/judge call is opt-in and clearly surfaced in config and logs.
- **SQL:** parameterized statements only; ORM/query-builder enforced; no string-built SQL. Least-privilege DB roles (writer vs reader vs migrator).
- **Transport:** TLS for all DB/collector connections in production.
- **Supply chain:** lockfile committed; `pip-audit` in CI blocks on high/critical; CycloneDX SBOM published with each release; releases signed.
- **Responsible disclosure:** `SECURITY.md`, `security.txt`, a contact, and a stated response window before GA.

### 2.9 Documentation Standard

- **ADRs** in `docs/adr/NNNN-title.md` for every significant decision (see [`§2.1`](#21-recommended-stack-adr-backed) for the initial set). Format: Context → Decision → Consequences → Alternatives.
- **Inline docs:** docstrings on all public APIs (Google style), rendered by `mkdocstrings`.
- **Runbooks** in `docs/runbooks/` for each failure mode of *the system itself* (DB down, worker stuck, capture gaps, queue backlog, restore from backup).
- **Changelog** auto-generated from conventional commits.
- **Operator guide** separate from **developer guide** separate from **quickstart**.

### 2.10 Production Operations Standard

- **Deployment:** reference target is `docker compose` (Postgres + N evaluator workers + policy/review service). Helm chart is a post-GA option, not a v1 gate.
- **Config:** 12-factor; all config via env/`pydantic-settings`; no code changes for tuning thresholds; every threshold documented with a safe default.
- **Observability of the observer:** Sentinel monitors *itself* — capture completeness (session/sequence-gap detection), worker lag, flag latency, gate latency, DB health. This is required because a blind safety layer is worse than none.
- **SLOs (initial):** capture completeness 99.9%, event write p99 < 10 ms, evaluator lag < 60 s, gate decision < 200 ms, availability 99.5% for the policy service.
- **Backups/DR:** WAL archiving + PITR; documented retention; **a restore drill must be executed and recorded** before GA.
- **Incident process:** severity levels, an incident doc template, blameless post-mortems, runbook-first response. Solo mitigation: automated alerts must be actionable within 30 minutes and every alert links to a runbook.
- **Release process:** CI builds artifacts → canary in a staging compose → promote → rollback by version pin. Every release is reproducible from a tag.

### 2.11 Gate Failure Protocol

If a sprint's exit gate fails:
1. The sprint **stays open**; do not start the next sprint's dependent work.
2. Record the failure in [`§3.4 Roadmap Status Board`](#34-roadmap-status-board) with the blocking task ID.
3. Open an issue labeled `gate-failure` describing the observed vs required result.
4. Decide explicitly: **fix**, **rescope** (with an ADR if the design changes), or **descope** (drop the feature from v1, recorded as an ADR). Never silently skip a gate.

---

## 3. The Roadmap: Phases, Sprints, and Maturity

### 3.1 Sprint Sequence Overview

Estimates are **solo, realistic, with buffer**. A 2–4 person team can parallelize the ways noted per sprint, roughly halving wall-clock from Sprint 1 onward but not below the dependency chain (store must exist before evaluators, etc.).

| Sprint | Title | Maturity | Solo est. | Phase |
|---|---|---|---|---|
| `S-1` | Foundations & Accountability | `-1 → 0` | 1–2 wk | Phase 0 — Bootstrap |
| `S0` | Vertical Slice "Hello Sentinel" | `0 → 5` | 1 wk | Phase 0 — Bootstrap |
| `S1` | Instrumentation Layer Core | `5 → 15` | 3 wk | Phase 1 — Capture |
| `S2` | Event Store Hardening & Query/Replay | `15 → 25` | 3 wk | Phase 1 — Capture |
| `S3` | Module: Tool-Use Grounding & Provenance | `25 → 35` | 3–4 wk | Phase 2 — Evaluate |
| `S4` | Module: Memory & Context Integrity | `35 → 45` | 3 wk | Phase 2 — Evaluate |
| `S5` | Module: Reasoning Faithfulness | `45 → 55` | 4 wk | Phase 2 — Evaluate |
| `S6` | Module: Specification Gaming & Objective Drift | `55 → 65` | 3 wk | Phase 2 — Evaluate |
| `S7` | Policy, Gating & Review Interface | `65 → 75` | 3 wk | Phase 3 — Act |
| `S8` | Module: Evaluation-Awareness & Sandbagging | `75 → 80` | 4 wk | Phase 3 — Act |
| `S9` | Adversarial Validation, Security & Supply Chain | `80 → 85` | 3 wk | Phase 4 — Harden |
| `S10` | Performance, Scale, Reliability & DR | `85 → 90` | 3 wk | Phase 4 — Harden |
| `S11` | Documentation, Packaging & DX | `90 → 95` | 3 wk | Phase 5 — Ship |
| `S12` | Compliance & Governance Readiness | `95 → 98` | 3 wk | Phase 5 — Ship |
| `S13` | Design-Partner Launch & Production Tuning | `98 → 100` | 4–6 wk | Phase 6 — Operate |
| `S14` | GA, Distribution & Traction | `100 → 101` | ongoing | Phase 6 — Operate |

**Critical path:** `S1 → S2 → S3 → S7 → S9 → S10 → S11 → S13 → S14`. Everything else can slip modestly without blocking the sellable product; those nine cannot.

**Total to GA (`100`):** ~44–52 focused weeks ≈ **11–13 months solo**, plus contingency. Plan for **18 months to GA** with real-life variance. Traction (`101`) is ongoing.

### 3.2 Phase Intent

- **Phase 0 — Bootstrap (`S-1`,`S0`):** make the project real and de-risked before any feature work. Establish accountability, tooling, CI, and a thin end-to-end guess.
- **Phase 1 — Capture (`S1`,`S2`):** the foundation everything depends on. Capture must be lossless, fast, and queryable. Do not rush this.
- **Phase 2 — Evaluate (`S3`–`S6`):** build detectors as independent workers. Provenance-based modules first (structurally verifiable), probabilistic modules later (need design care).
- **Phase 3 — Act (`S7`,`S8`):** turn flags into action (gating) and add the hardest statistical module (evaluation-awareness).
- **Phase 4 — Harden (`S9`,`S10`):** adversarial validation, security, supply chain, scale, DR. This phase is why the product can be trusted.
- **Phase 5 — Ship (`S11`,`S12`):** docs, packaging, compliance artifacts. This is the difference between "a repo" and "a product".
- **Phase 6 — Operate (`S13`,`S14`):** real traffic, tuning, GA, distribution, traction.

### 3.3 Gate Failure Protocol

See [`§2.11`](#211-gate-failure-protocol). Restated for emphasis: **a failed gate stops progression.** Record it in the Status Board below.

### 3.4 Roadmap Status Board

> Update this table at the end of every sprint. `[ ]` = not started, `[~]` = in progress, `[x]` = passed gate.

| Sprint | Maturity | Status | Gate result | Notes / blockers |
|---|---|---|---|---|
| `S-1` | `-1 → 0` | `[x]` | Passed | Gate green on `main` (lint, typecheck, test 3.12/3.13, security, build, docs). Branch protection enabled, `v0.0.1` GPG-signed tag. |
| `S0` | `0 → 5` | `[x]` | Passed | Vertical slice green on `main`: event envelope, session/raw-`instrument_ollama_call` capture, SQLite store, `sentinel replay` CLI, E2E vs respx-stubbed Ollama. 31 tests / 99.3% coverage, `mypy --strict` clean, `v0.0.2` tagged. |
| `S1` | `5 → 15` | `[x]` | Passed | Instrumentation layer core green on `main`: event taxonomy + INV-3 refs, registry/config, capture worker (bounded queue, fail-open, redaction), LangChain + LangGraph + raw-Ollama/OpenAI-compat + generic `trace` + memory adapters, call-graph query helper. 115 tests / 95.3% coverage, `mypy --strict`/ruff/format/pre-commit clean, `v0.0.3` tagged. `S1-T15` sampling and `S1-T16` streaming caps deferred to `S2`. |
| `S2` | `15 → 25` | `[x]` | Passed | Event store hardening green on `main`: Alembic migrations, Postgres store (batched append, streaming replay, call graph, session listing, health, gap detection, keyset retention prune with tombstones), SQLite/Postgres parity contract, 1M-event losslessness gate, truncation (`S1-T16`) + sampling (`S1-T15`), least-privilege DB roles + compose reference stack. 190 tests / 92.4% coverage, `mypy --strict`/ruff/bandit clean; `S2-T19` backup/restore smoke skips where `pg_dump`/`psql` are absent (runs in CI with tooling). |
| `S3` | `25 → 35` | `[~]` | Conditional | First detector green on `main`: universal `Flag` row (deterministic identity, typed evidence, first-write-wins adjudication, ADR-0012), `EvaluatorWorker` (checkpoints, retries, batching, `SessionView`, call graph), claim extraction + deictic grounding lexicon, contradiction diff with `observed_value` in evidence, review routing, 22-case adversarial corpus. **FP 0.00%, FN 0.00%** via `sentinel eval-fixtures --module provenance`; 400 offline tests at 91% coverage plus 114 integration tests (0 skipped) with the Postgres-only gate at 95% via `.coveragerc.postgres`, `mypy --strict`/ruff/bandit clean. **Conditional, not complete:** contradiction detection ships, fabricated *citations* and cherry-picked numbers do not — see "Known gaps carried out of `S3`". Docs: [`docs/modules/provenance.md`](../modules/provenance.md), [`docs/flag-schema.md`](../flag-schema.md). |
| `S4` | `35 → 45` | `[ ]` | — | — |
| `S5` | `45 → 55` | `[ ]` | — | — |
| `S6` | `55 → 65` | `[ ]` | — | — |
| `S7` | `65 → 75` | `[ ]` | — | — |
| `S8` | `75 → 80` | `[ ]` | — | — |
| `S9` | `80 → 85` | `[ ]` | — | — |
| `S10` | `85 → 90` | `[ ]` | — | — |
| `S11` | `90 → 95` | `[ ]` | — | — |
| `S12` | `95 → 98` | `[ ]` | — | — |
| `S13` | `98 → 100` | `[ ]` | — | — |
| `S14` | `100 → 101` | `[ ]` | — | — |

---

## 4. Sprint Work Packages

Each package has the same shape: **Objective · Maturity · Duration · Deliverables · Task Breakdown · Standards Focus · Tests & Verification · Exit Criteria · Risks · Dependencies · Handoff · GO/NO-GO**.

---

### Sprint `S-1` — Foundations & Accountability

**Objective.** Turn a conceptual design document into an accountable, tooled, version-controlled project with a locked stack, a CI pipeline, and the rules of engagement that every later sprint relies on. No product features.

**Maturity:** `-1 → 0` · **Duration:** 1–2 weeks · **Phase:** 0 Bootstrap
**Team acceleration:** N/A — this is cheap and must be done once, alone.

#### Deliverables

- A public/private Git repository with `main` protected, branch rules, and a signed initial tag `v0.0.1`.
- A working Python package skeleton that builds and passes an empty test suite in CI.
- ADRs `0001`–`0010` recorded (initial decision set).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `LICENSE`, `README.md`, PR/issue templates.
- CI pipeline: lint, type-check, test, build, SBOM, dependency audit.
- This document (`SENTINEL_TDD.md`) committed as the source of truth, linked from `README`.

#### Task Breakdown

**Repository & governance**
- [ ] `S-1-T1` (P0) Create repository. Default branch `main`. Add branch protection: require PR, require CI green, dismiss stale approvals, disallow force-push and deletion.
- [ ] `S-1-T2` (P0) Choose and add `LICENSE` (recommendation: **Apache-2.0** for the core SDK; commercial license reserved for future enterprise features — record as ADR). Add copyright headers policy.
- [ ] `S-1-T3` (P0) Add `CONTRIBUTING.md` (workflow from [`§2.3`](#23-version-control--collaboration-standard)), `CODE_OF_CONDUCT.md`, `SECURITY.md` (disclosure policy), `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/{bug,feature,gate-failure}.md`.
- [ ] `S-1-T4` (P1) `.gitignore`, `.gitattributes` (LF normalization), `.editorconfig`.
- [ ] `S-1-T5` (P0) Commit `SENTINEL_TDD.md` and link it prominently from `README.md`.

**Project skeleton**
- [ ] `S-1-T6` (P0) `pyproject.toml` (hatchling) with name `sentinel-sdk`, version `0.0.1`, `requires-python = ">=3.12"`, dependency groups (`dev`, `test`, `docs`). Commit `uv.lock`.
- [ ] `S-1-T7` (P0) `src/sentinel/__init__.py` exposing `__version__` and a minimal public surface stub. Enforce `src/` layout.
- [ ] `S-1-T8` (P1) Configure `ruff` (lint rules + format), `mypy` (`strict = true`), `bandit`, and `.pre-commit-config.yaml` wiring them together with `end-of-file-fixer`/`trailing-whitespace`.
- [ ] `S-1-T9` (P1) `pytest` config (`pyproject.toml`), `tests/` skeleton with one trivial passing test and `pytest-cov` threshold config.

**Decision records**
- [ ] `S-1-T10` (P0) Write ADR `0001` — Python 3.12 async-first SDK. Alternatives: Rust/Go core, TS/Node.
- [ ] `S-1-T11` (P0) Write ADR `0002` — Postgres reference store, SQLite dev-only. Alternatives: ClickHouse, DuckDB, vector DB.
- [ ] `S-1-T12` (P0) Write ADR `0003` — capture/evaluation separation (INV-1).
- [ ] `S-1-T13` (P0) Write ADR `0004` — evaluators as independent workers on the event stream.
- [ ] `S-1-T14` (P0) Write ADR `0005` — gating as a minimal auditable rules engine.
- [ ] `S-1-T15` (P1) Write ADR `0006` — self-hosted core path; opt-in external models.
- [ ] `S-1-T16` (P1) Write ADR `0007` — versioned event schemas + backward-compatible readers.
- [ ] `S-1-T17` (P1) Write ADR `0008` — local Ollama defaults for embeddings/judge; pluggable interface.
- [ ] `S-1-T18` (P1) Write ADR `0009` — small stable public API surface; SemVer/deprecation policy.
- [ ] `S-1-T19` (P1) Write ADR `0010` — ULID event IDs + monotonic sequence for replay/idempotency.

**CI/CD foundation**
- [ ] `S-1-T20` (P0) GitHub Actions `ci.yml`: matrix (3.12, 3.13) → `ruff check`, `ruff format --check`, `mypy`, `pytest --cov` with threshold, package build.
- [ ] `S-1-T21` (P1) Security job: `pip-audit` (fail on high/critical), `bandit`, secret scan (`gitleaks`).
- [ ] `S-1-T22` (P1) `python-semantic-release` configured for conventional commits; dry-run works locally.
- [ ] `S-1-T23` (P2) Add `commitlint`/git hook to enforce conventional commit format locally.

#### Standards Focus

Governance, CI, ADR discipline, DoD. This sprint *defines* the standard later sprints are held to.

#### Tests & Verification

- [ ] `uv run pytest` passes (trivially).
- [ ] `uv run ruff check . && uv run ruff format --check .` clean.
- [ ] `uv run mypy src` clean (strict).
- [ ] `uv build` produces a wheel and sdist that `pip install` into a fresh venv.
- [ ] CI green on a throwaway PR proving branch protection works.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | CI is green on `main` and blocks merges when red | throwaway PR |
| 2 | Fresh clone → `uv sync` → `pytest` passes in a clean checkout | manual |
| 3 | ADRs 0001–0010 committed and indexed in `docs/adr/README.md` | review |
| 4 | Package builds and installs from wheel | `uv build` + fresh venv |
| 5 | Repo governance files present; branch protection active | review |
| 6 | `SENTINEL_TDD.md` committed and linked from README | review |

#### Risks & Mitigations

- **Over-engineering the scaffolding.** Mitigation: timebox to 2 weeks; anything beyond lint/type/test/audit/build is deferred.
- **License choice paralysis.** Mitigation: decide Apache-2.0 now via ADR; revisit before enterprise features.

#### Dependencies

None. This is the root.

#### Handoff

`sentinel-sdk` package skeleton with enforced quality gates. Next sprint builds the first vertical slice.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Status Board updated: `S-1` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If any row is red → **NO-GO**: resolve before `S0`.

---

### Sprint `S0` — Vertical Slice "Hello Sentinel"

**Objective.** Prove the entire pipeline end-to-end in the thinnest possible way: instrument **one** raw Ollama LLM call, capture the request/response as an event, persist it, and replay the session. No schema gymnastics, no evaluators. The point is to remove integration risk and validate the core abstraction before investing in it.

**Maturity:** `0 → 5` · **Duration:** 1 week · **Phase:** 0 Bootstrap
**Team acceleration:** N/A — intentionally tiny.

#### Deliverables

- A minimal `sentinel.instrument` decorator/wrapper that captures an Ollama chat call.
- A minimal async event-writer abstraction with a SQLite dev backend.
- A `sentinel replay <session_id>` CLI (or `python -m sentinel replay`) that prints the captured session.
- One end-to-end test that runs a fake Ollama endpoint and asserts a replayed event.

#### Task Breakdown

**Instrumentation proof**
- [x] `S0-T1` (P0) Define the **thin event envelope**: `event_id` (ULID), `session_id`, `seq` (monotonic int), `ts` (UTC), `type`, `payload`, `refs` (parent event IDs). Pydantic model, `schema_version="0.1"`.
- [x] `S0-T2` (P0) Implement `capture(event)` async interface + a `session()` context manager that allocates `session_id` and tracks `seq`.
- [x] `S0-T3` (P0) Implement an `instrument_ollama_call` wrapper using `httpx` transport hook or a callable wrapper around a POST to `/api/chat`. Capture request, response, latency, model name.
- [x] `S0-T4` (P1) Enforce INV-1 in code review: the wrapper contains no analysis, only serialization.

**Storage proof**
- [x] `S0-T5` (P0) Define `EventStore` protocol (`append(event)`, `get_session(session_id)`). Implement `SQLiteEventStore` (aiosqlite) with a single `events` table.
- [x] `S0-T6` (P0) Persist `event_id`/`session_id`/`seq` with a unique index on `(session_id, seq)`.

**Replay proof**
- [x] `S0-T7` (P0) Implement `replay_session(session_id)` returning ordered events; CLI prints a human-readable trace.
- [x] `S0-T8` (P1) Assert replay ordering and idempotency: replaying twice yields identical output.

**Verification & docs**
- [x] `S0-T9` (P0) E2E test: fake Ollama server (or `respx`) → instrumented call → SQLite → replay asserts content equality.
- [x] `S0-T10` (P1) Add a 20-line `examples/hello_ollama.py` and document it in `README`.
- [x] `S0-T11` (P1) Record an ADR amendment if the event envelope shape differs from the S-1 assumptions. (Envelope unchanged: `schema_version="0.1"`, ULID ids per ADR-0010 — no amendment required.)

#### Standards Focus

Public API shape (this is the first real public API), async correctness, the smallest viable schema, test-first integration.

#### Tests & Verification

- [x] Unit: envelope validation rejects missing required fields.
- [x] Integration: instrumented call against a stubbed Ollama writes exactly one request and one response event with correct `refs`.
- [x] E2E: replay output matches captured payload.
- [x] `mypy --strict` clean; coverage ≥ 90% on touched modules (99.3% measured).

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | One real-ish Ollama call captured, stored, and replayed losslessly | E2E test |
| 2 | Public API (`capture`, `session`, `instrument_ollama_call`) exists, typed, documented | review |
| 3 | INV-1 demonstrably honored (capture module imports no evaluator code) | dependency/import test |
| 4 | `examples/hello_ollama.py` runs with a local Ollama and prints a replay | manual |
| 5 | CI green; package version bumped to `0.0.2` via conventional commits | CI |

#### Risks & Mitigations

- **Choosing a transport hook that's too invasive.** Mitigation: prefer wrapping the call site over monkey-patching internals; document the supported integration pattern.
- **Premature schema complexity.** Mitigation: only the fields needed for replay now; evolve under `schema_version`.

#### Dependencies

`S-1` complete (tooling, CI, ADRs).

#### Handoff

A working "spine". Sprint `S1` widens capture to all boundaries (LLM, tools, memory) and adds real backends in `S2`.

#### GO / NO-GO Checklist

- [x] All Exit Criteria rows pass. (E2E lossless replay; public API typed via `mypy --strict`; INV-1 enforced by import-boundary test; example added; CI green, version `0.0.2`.)
- [x] The event envelope is frozen enough to build on (or an ADR records why it changed).
- [x] Status Board updated: `S0` = `[x]`.
- [x] No open `gate-failure` issue.
- [x] If capture is not lossless for the trivial case → **NO-GO**; fixing it is a prerequisite for `S1`.

---

### Sprint `S1` — Instrumentation Layer Core

**Objective.** Widen capture from the single proven Ollama path to **all three boundaries** — language-model calls, tool calls and their returns, and memory read/write operations — across LangChain, LangGraph, and raw HTTP. Deliver a stable, extensible instrumentor registry so future frameworks can be added without touching the core.

**Maturity:** `5 → 15` · **Duration:** 3 weeks · **Phase:** 1 Capture
**Team acceleration:** one dev on LangChain/LangGraph instrumentors, one on the raw HTTP + memory instrumentors, one on the registry/tests — ~2 weeks.

#### Deliverables

- A finalized public instrumentation API: session management, capture context, instrumentor registration.
- Framework instrumentors: LangChain (LLM + tools + memory), LangGraph (nodes/edges → call-graph events), raw Ollama/OpenAI-compatible HTTP, and a generic decorator for arbitrary functions.
- Explicit call-graph linking: tool-response events reference the tool-call event; outputs reference the LLM call; memory reads reference the writes they load.
- A capture pipeline that is asynchronous, batched, and **never blocks or crashes the host agent** (INV-6).
- Documentation of every event `type` and its payload schema.

#### Task Breakdown

**Public API & registry**
- [x] `S1-T1` (P0) Freeze the public surface: `sentinel.session()`, `sentinel.instrument.*` (langchain, langgraph, ollama, openai_compat, memory, generic), `sentinel.configure(...)`. Everything else private.
- [x] `S1-T2` (P0) Implement an instrumentor **registry** with `register()`/`enable()`/`disable()`; each instrumentor declares the event types it emits and is idempotent on double-enable.
- [x] `S1-T3` (P0) Implement `configure()` (pydantic-settings): store DSN, capture toggles, redaction policy, batching, sampling rate, fail-open behavior.

**Event taxonomy & linking (INV-3)**
- [x] `S1-T4` (P0) Define event types: `session.start`, `session.end`, `llm.request`, `llm.response`, `tool.call`, `tool.result`, `memory.read`, `memory.write`, `agent.step` (LangGraph node), `error`, `capture.dropped`.
- [x] `S1-T5` (P0) Define `refs` semantics: `parent`, `caused_by`, and specifically `grounds` (tool.result that a later claim can cite). Enforce referential integrity at append time.
- [x] `S1-T6` (P0) Add a call-graph query helper: given a session, return tool calls ↔ results ↔ dependent LLM calls.

**Framework instrumentors**
- [x] `S1-T7` (P0) **LangChain** — instrument LLM calls and tool invocations via callbacks/handlers; capture inputs, outputs, latency, model id, tool name.
- [x] `S1-T8` (P0) **LangGraph** — instrument node entry/exit as `agent.step` events, capturing state deltas and producing a navigable execution graph.
- [x] `S1-T9` (P0) **Raw HTTP (Ollama / OpenAI-compatible)** — transport-level capture with request/response bodies, streaming handled correctly (accumulate stream chunks without buffering the whole stream in memory before forwarding).
- [x] `S1-T10` (P1) **Generic decorator** — `@sentinel.instrument.trace(kind=...)` for custom functions/tools.
- [x] `S1-T11` (P1) **Memory adapter** — protocol for memory stores (vector, KV, custom) with `read`/`write` instrumentation; ship an in-memory and a Postgres-backed reference adapter.

**Capture pipeline robustness (INV-6)**
- [x] `S1-T12` (P0) Async batching writer with bounded queue; on overflow emit `capture.dropped` (counted, surfaced) rather than blocking the agent.
- [x] `S1-T13` (P0) Fail-open default for capture: exceptions in capture are logged, counted, and swallowed; the host call proceeds. Configurable to fail-closed.
- [x] `S1-T14` (P0) Redaction hook executed before persistence; block obvious secrets; configurable per-field strategy.
- [x] `S1-T15` (P1) Sampling support (capture N% of non-critical events) with guaranteed capture of errors and gating-relevant events. *(done in `S2`)*
- [x] `S1-T16` (P1) Streaming-safe serialization: cap payload size with truncation markers + hash, never store more than a configured maximum per event. *(done in `S2`)*

**Docs & examples**
- [x] `S1-T17` (P0) `docs/event-schema.md` describing every event type, payload fields, refs, and `schema_version` evolution rules.
- [x] `S1-T18` (P1) Examples: `examples/langchain_agent.py`, `examples/langgraph_agent.py`, `examples/raw_ollama.py`.
- [x] `S1-T19` (P1) `docs/integration-guide.md` describing supported frameworks and the generic decorator escape hatch.

#### Standards Focus

Public API stability, async correctness, back-pressure, redaction/security, schema documentation, INV-1/INV-3/INV-6 enforcement.

#### Tests & Verification

- [ ] Unit per instrumentor with mocked frameworks.
- [ ] Integration: each framework produces a correctly-linked call graph for a scripted multi-step run.
- [ ] Property (Hypothesis): `seq` is strictly monotonic per session under concurrent emission; no duplicate `event_id` ever.
- [ ] Failure injection: store raises → agent call still returns; `capture.dropped` recorded.
- [ ] Streaming test: a chunked LLM stream is reconstructed to the same payload as a non-streamed equivalent.
- [ ] Redaction tests: injected API key/token is absent from persisted payload.
- [x] Overhead micro-benchmark recorded (baseline for `S2`/`S10`): `perf/overhead-baseline.md` (2026-09-23).

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | A full LangChain agent session replays with every model call, tool call, tool result, and memory op present | integration test |
| 2 | Same for LangGraph and raw Ollama | integration tests |
| 3 | Call graph is explicit and queryable (tool result ↔ tool call ↔ dependent LLM call) | call-graph test |
| 4 | Capture never blocks the host under queue saturation; drops are recorded | failure-injection test |
| 5 | Secrets are redacted before persistence | redaction test |
| 6 | Every event type is documented in `docs/event-schema.md` | review |
| 7 | Instrumentation overhead measured and within an order of magnitude of budget | benchmark log |

#### Risks & Mitigations

- **Framework internals changing / brittle hooks.** Mitigation: prefer documented callbacks over monkey-patching; isolate each instrumentor in its own module; add a compatibility test pinned to a framework version and a matrix job.
- **Streaming capture complexity.** Mitigation: capture chunk metadata and a bounded reconstruction; never hold full streams unnecessarily.
- **Instrumentors leaking analysis logic.** Mitigation: an import-boundary test asserts `sentinel.instrument.*` imports nothing from `sentinel.eval.*`.

#### Dependencies

`S0` (envelope, store protocol, session context).

#### Handoff

Complete, lossless capture across all three boundaries. `S2` hardens storage, migration, replay, and throughput so evaluators can rely on it.

#### GO / NO-GO Checklist

- [x] All Exit Criteria rows pass. (Lossless capture per framework, call graph queryable, capture never blocks, secrets redacted, `docs/event-schema.md` complete, overhead benchmark recorded.)
- [x] `docs/event-schema.md` is complete and reviewed.
- [x] Import-boundary test (INV-1) passes in CI.
- [x] Status Board updated: `S1` = `[x]`.
- [x] No open `gate-failure` issue.
- [x] If any framework fails lossless replay → **NO-GO**; do not build evaluators on unreliable capture.

---

### Sprint `S2` — Event Store Hardening & Query/Replay

**Objective.** Turn the throwaway SQLite store from `S0` into the production-grade **Postgres-backed event store**: real migrations, indexing, append-only guarantees, retention, foreign-key integrity, a query layer, and a losslessness/throughput proof. This is the foundation every evaluator depends on.

**Maturity:** `15 → 25` · **Duration:** 3 weeks · **Phase:** 1 Capture
**Team acceleration:** one dev on schema/migrations/queries, one on performance/load and retention — ~2 weeks.

#### Deliverables

- `PostgresEventStore` implementing the `EventStore` protocol via SQLAlchemy 2.0 async + `asyncpg`, with `Alembic` migrations.
- Append-only enforcement at the DB level (revoked UPDATE/DELETE for the writer role; tombstones only via new events).
- Indexes and constraints tuned for the known query patterns (session replay, call graph, flagged sessions, time-range scans).
- Retention/compaction policy with documented compliance semantics.
- A **losslessness proof**: N events emitted under load → N persisted, zero gaps in `seq`.
- The `SQLiteEventStore` retained as dev-only with parity tests.

#### Task Breakdown

**Schema & migrations**
- [x] `S2-T1` (P0) SQLAlchemy 2.0 declarative models: `sessions`, `events`, `event_refs`, `flags`, plus a `schema_meta` table. UUID/ULID columns typed correctly.
- [x] `S2-T2` (P0) Alembic environment with async engine; initial migration `0001_initial`; migration test that applies to an empty DB and downgrades cleanly.
- [x] `S2-T3` (P0) Enforce append-only: writer role has `INSERT`/`SELECT` only; `UPDATE`/`DELETE` revoked. Document the three roles: `sentinel_migrator`, `sentinel_writer`, `sentinel_reader`.
- [x] `S2-T4` (P0) Constraints: unique `(session_id, seq)`; unique `event_id`; FK from `event_refs.event_id`/`ref_event_id` to `events`; `flags.event_id` FK.
- [x] `S2-T5` (P0) Indexes: `(session_id, seq)`, `(session_id, type, ts)`, `(ts)` for retention, `(flag.severity, flag.created_at)` for the gate, GIN on refs if needed.
- [x] `S2-T6` (P1) JSONB payload with a `schema_version` column; add a validation-on-read path for older schema versions (INV / ADR-0007).

**Query & replay layer**
- [x] `S2-T7` (P0) `get_session(session_id)` ordered by `seq`, plus `iter_session` streaming for large sessions.
- [x] `S2-T8` (P0) Call-graph query: `get_call_graph(session_id)` returning typed edges.
- [x] `S2-T9` (P0) Session listing/search by time range, agent id, and flag presence.
- [x] `S2-T10` (P0) Replay CLI upgraded: `sentinel replay --session <id> [--json|--pretty]`, and `sentinel sessions list`.
- [x] `S2-T11` (P1) A `store health` command reporting gaps, row counts, oldest/newest event, and worker lag placeholders for `S7`.

**Retention, integrity, operations**
- [x] `S2-T12` (P0) Retention policy engine: per-event-type TTL, legal-hold override, tombstone records for deletions, and a documented compliance story (references `S12`).
- [x] `S2-T13` (P0) Gap detector: given a session, assert `seq` is contiguous; emit an ops metric and a report for missing ranges.
- [x] `S2-T14` (P1) Backpressure/connection-pool tuning; bounded writer concurrency; timeouts and retry-with-jitter on transient DB errors (but never on append-only violations).
- [x] `S2-T15` (P1) `docker compose` reference deployment: Postgres + a capture service + migration runner.

**Verification**
- [x] `S2-T16` (P0) Losslessness load test: emit 1,000,000 events across 1,000 sessions; assert zero gaps and exact counts.
- [x] `S2-T17` (P0) Parity tests: SQLite and Postgres stores satisfy the same protocol contract tests.
- [x] `S2-T18` (P1) Migration compatibility: a fixture DB from the previous schema version reads correctly.
- [x] `S2-T19` (P1) Backup/restore smoke: `pg_dump`/restore a seeded DB and assert replay equality.

#### Standards Focus

Data modeling, migrations, DB security/least privilege, performance budgets, compliance-aware retention, reproducibility.

#### Tests & Verification

- [x] Integration against a real Postgres (testcontainers or CI service): CI `integration-postgres` job against a Postgres service container; exercised locally against Postgres 18.4.
- [x] Property: append is idempotent by `event_id`; re-appending is a no-op, not a duplicate: `test_store_parity.py::test_append_is_idempotent_by_event_id` on both stores + `tests/property/test_replay_and_seq.py`.
- [x] Property: `seq` ordering is total and gap-free per session: replay-order parity tests + `tests/property/test_replay_and_seq.py` + the 1M-event losslessness gate (zero gaps).
- [x] Perf: event write p99 < 10 ms at target concurrency; documented numbers: `perf/write-benchmark.md` — `append_batch` p99 1.45 ms/event, single `append` p99 6.54 ms (local Postgres, 2026-09-24).
- [x] Ops: gap detector correctly identifies a deliberately injected missing `seq`: `test_store_parity.py::test_detect_gaps` on a holey session (both stores).
- [x] Restore drill executed and recorded: `tests/integration/test_backup_restore.py` (pg_dump → restore → replay equality); runs in CI when `pg_dump`/`psql` are on PATH, skips locally.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | Migrations apply/downgrade cleanly; parity tests pass on SQLite + Postgres | CI |
| 2 | 1M-event losslessness test passes with zero gaps | load test |
| 3 | Event write p99 < 10 ms and overhead within budget | benchmark |
| 4 | Append-only enforced (UPDATE/DELETE rejected for writer role) | DB test |
| 5 | Retention + legal hold + tombstone semantics documented and tested | tests + doc |
| 6 | Replay CLI works against Postgres; `docker compose` brings up the stack | manual |
| 7 | Backup/restore smoke test recorded | runbook |

#### Risks & Mitigations

- **Retention semantics conflict with compliance.** Mitigation: make retention configurable, default to conservative, and defer final legal semantics to `S12` with an ADR.
- **Performance regressions from over-indexing.** Mitigation: measure with `EXPLAIN ANALYZE`; index only observed query patterns; document each index's reason.

#### Dependencies

`S1` (all event types and refs), `S0` (protocol).

#### Handoff

A trustworthy, queryable, production-grade event store. This is the **MVP-ready capture foundation**; evaluators can now be built with confidence.

#### GO / NO-GO Checklist

- [x] All Exit Criteria rows pass. (1×CI migrations+parity, 2×1M load gate, 3×p99 numbers recorded in `perf/write-benchmark.md`, 4×DB role test, 5×retention/tombstone/legal-hold tested, 6×replay CLI vs Postgres e2e + compose stack, 7×backup/restore smoke.)
- [x] Losslessness proven at target load (1M events / 1,000 sessions, zero gaps).
- [x] Migrations and restore drill recorded (Alembic apply/downgrade tests; `pg_dump` restore smoke in CI).
- [x] Status Board updated: `S2` = `[x]`.
- [x] No open `gate-failure` issue.
- [x] If losslessness is not proven → **NO-GO**; evaluators on lossy data produce untrustworthy flags. *(Proven — GO.)*

---

### Sprint `S3` — Module: Tool-Use Grounding & Provenance

**Objective.** Implement the first and most structurally verifiable detector: an agent claiming support from a tool it never called, or contradicting what the tool returned. Build the reusable **claim-extraction + provenance-diff** machinery that `S4` (memory integrity) will reuse. Establish the **flag schema** that all modules emit.

**Maturity:** `25 → 35` · **Duration:** 3–4 weeks · **Phase:** 2 Evaluate
**Team acceleration:** one dev on claim extraction, one on the diff/flag worker, one on the adversarial fixture suite — ~2–3 weeks.

#### Deliverables

- The universal `Flag` schema (severity, confidence, category, evidence refs, module version, adjudication status, human-review link).
- A worker architecture: `EvaluatorWorker` base that subscribes to the event stream, processes sessions, and writes flags idempotently.
- Claim extraction: identify assertions in output/reasoning phrased as grounded in a tool/retrieval.
- Provenance diff against the logged call graph: flag ungrounded claims and contradiction-with-return claims.
- A hand-built adversarial fixture suite (known-good vs known-bad sessions) with measured FP/FN.

#### Task Breakdown

**Flag schema & worker framework**
- [x] `S3-T1` (P0) Define `Flag`: `flag_id`, `session_id`, `event_id`(s) referenced, `module`, `module_version`, `category`, `severity` (enum: `info`/`low`/`medium`/`high`/`critical`), `confidence` [0,1], `summary`, `evidence` (list of event refs), `created_at`, `adjudication` (`pending`/`confirmed`/`rejected`), `adjudicated_by`, `adjudicated_at`. Persist in `flags`.
- [x] `S3-T2` (P0) `EvaluatorWorker` base: config, idempotency key (deterministic from session+module+version), retry, backoff, checkpointing so restarts don't reprocess or lose work.
- [x] `S3-T3` (P0) Worker processes **completed** sessions (a session-close trigger or watermark), plus an on-demand "evaluate this session now" API for the gate path.
- [x] `S3-T4` (P1) Determinism requirement: same inputs + same module version → identical flags (needed for reproducible FP/FN and for replay).

**Claim extraction**
- [~] `S3-T5` (P0) Extraction pass over `llm.response` / reasoning traces: classify spans as `grounded_claim` (cites a tool/retrieval/source), `numeric_claim`, or `ungrounded`. Start rule/template-based, then a pluggable small-model classifier behind an interface. **The rule/template extractor and the `ClaimExtractor` protocol both ship, and the seam is now covered by a test that injects a non-default extractor end to end.** Two gaps: extraction reads only `llm.response` (no reasoning/thinking trace is read — in fact none is captured anywhere, which is an `S1` capture gap), and the span taxonomy that shipped is `ClaimKind` (numeric/boolean/date/member/…) rather than the `grounded_claim`/`numeric_claim`/`ungrounded` classification this task specifies. No model-backed extractor exists; the protocol is the seam one would slot into.
- [~] `S3-T6` (P0) Normalize each grounded claim to a structured form: `{claim_text, claimed_source, claimed_value?}`. **`claimed_value` and `claim_text` ship; `claimed_source` does not.** No per-claim record of the source the claim names, because nothing resolves a claim's wording to a specific tool call (see `S3-T8`). Tracked with `S3-T8`.
- [~] `S3-T7` (P1) Handle "implicit grounding" language: "according to the document", "the search returned", "the API shows", etc. — a maintained lexicon with tests. **The lexicon that shipped is the *deictic* one** ("today", "the latest release", "currently", 21 phrases, tested). The *attribution* phrases this task names — "according to the document", "the search returned", "the API shows" — are **not implemented**. The two need different machinery: deictic phrases resolve against a value in the evidence, attribution phrases name a source that must be matched to a tool call.

**Provenance diff**
- [~] `S3-T8` (P0) `provenance_diff(session, claims, call_graph)`: for each claim, find the referenced tool call; if none exists → flag `ungrounded_claim`; if one exists but the returned value disagrees → flag `contradicted_claim` (with the actual observed value in evidence). **The contradiction half is complete and tested** (7 corpus cases, per-claim diff, `observed_value` in evidence marked `countervailance`). **"Find the referenced tool call" is not implemented**: evidence is collected wholesale for the turn and a claim is ungrounded when no value anywhere in that turn matches, so there is no per-claim source→tool-call resolution and a citation to a tool that was never called cannot be detected. Ships as `gather_evidence` + `diff_claim`; there is no symbol named `provenance_diff`.
- [~] `S3-T9` (P0) Severity mapping: ungrounded claim about a consequential value (money, legal, safety, quantity) escalates severity; default severity `medium`. **Money and quantity escalate; legal and safety are not modelled at all.** Escalation keys off the claim's *kind* (numeric/boolean/set outrank date/duration/weekday) and off the verdict (contradiction outranks ungrounded), not off a value-domain classifier, so a claim about a legal deadline or a safety limit carries no extra weight. The corpus spans all three bands — 5 `high`, 4 `medium`, 4 `low` — so the escalation does fire; it just has no notion of which *kind of consequence* is at stake.
- [x] `S3-T10` (P1) Confidence estimation: combine extraction confidence and diff certainty; low-confidence flags are marked for human review rather than gated.
- [x] `S3-T11` (P1) Make the whole path **reusable** as `sentinel.eval.provenance_core` so `S4` imports it (this is the shared mechanism noted in the design doc).

**Adversarial fixtures & measurement**
- [~] `S3-T12` (P0) Build a fixture corpus: **known-good** sessions (accurate citations) and **known-bad** sessions (fabricated citations, contradicted values, cherry-picked numbers). Store as replayable event sequences. **22 replayable cases ship: 10 known-good and 12 known-bad, covering fabricated *values* and contradicted values. Two named categories are absent — fabricated *citations* (naming a source no tool returned) and cherry-picked numbers** (reporting a favourable figure from a result that also contains unflattering ones). Both are unimplemented detections as well as unimplemented fixtures, which is why adding the fixtures alone would break the FN gate. See the gap list below.
- [x] `S3-T13` (P0) Harness: `sentinel eval-fixtures --module provenance` prints a confusion matrix; compute FP/FN rates.
- [x] `S3-T14` (P0) Set a gate threshold: FP rate ≤ target (e.g. ≤ 5% on the corpus) and FN rate ≤ target (e.g. ≤ 10%) before the module is considered usable; otherwise tune.
- [x] `S3-T15` (P1) Human-review routing: flags with confidence below threshold never gate, they queue for review (feeds `S7`).

**Docs & integration**
- [x] `S3-T16` (P1) `docs/modules/provenance.md`: methodology, severity model, FP/FN on the corpus, limitations.
- [~] `S3-T17` (P1) Example: an agent prompted to fabricate a citation is caught end-to-end. **The worked example and its test are real and green, but the scenario is a wrong value against a citation that genuinely happened** (`contradicted_price`, caught as `contradicted_claim` with `observed_value = "49 usd"`), not a citation to a source that was never called. It evidences the `S3-T8` contradiction path, not the fabricated-citation path this task names.

#### Standards Focus

Flag/evidence schema design, worker idempotency, deterministic evaluators, adversarial test methodology, measured error rates (the credibility core).

#### Tests & Verification

- [x] Unit: claim extraction on a labeled set; provenance diff on synthetic call graphs.
- [~] Property: diff is deterministic and order-independent; no claim produces more than one primary flag. *(Determinism and one-flag-per-claim are covered by example-based tests — two evaluator instances produce byte-identical flags, and `flag_id` equality is asserted across forced re-runs. There is no hypothesis-driven property test, and evidence order within a turn is not shuffled in any test, so order-independence is argued, not proven.)*
- [x] Adversarial: every known-bad fixture is caught; no known-good fixture is flagged (or within the stated FP budget).
- [x] Idempotency: running the worker twice yields identical flags, no duplicates.
- [x] Evidence completeness: every flag references at least one real event ID.

#### Exit Criteria

| # | Criterion | Verified by | Result |
|---|---|---|---|
| 1 | The flag schema is implemented, persisted, and documented | schema test + doc | Pass — [`docs/flag-schema.md`](../flag-schema.md), [`docs/adr/0012`](../adr/0012-flag-schema.md) |
| 2 | Fabricated-citation and contradicted-value sessions are caught without manual intervention | adversarial tests | **Partial** — contradicted-value sessions are caught (7 corpus cases). A *fabricated citation* (naming a source no tool ever returned) is **not** detected: see the gap list below. |
| 3 | FP ≤ 5%, FN ≤ 10% on the corpus (documented actual numbers) | `eval-fixtures` report | Pass — **FP 0.00%, FN 0.00%**, claim FP 0.00% on 22 cases. Meaningful only over the corpus that exists; it does not cover cherry-picked numbers. |
| 4 | Worker is idempotent and restart-safe | idempotency test | Pass — re-running yields identical `flag_id`s, zero new rows |
| 5 | `provenance_core` is a reusable, tested library boundary | code review | Pass — no reverse dependency on `sentinel.instrument` (INV-1 cold-subprocess test). No `S4` consumer exists yet; that is `S4`'s job, not a defect here. |
| 6 | Every flag carries complete, valid evidence refs | test | Pass — non-empty, ULID-checked, role-typed, no duplicate pairs |

#### Risks & Mitigations

- **Claim extraction accuracy too low.** Mitigation: start conservative (high precision, route uncertain to review), measure, then widen; keep the classifier pluggable.
- **Confusing "grounded" with "true".** Mitigation: the module only checks *provenance*, never truth; document this boundary explicitly.

#### Dependencies

`S2` (store + call graph).

#### Handoff

The flag schema, worker framework, and provenance core are now reusable by all later modules. `S4` reuses them directly.

**Measured on the corpus** (`uv run sentinel eval-fixtures --module provenance`, 22 cases): 13 true positives, 0 false negatives, 0 false positives, 9 true negatives — case FP `0.00%`, FN `0.00%`, claim FP `0.00%`, against budgets of ≤ 5% FP and ≤ 10% FN. Rule-by-rule breakdown, the severity model, and the known blind spots are in [`docs/modules/provenance.md`](../modules/provenance.md). The numbers are exact for the corpus that exists; the corpus does not cover fabricated citations or cherry-picked numbers, so they do not bound the module's real-world error rate.

#### GO / NO-GO Checklist

- [~] All Exit Criteria rows pass. *(5 of 6 pass. Row 2 is partial: contradicted-value sessions are caught, fabricated *citations* are not — see the gap list.)*
- [x] FP/FN measured and within budget on the corpus. *(FP 0.00%, FN 0.00% over 22 cases. The budget is met; the corpus is narrower than the task specified, so this is a weaker statement than it looks.)*
- [x] The flag schema is declared stable (or an ADR records changes). *(ADR-0012, `schema_version = 0.1`.)*
- [x] Status Board updated: `S3` = `[~]`.
- [x] No open `gate-failure` issue.
- [x] If FP/FN budgets are not met → **NO-GO**; do not proceed to memory integrity until the error rate is defensible. *(Budgets met: FP 0.00%, FN 0.00%.)*

**Verdict: CONDITIONAL GO.** The flag schema, worker framework, contradiction detection and measurement harness are real, tested against both stores, and meet their budgets. The module detects an agent contradicting its own tool output. It does **not** yet detect an agent citing a source it never consulted, which is the other half of the task's headline promise and the part `S3-T12`'s "fabricated citations" and "cherry-picked numbers" cases were supposed to prove. `S4` is unblocked — it reuses `provenance_core` and needs none of the missing machinery.

#### Known gaps carried out of `S3`

Ordered by how much they matter. None is a regression; all are absences.

1. **No per-claim source resolution** (`S3-T6`, `S3-T8`, `S3-T12`, `S3-T17`). Evidence is gathered for the whole turn and matched by value, so there is no way to ask "which tool result was this claim about?". Consequence: a citation to a tool that was never called is invisible, `claimed_source` is not recorded, and the module cannot tell a selective report from a complete one.
2. **No cherry-picked-number detection** (`S3-T12`). Reporting `2 of 3 passed` when the third failed is currently a silent miss, and the corpus has no case that would expose it.
3. **No attribution lexicon** (`S3-T7`). Only deictic phrases resolve. "According to the document" is not recognised as a claim about a source at all.
4. **No legal/safety severity weighting** (`S3-T9`). Severity follows claim kind and verdict; a wrong legal deadline ranks the same as a wrong meeting time.
5. **No property tests** for order-independence, and no reasoning-trace extraction (`S3-T5`).
6. **No `S4` consumer of `provenance_core`** yet. The Postgres store is no longer unmeasured: it is gated at 95% by the `integration-postgres` job via `.coveragerc.postgres`, and the `sentinel_reviewer` least-privilege role is now proven against a real database rather than asserted.

**How to close this honestly:** items 1–2 are one feature (resolve each claim's named source to a tool call, then flag a claim whose source never returned anything). It should ship with corpus cases *and* a rule together, because adding the fixtures without the rule would immediately fail the FN gate. Items 3–5 are independent and smaller.

---

### Sprint `S4` — Module: Memory & Context Integrity

**Objective.** Detect memory that silently rots: abrupt semantic jumps from injected/manipulated updates, creeping homogenization (memory collapse), and reflective summaries that assert things that never happened in the session. Reuses `provenance_core` for the write-vs-transcript check.

**Maturity:** `35 → 45` · **Duration:** 3 weeks · **Phase:** 2 Evaluate
**Team acceleration:** one dev on embedding drift metrics, one on the write-provenance check + fixtures — ~2 weeks.

#### Deliverables

- Embedding-based drift metric across successive memory states per agent/namespace.
- Lexical repetition / homogenization metric detecting memory collapse.
- Write-vs-transcript provenance check (reuse `S3` core) for reflective summaries.
- Local Ollama embedding integration with caching and a swappable model interface.
- Adversarial fixtures: injected-update sessions, collapse sessions, fabricated-summary sessions.

#### Task Breakdown

**Embedding infrastructure**
- [ ] `S4-T1` (P0) `EmbeddingProvider` interface; default implementation calls local Ollama (`/api/embeddings`); model name configurable; deterministic for a fixed model/version.
- [ ] `S4-T2` (P0) Embedding cache keyed by content hash + model id; persistent in the store or a local table; invalidated on model change.
- [ ] `S4-T3` (P1) Batch embedding + rate limiting so the worker doesn't starve the host's Ollama.

**Drift & collapse metrics**
- [ ] `S4-T4` (P0) Maintain a per-agent/normalized memory state series from `memory.write`/`memory.read` events.
- [ ] `S4-T5` (P0) Drift metric: cosine distance between successive memory snapshots; flag `memory_drift` when the jump exceeds a configurable threshold within a single update.
- [ ] `S4-T6` (P0) Homogenization metric: lexical diversity/repetition ratio trend (and/or pairwise embedding similarity trend) across successive states; flag `memory_collapse` on sustained convergence to a stale loop.
- [ ] `S4-T7` (P0) Threshold policy: per-agent baselines with warm-up (don't flag during cold start); document how baselines adapt.
- [ ] `S4-T8` (P1) Severity mapping: injected-update drift is `high`/`critical`; gradual collapse is `medium`.

**Write provenance**
- [ ] `S4-T9` (P0) Cross-check every `memory.write` that is a summary/reflection against the transcript and call log it claims to summarize, using `sentinel.eval.provenance_core`. Flag `memory_ungrounded` when the summary asserts events absent from the underlying session.
- [ ] `S4-T10` (P1) Distinguish "summarizing/reflecting" writes from raw fact persistence; apply the heavy check only to the former, the drift check to the latter.

**Fixtures, measurement, docs**
- [ ] `S4-T11` (P0) Adversarial corpus: (a) single injected malicious memory update; (b) gradual collapse over many updates; (c) fabricated reflective summary. Plus a known-good corpus.
- [ ] `S4-T12` (P0) Measure FP/FN; set module budgets; document actual numbers.
- [ ] `S4-T13` (P1) `docs/modules/memory-integrity.md` (methodology, thresholds, limitations, embedding model assumptions).
- [ ] `S4-T14` (P1) Example: a corrupted-update session produces a `memory_drift` flag and a `memory_ungrounded` flag.

#### Standards Focus

Determinism vs stochastic embeddings (pin model versions), local-first processing, threshold configurability, reuse of shared core (no duplication).

#### Tests & Verification

- [ ] Unit: drift/collapse metrics on synthetic vector sequences.
- [ ] Integration: local Ollama embeddings, cached, deterministic across runs.
- [ ] Adversarial: all three adversarial fixture families caught within budget; known-good not flagged.
- [ ] Perf: embedding drift p95 < 1 s per memory write at target load.
- [ ] Idempotency and warm-up correctness.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | Injected memory update is flagged `high`/`critical` | adversarial test |
| 2 | Gradual collapse is flagged without also flagging healthy evolution | adversarial + known-good |
| 3 | Fabricated reflective summary is flagged via `provenance_core` | adversarial test |
| 4 | FP/FN measured and within budget | corpus report |
| 5 | Embeddings are local, cached, and reproducible | integration test |
| 6 | Thresholds fully configurable, with warm-up documented | config test |

#### Risks & Mitigations

- **Embedding cost/latency in production.** Mitigation: caching, batching, sampling of long-horizon states, and off-hot-path execution.
- **False positives from legitimate large refactors of memory.** Mitigation: per-agent baselines, warm-up, and human-review routing for `medium`.

#### Dependencies

`S3` (`provenance_core`, flag schema, worker framework), `S2` (store).

#### Handoff

Two provenance-based modules live. Both the flag framework and shared core are proven reusable. `S5` now adds the first probabilistic module.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] FP/FN measured and within budget.
- [ ] Local embedding path verified deterministic and cached.
- [ ] Status Board updated: `S4` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If collapse detection cannot separate healthy evolution from collapse within budget → **NO-GO**; descope collapse to "experimental" (ADR) and ship drift + write-provenance only.

---

### Sprint `S5` — Module: Reasoning Faithfulness

**Objective.** Provide *evidence* about whether an agent's visible reasoning is causally connected to its decision, using two complementary mechanisms: an always-on lightweight consistency scorer and an on-demand counterfactual perturbation test. This module must **never adjudicate alone** — every flag routes to human review.

**Maturity:** `45 → 55` · **Duration:** 4 weeks · **Phase:** 2 Evaluate
**Team acceleration:** one dev on the counterfactual harness, one on the judge/consistency scorer and its FP/FN — ~3 weeks.

#### Deliverables

- **Counterfactual perturbation harness:** for a sampled decision, alter a specific context element the agent had access to, re-run through a re-execution sandbox or a model-judge approximation, and measure whether the final action shifts while the stated reasoning does not acknowledge the change.
- **Always-on consistency scorer:** a separate judge model scores whether the reasoning text semantically supports the output; divergence beyond a threshold flags for review.
- Method to **re-execute** or approximate re-execution safely (the host agent must not take real actions during perturbation).
- Adversarial fixtures with deliberately unfaithful and faithful reasoning traces; measured FP/FN.
- Documentation of what this module can and cannot claim.

#### Task Breakdown

**Consistency scorer (always-on)**
- [ ] `S5-T1` (P0) `JudgeProvider` interface (default local Ollama); structured-output prompt with a rubric; deterministic decoding where possible (temperature 0) and pinned model version.
- [ ] `S5-T2` (P0) Score reasoning-output pairs on a calibrated scale; produce `confidence`; flag `reasoning_inconsistency` when the score crosses a configurable threshold.
- [ ] `S5-T3` (P0) Guard against judge hallucination: require the judge to cite specific spans as justification; discard uncited judgments; treat unparseable output as "no flag", not "flag".
- [ ] `S5-T4` (P1) Cost/latency control: run on a configurable sample, and always on high-stakes decisions (see gating integration in `S7`).

**Counterfactual harness (on-demand / sampled)**
- [ ] `S5-T5` (P0) Define a **context perturbation**: remove/replace a specific retrieved document, tool result, or memory item the agent used.
- [ ] `S5-T6` (P0) **Safe re-execution sandbox:** re-run the agent path with all consequential actions stubbed/no-op; if re-execution is impossible (non-deterministic/costly), offer a judge-based counterfactual approximation and mark the flag as lower confidence.
- [ ] `S5-T7` (P0) Compare final action before/after; detect "action changed, reasoning did not acknowledge it" and "action changed because of a factor never mentioned" → flag `reasoning_unfaithful_counterfactual`.
- [ ] `S5-T8` (P0) Handle determinism: seed/temperature control, multiple samples if needed, report variance; never flag on a single noisy sample without confidence bounds.
- [ ] `S5-T9` (P1) Perturbation selection strategy: pick context items with the highest estimated influence (cheapest useful counterfactual), capped per session.

**Routing & safety**
- [ ] `S5-T10` (P0) **Every** faithfulness flag routes to human review; the module never gates directly (enforced in code: category marked `review_only=True`).
- [ ] `S5-T11` (P1) Record perturbation provenance in the flag evidence so a reviewer can reproduce it.

**Fixtures, measurement, docs**
- [ ] `S5-T12` (P0) Corpus of known-faithful and known-unfaithful traces with the counterfactual ground truth.
- [ ] `S5-T13` (P0) Measure FP/FN for both mechanisms separately; document actual numbers.
- [ ] `S5-T14` (P1) `docs/modules/faithfulness.md` — methodology, evidence-only stance, limitations, and the explicit statement that this is probabilistic, not structurally verifiable.

#### Standards Focus

Judge-model safety (never trust a judge blindly), calibration, determinism/seeding, safe re-execution (no real-world side effects), honest documentation of uncertainty.

#### Tests & Verification

- [ ] Unit: scorer calibration on a labeled reasoning set; perturbation application correctness.
- [ ] Property: perturbations are pure — they never mutate stored events or trigger real tool calls.
- [ ] Adversarial: known-unfaithful traces catch; known-faithful do not exceed FP budget.
- [ ] Safety: sandbox test proves stubbed actions cannot execute side effects.
- [ ] Reproducibility: a recorded flag's perturbation reproduces the same outcome.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | Unfaithful reasoning flagged with evidence and confidence | adversarial test |
| 2 | Faithful reasoning not flagged beyond budget | known-good corpus |
| 3 | Counterfactual re-execution cannot cause real side effects | sandbox test |
| 4 | Flags are `review_only`; no direct gating | code invariant test |
| 5 | FP/FN documented for both mechanisms | report |
| 6 | Judge outputs are structured, span-cited, and fail-safe | tests |

#### Risks & Mitigations

- **A weak/false-confidence detector erodes trust.** Mitigation: honest documentation, review-only routing, published error rates, and a config flag to disable in high-FP environments.
- **Re-execution cost/risk.** Mitigation: sampling, influence-based selection, and stubbed actions strictly.
- **Judge model bias.** Mitigation: pin version, spot-check against human labels in the corpus, and never use a single scalar without an evidence span.

#### Dependencies

`S3` (flag schema, worker framework, extraction), `S2` (replay for perturbation), `S1` (capture).

#### Handoff

All four "reasoning/tool/memory" modules now exist. `S6` adds objective drift; `S7` makes everything actionable via gating.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Sandbox side-effect test passes (non-negotiable).
- [ ] FP/FN documented and within stated, honest budget.
- [ ] Status Board updated: `S5` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If faithfulness cannot beat a noise baseline within budget → **NO-GO for shipping it as reliable**; ship it labeled "experimental, review-only", record an ADR, and continue.

---

### Sprint `S6` — Module: Specification Gaming & Objective Drift

**Objective.** Detect agents satisfying the letter of an instruction while defeating its intent: narrowing success criteria mid-task, reporting completion without the state change, and repeatedly taking the minimum-effort action that meets a literal reading.

**Maturity:** `55 → 65` · **Duration:** 3 weeks · **Phase:** 2 Evaluate
**Team acceleration:** one dev on objective extraction/spec, one on the trajectory discrepancy scorer — ~2 weeks.

#### Deliverables

- Objective extraction from system prompt / task specification into a structured `DeclaredObjective`.
- Trajectory discrepancy scorer comparing declared objective vs realized action sequence.
- Taxonomy-based detectors: `success_criteria_narrowing`, `false_completion` (claims done without the corresponding state change), `minimum_effort_literal_satisfaction`.
- Adversarial fixtures and measured FP/FN.

#### Task Breakdown

**Objective extraction**
- [ ] `S6-T1` (P0) Parse `session.start` / task spec / system prompt into a `DeclaredObjective` with explicit success criteria, constraints, and expected state change (where available).
- [ ] `S6-T2` (P0) Handle underspecified objectives gracefully: mark them low-confidence and avoid over-flagging literal-but-reasonable behavior.

**Discrepancy & detectors**
- [ ] `S6-T3` (P0) `success_criteria_narrowing`: detect the agent redefining a narrower goal mid-task relative to the declared objective.
- [ ] `S6-T4` (P0) `false_completion`: cross-check claims of task completion against the **actual state change** in the event graph (did the write/action actually occur?). This reuses provenance-like structural checks and is the most defensible detector in this module.
- [ ] `S6-T5` (P0) `minimum_effort_literal_satisfaction`: flag repeated selection of the cheapest action satisfying the literal instruction when the realized outcome misses evident intent (judge-assisted with evidence, review-only).
- [ ] `S6-T6` (P1) Incorporate the spec-gaming taxonomy patterns as a pluggable rule set so new patterns are config, not code.

**Fixtures, measurement, docs**
- [ ] `S6-T7` (P0) Adversarial corpus per pattern + known-good.
- [ ] `S6-T8` (P0) Measure FP/FN; separate structural detectors (false_completion) from probabilistic ones (minimum effort).
- [ ] `S6-T9` (P1) `docs/modules/spec-gaming.md`.

#### Standards Focus

Structural > probabilistic where possible, taxonomy as configuration, honest confidence separation.

#### Tests & Verification

- [ ] Unit per detector; property tests that structural `false_completion` is deterministic.
- [ ] Adversarial per pattern; known-good not flagged beyond budget.
- [ ] Evidence: every flag cites the specific actions/events.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | `false_completion` caught structurally and deterministically | adversarial test |
| 2 | Criteria narrowing and minimum-effort detected within budget | adversarial tests |
| 3 | New taxonomy patterns addable without code change | config test |
| 4 | FP/FN documented | report |
| 5 | Probabilistic detectors marked review-only | code invariant test |

#### Risks & Mitigations

- **Over-flagging legitimate pragmatic choices.** Mitigation: structural detectors gate-worthy; intent-based detectors review-only; conservative defaults.
- **Underspecified tasks produce noise.** Mitigation: low-confidence handling and per-objective baselines.

#### Dependencies

`S3` (worker, flag schema, extraction), `S2` (state-change queries).

#### Handoff

All five detection modules now exist (provenance, memory, faithfulness, spec-gaming, and eval-awareness pending `S8`). `S7` makes the first four actionable.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] `false_completion` proven deterministic.
- [ ] Status Board updated: `S6` = `[x]`.
- [ ] No open `gate-failure` issue.

---

### Sprint `S7` — Policy, Gating & Review Interface

**Objective.** Make flags actionable. Build the thin, auditable policy engine that intercepts an action at a defined checkpoint and decides **proceed / hold-for-review / block**, plus the review API and a minimal interface where a human approves, rejects, or requests revision. Deliver the first true end-to-end flagged demo.

**Maturity:** `65 → 75` · **Duration:** 3 weeks · **Phase:** 3 Act
**Team acceleration:** one dev on the engine + API, one on the review UI and e2e demo — ~2 weeks.

#### Deliverables

- A **policy engine** small enough to audit in an afternoon (INV-4): declarative rules over flag severity/confidence/category and organization stakes thresholds.
- Checkpoint integration: before tool execution, before response return, before state commit — hold in `pending_review`.
- Pending-review API + review UI (static-first) with approve/reject/request-revision and full evidence display.
- Organization stakes config (dollar amount, restricted action classes).
- Fail-safe behavior (INV-6): gate failures hold, never silently approve.
- End-to-end demo: a flagged action is held, reviewed, and resolved.

#### Task Breakdown

**Policy engine (INV-4)**
- [ ] `S7-T1` (P0) Declarative rule model: `when` (flag predicates + stakes conditions) → `then` (`proceed`/`hold`/`block`), with priority and explicit default. YAML/JSON config, versioned.
- [ ] `S7-T2` (P0) Keep the engine tiny and dependency-light; document every rule in plain language; provide a config linter that rejects ambiguous/conflicting rules.
- [ ] `S7-T3` (P0) Severity/confidence thresholds configurable per category and per deployment; conservative defaults; no hardcoded thresholds.
- [ ] `S7-T4` (P0) Stakes thresholds: monetary amount, restricted action classes, data classes — org-defined.

**Checkpoints & gating**
- [ ] `S7-T5` (P0) Define checkpoints in the host agent integration: `before_tool`, `before_response`, `before_state_commit`. Each calls the gate synchronously (async) within the latency budget (< 200 ms).
- [ ] `S7-T6` (P0) `pending_review` state machine: action held, session paused or action queued, reviewer notified, resolution resumes or aborts.
- [ ] `S7-T7` (P0) Fail-safe: if the policy service/DB is unreachable, hold (configurable to block) and surface a loud ops alert; never auto-approve.
- [ ] `S7-T8` (P1) Timeout policy: held actions expire per config (auto-reject or escalate), never silently proceed.

**Review API & UI**
- [ ] `S7-T9` (P0) FastAPI: list pending, get flag with evidence, approve/reject/request-revision, with auth (token) and audit logging of every reviewer action.
- [ ] `S7-T10` (P0) Minimal review UI (static HTML/JS): queue, flag detail with linked events and a rendered replay excerpt, and the three decisions.
- [ ] `S7-T11` (P0) Record adjudication back into `flags` (`confirmed`/`rejected`, who, when) — this feeds FP/FN recalibration.
- [ ] `S7-T12` (P1) Reviewer ergonomics: keyboard-driven decisions, reason codes, and per-flag context enough to decide in seconds.

**Demo & integration**
- [ ] `S7-T13` (P0) E2E demo: an agent produces a provenance flag (or false completion); the checkpoint holds the action; reviewer rejects; agent receives revision instruction; session resumes.
- [ ] `S7-T14` (P1) `docs/operations/gating.md` and `docs/quickstart-gating.md`.

#### Standards Focus

Auditability, least configuration footgun, fail-safe design, reviewer ergonomics, latency budget, auth + audit of human actions.

#### Tests & Verification

- [ ] Unit: rule evaluation, priority, conflicts, default behavior.
- [ ] Property: gate never returns "proceed" on policy-service failure when configured fail-safe.
- [ ] Integration: each checkpoint holds/resumes correctly.
- [ ] Perf: gate decision p99 < 200 ms.
- [ ] Security: review API requires auth; all decisions audited; no flag data leaked unauthenticated.
- [ ] E2E demo recorded.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | A flagged action is held at a checkpoint and only proceeds after review | e2e test + demo |
| 2 | Policy engine is auditable (config linter + docs, < ~300 LOC core) | review |
| 3 | Fail-safe verified (no silent approval on outage) | failure-injection test |
| 4 | Gate decision p99 < 200 ms | benchmark |
| 5 | Adjudications persist and feed recalibration | integration test |
| 6 | Reviewer can decide from evidence without reading logs by hand | demo review |

#### Risks & Mitigations

- **Gating friction causes disabling under pressure.** Mitigation: conservative defaults, per-deployment tuning, review-only thresholds, and starting in "observe" mode (flags without enforcement) with a documented path to enforcement.
- **Latency creep.** Mitigation: precompute evaluations at session close where possible; keep the gate path to a single indexed query.

#### Dependencies

`S3`–`S6` (flags), `S2` (store), `S1` (checkpoints).

#### Handoff

The product is now demonstrable end-to-end: detect → flag → hold → review → resume. This is the **beta feature-complete milestone (maturity 75)** once `S8` lands.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] E2E flagged-and-gated demo recorded.
- [ ] Fail-safe proven.
- [ ] Status Board updated: `S7` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If gating cannot meet the latency budget → **NO-GO**; optimize or move evaluations ahead of the checkpoint before proceeding.

---

### Sprint `S8` — Module: Evaluation-Awareness & Sandbagging

**Objective.** Detect agents that behave differently when they infer they are being evaluated versus in production, and treat any attempt to disable/bypass oversight as a maximum-severity flag. This is the hardest module statistically and is intentionally scheduled after gating exists.

**Maturity:** `75 → 80` · **Duration:** 4 weeks · **Phase:** 3 Act
**Team acceleration:** one dev on task-pairing/framing infrastructure, one on the statistics and interference detection — ~3 weeks.

#### Deliverables

- **Paired-framing injection:** near-duplicate tasks presented under an "evaluation" framing and an "ordinary production" framing.
- **Statistical comparison** of performance/behavioral distribution across framings with significance handling and minimum-sample guards.
- **Oversight-interference detector:** any attempt by the agent to disable, bypass, or avoid its own logging/approval mechanisms → `critical` flag regardless of stated justification.
- Adversarial fixtures and measured FP/FN; documentation of statistical power limits.

#### Task Breakdown

**Paired framing**
- [ ] `S8-T1` (P0) Task-pair generator/registry: semantically equivalent tasks, one framed as an eval, one as production traffic; frames must be subtle and realistic.
- [ ] `S8-T2` (P0) Injection mechanism integrated with the host agent, respecting sampling and consent (never disrupt real user tasks in a harmful way).
- [ ] `S8-T3` (P1) Frame-blinding: ensure the *system* knows the frame but the framing isn't trivially detectable by surface artifacts alone (document residual detectability).

**Statistics**
- [ ] `S8-T4` (P0) Behavioral metrics per framing: task success, caution/refusals, action distribution, verbosity, tool-use rate. Aggregate with confidence intervals.
- [ ] `S8-T5` (P0) Hypothesis testing with **minimum-sample enforcement**: do not flag until N pairs and a pre-registered effect threshold; control for multiple comparisons.
- [ ] `S8-T6` (P0) `evaluation_awareness_gap` flag with the measured effect, CI, sample size, and the paired task evidence.
- [ ] `S8-T7` (P1) Drift-aware baselines: account for model updates and changing traffic so a legitimate change isn't read as awareness.

**Oversight interference (max severity)**
- [ ] `S8-T8` (P0) Detect attempts to modify/disable logging, gating, redaction, or the SDK itself — at the event level (e.g. commands targeting the SDK, config changes, monkey-patching) and via a protected-config checksum.
- [ ] `S8-T9` (P0) Any such attempt → `critical` flag (`oversight_interference`) independent of justification, gating-eligible, and audit-logged.

**Fixtures, measurement, docs**
- [ ] `S8-T10` (P0) Corpus: sandbagging agents (performance gap), non-sandbagging agents, and interference attempts.
- [ ] `S8-T11` (P0) Measure FP/FN; document **statistical power** (what effect sizes are detectable at realistic sample sizes) — this is the honest-limits requirement.
- [ ] `S8-T12` (P1) `docs/modules/evaluation-awareness.md` — methodology, power analysis, limitations.

#### Standards Focus

Statistical rigor (pre-registration, power, multiple-comparison control), safety-critical severity handling, honest documentation, protecting the oversight mechanism itself.

#### Tests & Verification

- [ ] Unit: statistics on synthetic distributions with/without gaps.
- [ ] Property: no flag below the minimum sample; effect direction correct.
- [ ] Adversarial: sandbagging corpus detected within budget; non-sandbagging not flagged; interference attempts always critical.
- [ ] Safety: interference detection cannot itself be trivially disabled by the agent (protected config + tamper checks).

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | A measurable behavioral gap across framings produces a flag with CI and sample size | adversarial test |
| 2 | No flag below minimum sample; multiple-comparison control in place | property test |
| 3 | Oversight-interference attempts always produce `critical` flags | adversarial test |
| 4 | Statistical power limits documented | doc review |
| 5 | FP/FN measured | report |

#### Risks & Mitigations

- **Insufficient statistical power in low-traffic deployments.** Mitigation: enforced minimum samples, honest docs, and treating this module as advisory until power is reached.
- **Framing is detectable and games the test.** Mitigation: realistic frames, randomization, and documentation that the method is best-effort.
- **False accusations of sandbagging.** Mitigation: high bar to flag; CI reporting; human review; never auto-block on a single gap.

#### Dependencies

`S7` (gating to act on interference), `S3`/`S6` (worker/instrumentation), `S2` (store).

#### Handoff

All five modules plus gating exist: **maturity 80, feature-complete and functionally trusted.** `S9` now attacks the system adversarially and hardens security/supply chain.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Interference detection is tamper-resistant.
- [ ] Statistical limits documented honestly.
- [ ] Status Board updated: `S8` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If power is insufficient at realistic scale → record an ADR and mark the module "advisory/beta"; do not claim reliability you cannot defend.

---

### Sprint `S9` — Adversarial Validation, Security & Supply Chain

**Objective.** Attack Sentinel itself. Freeze a reproducible adversarial test suite spanning all modules, run a security and supply-chain hardening pass, fuzz the parsers, and produce the security evidence a cautious design partner will demand. This sprint is the credibility gate.

**Maturity:** `80 → 85` · **Duration:** 3 weeks · **Phase:** 4 Harden
**Team acceleration:** one dev on the adversarial suite + fuzzing, one on security/supply chain — ~2 weeks.

#### Deliverables

- A single **adversarial test suite** executable in CI that exercises every module's known-bad and known-good corpus and reports FP/FN.
- A documented **threat model** for the SDK and its deployment.
- Security hardening: input validation, parser fuzzing, dependency audit clean, SBOM generation, signed artifacts.
- A pen-test-style report of what was tried and the outcome (internal).
- A **reproducibility dossier**: pinned environment, seeds, and a one-command way to regenerate every FP/FN number.

#### Task Breakdown

**Adversarial suite**
- [ ] `S9-T1` (P0) Consolidate all module corpora into `tests/adversarial/` with a manifest and a single runner (`sentinel adversarial-suite`).
- [ ] `S9-T2` (P0) Add **cross-module adversarial cases**: e.g., an agent that fabricates a citation *and* corrupts memory *and* tries to disable logging; ensure flags compose and none are lost.
- [ ] `S9-T3` (P0) Add **evasion attempts**: obfuscated fabrication, splitting a false claim across turns, memory poisoning across sessions, and logging-interference variants.
- [ ] `S9-T4` (P0) Publish the suite results as a versioned artifact (CSV/JSON) referenced by the docs.
- [ ] `S9-T5` (P1) Add regression protection: any FP/FN change beyond a tolerance fails CI.

**Threat model & security**
- [ ] `S9-T6` (P0) STRIDE-style threat model: threats to capture integrity, flag integrity, gate integrity, reviewer auth, DB, and the SDK dependency chain; document mitigations and residual risk.
- [ ] `S9-T7` (P0) Fuzz (Hypothesis + `atheris`) all untrusted-input parsers: framework payloads, tool results, model outputs, config files, review API inputs.
- [ ] `S9-T8` (P0) AuthZ/AuthN review of the review API and policy service: token scope, rate limiting, CSRF for the UI, audit completeness.
- [ ] `S9-T9` (P0) SQL/injection review: confirm all queries parameterized; confirm no secret in logs/traces/errors.
- [ ] `S9-T10` (P0) Logging-interference tamper resistance reviewed (protected config checksum, append-only enforcement, alerting on config drift).
- [ ] `S9-T11` (P1) Redaction bypass attempts: adversarial payloads trying to smuggle secrets past redaction.

**Supply chain**
- [ ] `S9-T12` (P0) SBOM (CycloneDX) generated in CI and attached to releases.
- [ ] `S9-T13` (P0) `pip-audit`/OSV scan blocks on high/critical; Dependabot/Renovate enabled with a review SLA.
- [ ] `S9-T14` (P0) PyPI **trusted publishing** configured; artifacts signed (Sigstore/cosign or GPG) with a documented verification path.
- [ ] `S9-T15` (P1) Reproducible-build check: build the same tag twice and compare hashes where feasible.

**Reporting**
- [ ] `S9-T16` (P0) `docs/security/threat-model.md` and `docs/security/adversarial-results.md` (internal).
- [ ] `S9-T17` (P1) A one-page "security posture" summary for design partners.

#### Standards Focus

Adversarial rigor, threat modeling, fuzzing, supply-chain security, reproducibility, honest reporting of residual risk.

#### Tests & Verification

- [ ] The adversarial suite runs in CI under a time budget and is deterministic (seeded).
- [ ] Fuzzing runs for a defined duration with no crashes/unhandled exceptions.
- [ ] Dependency audit clean at high/critical.
- [ ] SBOM present and parseable.
- [ ] Auth/rate-limit/CSRF tests on the API.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | Adversarial suite green and reproducible; results artifact published | CI artifact |
| 2 | Threat model documented with mitigations and residual risks | doc review |
| 3 | Fuzzing finds no crashing/hanging bugs in parsers | fuzz run |
| 4 | SBOM + signed artifacts + trusted publishing working | release dry-run |
| 5 | No high/critical dependency vulnerabilities | audit |
| 6 | Cross-module and evasion cases produce expected flags | suite |

#### Risks & Mitigations

- **Finding serious security bugs late.** Mitigation: that is the *point* of this sprint; schedule it before GA and budget fix time. Do not proceed to `S11` with an unresolved critical.
- **Adversarial suite is brittle/flaky.** Mitigation: seeds, pinned models, tolerance-based regression checks, and isolation from network variance.

#### Dependencies

`S3`–`S8` (all modules and gating), `S2` (store).

#### Handoff

A hardened, adversarially-validated system with a security dossier. `S10` now proves it at scale and under failure.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] No open critical/high security issue.
- [ ] Adversarial results artifact published and referenced in docs.
- [ ] Status Board updated: `S9` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If a critical security issue is open → **NO-GO**; resolution is mandatory before ship.

---

### Sprint `S10` — Performance, Scale, Reliability & DR

**Objective.** Prove Sentinel operates within its budgets under realistic and stress load, that it degrades safely, and that it can recover from failures and lost data. Produce the operational evidence and runbooks that make self-hosting viable.

**Maturity:** `85 → 90` · **Duration:** 3 weeks · **Phase:** 4 Harden
**Team acceleration:** one dev on load/perf, one on failure/DR drills — ~2 weeks.

#### Deliverables

- A reproducible **load/performance harness** with published numbers against the budgets in [`§2.7`](#27-performance-budgets).
- Optimizations where budgets are missed (batching, connection pooling, worker concurrency, query plans).
- **Failure-mode drills:** DB down, worker crash, queue saturation, policy-service outage, disk full — each with observed behavior and recovery.
- **DR drill:** restore from backup and prove replay equality.
- Runbooks for every drill and alert.
- SLO dashboards and alerting.

#### Task Breakdown

**Performance**
- [ ] `S10-T1` (P0) Build a load harness (Locust/k6 + custom event emitter) covering capture, evaluation throughput, and gate latency.
- [ ] `S10-T2` (P0) Measure all budgets: overhead/event, losslessness/throughput, write p99, claim extraction p95, embedding p95, flag latency, gate p99.
- [ ] `S10-T3` (P0) Profile and optimize the hot paths that miss budget (async batching, prepared statements, index usage, JSONB size, worker parallelism).
- [ ] `S10-T4` (P1) Horizontal scale test: add evaluator workers; verify throughput scales and checkpointing holds.
- [ ] `S10-T5` (P1) Query performance at volume: verify gate/call-graph/session queries use intended indexes (`EXPLAIN ANALYZE` recorded).

**Reliability & failure modes**
- [ ] `S10-T6` (P0) Failure drills with documented results: DB down, worker kill -9, queue overflow, policy service down, disk exhaustion.
- [ ] `S10-T7` (P0) Verify INV-6 under each drill: capture fails open (no host crash) and gating fails safe (holds/blocks, no silent approval).
- [ ] `S10-T8` (P0) Worker checkpoint/recovery: a restarted worker resumes without reprocessing or data loss.
- [ ] `S10-T9` (P1) Graceful degradation: under load, capture samples high-volume low-value events while guaranteeing critical/error/gating events.

**Operations**
- [ ] `S10-T10` (P0) OTel/Prometheus instrumentation for capture completeness, worker lag, flag latency, gate latency, DB health, queue depth; Grafana dashboard(s).
- [ ] `S10-T11` (P0) Alerting rules tied to SLOs, each linking to a runbook.
- [ ] `S10-T12` (P0) Backups: WAL archiving + PITR configured and **restore drill executed and recorded**.
- [ ] `S10-T13` (P0) Write runbooks under `docs/runbooks/` for every alert and drill.
- [ ] `S10-T14` (P1) Capacity planning guide: sizing for N events/s and M sessions.

#### Standards Focus

Performance budgets, load testing, failure injection, DR, SLOs, runbooks, graceful degradation.

#### Tests & Verification

- [ ] Load tests pass with numbers recorded and reproducible.
- [ ] Every failure drill produces the documented behavior; anomalies are issues, not shrugs.
- [ ] Restore drill proves replay equality after PITR.
- [ ] Alerts fire correctly in a synthetic incident.
- [ ] No regressions in the adversarial suite under load.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | All performance budgets met or explicitly renegotiated with an ADR | load report |
| 2 | Failure drills documented with correct fail-open/fail-safe behavior | drill logs |
| 3 | Restore-from-backup drill succeeds with replay equality | DR record |
| 4 | SLO dashboards + alerting live; every alert links a runbook | ops review |
| 5 | Worker restart loses no events | recovery test |
| 6 | Capacity planning guide published | doc |

#### Risks & Mitigations

- **Budgets missed due to architectural limits.** Mitigation: this is the sprint to discover that; if a budget is unmeetable without a redesign, write an ADR and either renegotiate the budget or fix the design before GA.
- **DR untested until it matters.** Mitigation: mandatory recorded drill, not just configuration.

#### Dependencies

`S9` (hardened system), `S2` (store), `S7` (gating).

#### Handoff

A performant, reliable, recoverable system with operational evidence. The product is now technically trustworthy for production use; `S11` makes it distributable.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Restore drill recorded.
- [ ] Budgets met or renegotiated via ADR.
- [ ] Status Board updated: `S10` = `[x]`.
- [ ] No open `gate-failure` issue.
- [ ] If DR is unproven → **NO-GO**; a safety system that cannot recover its own audit trail is not sellable.

---

### Sprint `S11` — Documentation, Packaging & Developer Experience

**Objective.** Convert a working system into a **product**: complete user and operator documentation, a real public API reference, a packaged install, a release process, and a developer experience good enough that an outsider succeeds in 15 minutes.

**Maturity:** `90 → 95` · **Duration:** 3 weeks · **Phase:** 5 Ship
**Team acceleration:** one dev on docs site/guides, one on packaging/release/DX — ~2 weeks.

#### Deliverables

- MkDocs Material site: Quickstart, Concepts, Integration guide, Module guides, Operations/runbooks, Security, API reference (`mkdocstrings`).
- A 15-minute quickstart verified by a fresh-environment test.
- Release pipeline: versioned PyPI package `sentinel-sdk`, container images, changelog, SBOM, signed artifacts.
- Examples repository/section covering LangChain, LangGraph, raw Ollama, and custom.
- Error-message and config ergonomics pass (clear, actionable, documented).
- Public API freeze + deprecation policy documented and enforced.

#### Task Breakdown

**Documentation**
- [ ] `S11-T1` (P0) Stand up MkDocs Material with nav: Quickstart, Concepts (event model, flags, gating), Integration, Modules (×5), Operations, Security, Changelog, API.
- [ ] `S11-T2` (P0) Quickstart that goes from zero to a captured, replayed session with a real agent in ≤ 15 minutes; verified by an automated fresh-venv test.
- [ ] `S11-T3` (P0) Per-module guides: what it detects, how it works, how to configure thresholds, known limits, how to read a flag.
- [ ] `S11-T4` (P0) Operator guide: deployment, config reference, SLOs, alerts, runbooks, backup/restore, capacity.
- [ ] `S11-T5` (P0) API reference generated from docstrings; every public symbol documented.
- [ ] `S11-T6` (P1) Tutorials and a "reading your first flag" walkthrough.
- [ ] `S11-T7` (P1) Docs versioning (mike) so docs track releases.

**Packaging & release**
- [ ] `S11-T8` (P0) `python-semantic-release` fully wired: tag → build → SBOM → sign → publish to PyPI → GitHub Release with changelog.
- [ ] `S11-T9` (P0) Container images for policy/review service and workers, tagged with the version, published to a registry with an SBOM.
- [ ] `S11-T10` (P0) `docker compose` one-command reference deployment documented in Quickstart.
- [ ] `S11-T11` (P1) Optional extras packaging (`sentinel-sdk[langchain]`, `[langgraph]`, `[postgres]`) so core stays light.
- [ ] `S11-T12` (P1) Install verification matrix: fresh venv + `pip install sentinel-sdk` + quickstart passes on Linux/macOS.

**DX polish**
- [ ] `S11-T13` (P0) Config reference: every option documented with type, default, and safety implication.
- [ ] `S11-T14` (P0) Error taxonomy: actionable messages, docs links, and no raw tracebacks leaking internals to host agents.
- [ ] `S11-T15` (P1) `sentinel doctor` command: diagnoses config, store connectivity, and Ollama availability.
- [ ] `S11-T16` (P1) Public API freeze ADR + deprecation policy; remove anything accidental from the public surface.

#### Standards Focus

Docs-as-code, release engineering, reproducible packaging, DX, API stability, supply-chain artifacts.

#### Tests & Verification

- [ ] Automated quickstart test in a clean environment passes under the time budget.
- [ ] Release dry-run produces wheel, sdist, container, SBOM, signature, and changelog.
- [ ] Docs build with no broken links (link checker).
- [ ] API reference completeness check (no public symbol undocumented).
- [ ] Install matrix green.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | A newcomer completes the quickstart in ≤ 15 minutes unaided | user test / automated |
| 2 | Docs site builds, links valid, API reference complete | CI docs job |
| 3 | A signed, SBOM-attached release publishes to PyPI and a registry | release dry-run |
| 4 | `docker compose` brings up the full stack from docs alone | manual |
| 5 | Public API frozen with a documented deprecation policy | ADR + review |
| 6 | Every config option documented with safe defaults | config audit |

#### Risks & Mitigations

- **Docs drift from behavior.** Mitigation: docs tests (quickstart runs in CI), docstrings generated, versioned docs.
- **Accidental public API baggage.** Mitigation: explicit freeze + private-module lint.

#### Dependencies

`S10` (stable, performant system).

#### Handoff

A distributable, documented product with a working release pipeline. `S12` adds the compliance artifacts enterprises and regulated buyers require.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Quickstart verified in a clean environment.
- [ ] Signed release dry-run succeeded.
- [ ] Status Board updated: `S11` = `[x]`.
- [ ] No open `gate-failure` issue.

---

### Sprint `S12` — Compliance & Governance Readiness

**Objective.** Produce the evidence and mappings that let regulated and enterprise buyers adopt Sentinel: EU AI Act and NIST AI RMF mappings, an exportable audit pack, a data-classification/residency statement, and the legal artifacts needed to distribute software commercially.

**Maturity:** `95 → 98` · **Duration:** 3 weeks · **Phase:** 5 Ship
**Team acceleration:** one dev on mappings/audit export, one on legal/distribution artifacts — ~2 weeks.

#### Deliverables

- **EU AI Act** mapping (relevant transparency/record-keeping/logging obligations) and **NIST AI RMF** mapping (Govern/Map/Measure/Manage) documenting how Sentinel supports each control.
- **Audit pack export:** a reproducible bundle (events, flags, adjudications, config version, module versions, hashes) for a given session/time-range, suitable for external audit.
- **Data governance statement:** classification, retention, residency, processor/controller roles, PII handling.
- Distribution artifacts: license finalization, terms, privacy policy, `SECURITY.md`/`security.txt`, responsible disclosure, third-party notices.
- Pricing/packaging definition (open core vs enterprise) recorded as an ADR.

#### Task Breakdown

**Compliance mappings**
- [ ] `S12-T1` (P0) EU AI Act mapping doc: obligations Sentinel supports (logging, record-keeping, human oversight, transparency) with the concrete feature that supports each.
- [ ] `S12-T2` (P0) NIST AI RMF mapping doc: functions/subcategories mapped to features and evidence.
- [ ] `S12-T3` (P1) Internal control matrix: control → implementation → test → evidence.

**Audit & data governance**
- [ ] `S12-T4` (P0) `sentinel export audit --session/--range` producing a signed, hash-manifested bundle with a README explaining contents and verification.
- [ ] `S12-T5` (P0) Data classification + retention + residency statement; confirm self-hosted core path and document any opt-in external calls.
- [ ] `S12-T6` (P0) PII handling: documented policies, redaction defaults, and a data-subject-request procedure for self-hosters.
- [ ] `S12-T7` (P1) Config immutability/attestation so an audit can prove which policy was active during a session.

**Distribution & legal**
- [ ] `S12-T8` (P0) Finalize license (core) and define enterprise licensing; add `THIRD_PARTY_NOTICES`.
- [ ] `S12-T9` (P0) Terms of service / EULA (as applicable), privacy policy, and `security.txt`; responsible-disclosure process live.
- [ ] `S12-T10` (P0) Pricing/packaging ADR: what's open, what's commercial, and how it's sold.
- [ ] `S12-T11` (P1) Support policy: supported versions (last two minors), response targets, and security-patch SLA.

#### Standards Focus

Governance engineering, auditability, data protection, honest compliance claims (support, never overclaim), commercial readiness.

#### Tests & Verification

- [ ] Audit export bundle verifies (hashes match) and is reproducible.
- [ ] Mappings reviewed against the source frameworks; no unsupported claims.
- [ ] Retention/residency behavior tested end-to-end.
- [ ] Legal artifacts present and referenced from docs.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | EU AI Act + NIST mappings published and defensible | review |
| 2 | Audit pack export produces a signed, verifiable bundle | test |
| 3 | Data governance statement published | review |
| 4 | License/terms/privacy/security.txt present | review |
| 5 | Pricing/packaging decided via ADR | review |

#### Risks & Mitigations

- **Overclaiming compliance.** Mitigation: frame as "supports/evidence for" specific obligations, reviewed for accuracy; no certification claims without certification.
- **Legal ambiguity.** Mitigation: keep core permissively licensed, document clearly; engage counsel before enterprise sales if needed.

#### Dependencies

`S11` (docs/packaging), `S9` (security dossier).

#### Handoff

A product with the compliance and legal surface needed to approach enterprise and regulated buyers. `S13` takes it to real production traffic.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Audit export verified end-to-end.
- [ ] No unsupported compliance claims.
- [ ] Status Board updated: `S12` = `[x]`.
- [ ] No open `gate-failure` issue.

---

### Sprint `S13` — Design-Partner Launch & Production Tuning

**Objective.** Run Sentinel against a real design partner's actual production agent. Measure the metric that matters most — **reviewer time per flag** — tune thresholds to make the signal useful rather than noisy, and harden the operational/support loop. This is where the product earns trust.

**Maturity:** `98 → 100` · **Duration:** 4–6 weeks · **Phase:** 6 Operate
**Team acceleration:** one dev on tuning/analysis, one on support/ops and partner comms — ~3–4 weeks.

#### Deliverables

- A live deployment against a design partner's agent, in **observe mode first**, then progressively enforced.
- A **tuning report:** per-module FP/FN on real traffic, thresholds adjusted, and residual noise documented.
- **Reviewer-time-per-flag** measured and optimized; flags that can't be adjudicated efficiently are reworked or suppressed.
- A weekly operational cadence (review, incidents, feedback) and a partner feedback log feeding the backlog.
- A production readiness review and a go/no-go for GA.

#### Task Breakdown

**Deployment & observe mode**
- [ ] `S13-T1` (P0) Deploy to the partner environment (self-hosted) with documented install; validate capture completeness in production.
- [ ] `S13-T2` (P0) Start in **observe mode** (flags, no enforcement) to gather real distributions without friction.
- [ ] `S13-T3` (P0) Collect per-module flag volumes, severity distributions, and reviewer adjudications.

**Tuning**
- [ ] `S13-T4` (P0) Recalibrate thresholds per module using real data + adjudications; document before/after FP/FN.
- [ ] `S13-T5` (P0) Measure and reduce **reviewer time per flag**; target a concrete budget (e.g. median < 2 minutes) or rework/suppress the flag type.
- [ ] `S13-T6` (P0) Identify and fix the top sources of noise; where noise is irreducible, mark the module advisory and document why.
- [ ] `S13-T7` (P1) Gradually move selected high-confidence rules from observe to enforced gating, with partner sign-off.
- [ ] `S13-T8` (P1) Validate model-update drift handling: how thresholds behave when the host model changes.

**Operations & support**
- [ ] `S13-T9` (P0) Run the incident/on-call loop for real; verify every alert maps to a runbook; fix gaps.
- [ ] `S13-T10` (P0) Track support tickets/questions and convert recurring friction into docs or DX fixes.
- [ ] `S13-T11` (P1) Establish a feedback → backlog loop with the partner; prioritize by operational value, not novelty.
- [ ] `S13-T12` (P1) Produce a case-study draft (with partner approval) for `S14` distribution.

**Readiness**
- [ ] `S13-T13` (P0) GA readiness review against [`§8`](#8-definition-of-done-at-101): all criteria either met or explicitly waived with mitigation.
- [ ] `S13-T14` (P1) Postmortem of the launch period; update roadmap and risk register.

#### Standards Focus

Production operations, measurement discipline, human-in-the-loop ergonomics, feedback loops, honest signal-vs-noise handling.

#### Tests & Verification

- [ ] Production capture completeness validated (no unexplained gaps).
- [ ] Threshold changes reproduce expected FP/FN shifts on historical data (backtesting).
- [ ] Reviewer-time measurements recorded with methodology.
- [ ] Observe→enforce transitions auditable and reversible.

#### Exit Criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | Real production traffic captured with completeness and no host regressions | production metrics |
| 2 | Reviewer time per flag measured; noise reduced to an acceptable level | tuning report |
| 3 | At least one module enforced in production with partner sign-off | config + log |
| 4 | Incidents handled via runbooks; gaps fixed | incident log |
| 5 | GA readiness review completed | review doc |

#### Risks & Mitigations

- **Too much noise → partner disables it.** Mitigation: observe mode first, aggressive tuning, per-deployment thresholds, honest advisory labeling.
- **Production issue erodes trust.** Mitigation: runbooks, fast rollback, transparent comms, blameless postmortems.
- **Scope creep from partner requests.** Mitigation: log requests, defer non-critical to post-GA, protect the GA critical path.

#### Dependencies

`S12` (compliance/legal), `S11` (packaging/docs), `S10` (scale/DR), `S9` (security).

#### Handoff

A battle-tested system with real-world evidence and tuned signal. `S14` takes it to general availability and traction.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] Reviewer-time metric acceptable.
- [ ] No unresolved production incidents.
- [ ] GA readiness review signed off.
- [ ] Status Board updated: `S13` = `[x]`.
- [ ] If the partner disables the system due to noise/friction → **NO-GO**; revisit tuning before GA.

---

### Sprint `S14` — GA, Distribution & Traction

**Objective.** Publicly release Sentinel as a general-availability product, distribute it, and build the traction loop: users, feedback, retention, and a path to revenue. Maturity `101` is an ongoing state, not a finish line.

**Maturity:** `100 → 101` · **Duration:** ongoing · **Phase:** 6 Operate
**Team acceleration:** this phase benefits most from a small team (developer relations, support, sales) if available.

#### Deliverables

- Public GA announcement with a launch asset (post, talk, or demo video).
- Distribution channels live: PyPI, container registry, docs site, examples, community space.
- A support model and published SLA/version policy.
- A metrics loop: installs, active deployments, flags reviewed, retention, and qualitative feedback.
- A prioritized post-GA backlog informed by real usage.

#### Task Breakdown

**Launch**
- [ ] `S14-T1` (P0) GA release: version `1.0.0`, changelog, migration notes, signed artifacts, SBOM.
- [ ] `S14-T2` (P0) Launch assets: a 5-minute demo (capture → flag → hold → review), a technical write-up, and a short "why now" narrative.
- [ ] `S14-T3` (P0) Publish to PyPI, registry, and the docs site; verify all install paths.
- [ ] `S14-T4` (P1) Community space (Discussions/Discord) with contribution and issue templates and a moderation policy.

**Distribution & support**
- [ ] `S14-T5` (P0) Support model live: channels, response targets, supported-version policy.
- [ ] `S14-T6` (P0) Sales/design-partner one-pager: problem, mechanism, evidence, deployment, pricing.
- [ ] `S14-T7` (P1) Partner/referral program if pursuing commercial deals.
- [ ] `S14-T8` (P1) Optional: managed offering exploration (only if it doesn't compromise the self-hosted core promise).

**Traction & iteration**
- [ ] `S14-T9` (P0) Instrument (privacy-respectfully, opt-in) or otherwise measure adoption: installs, active deployments, flags adjudicated.
- [ ] `S14-T10` (P0) Collect and triage user feedback; maintain a public roadmap.
- [ ] `S14-T11` (P1) Publish follow-up technical content (methodology deep-dives, FP/FN transparency reports).
- [ ] `S14-T12` (P1) Establish a regular release cadence (e.g. monthly minor, immediate patch for security).

#### Standards Focus

Release discipline, community building, honest marketing, metric-driven iteration, sustainability.

#### Tests & Verification

- [ ] Public install paths verified post-launch (fresh environment).
- [ ] Release artifacts signed and verifiable by a third party.
- [ ] Support channels monitored and responsive.
- [ ] Metrics pipeline produces actionable numbers.

#### Exit Criteria (ongoing state)

| # | Criterion | Verified by |
|---|---|---|
| 1 | GA `1.0.0` published and installable by the public | fresh install |
| 2 | Docs, examples, and community operational | review |
| 3 | Support/version policy published | review |
| 4 | Adoption metrics flowing | dashboard |
| 5 | Post-GA roadmap prioritized from real usage | review |

#### Risks & Mitigations

- **Launch without adoption.** Mitigation: design-partner case study, targeted outreach to agent-building teams, and content that demonstrates the specific failure modes.
- **Support burden on a solo maintainer.** Mitigation: excellent docs, templates, a bounded support policy, and community self-help.
- **Drift from the product's integrity.** Mitigation: never compromise the self-hosted core or overclaim reliability; transparency reports keep trust.

#### Dependencies

`S13` (production validation and GA review).

#### Handoff

The project reaches maturity `101`: a distributed, supported, evidence-backed product with a traction loop, and a living roadmap for what comes after.

#### GO / NO-GO Checklist

- [ ] All Exit Criteria rows pass.
- [ ] `1.0.0` artifacts verified.
- [ ] Support and version policy live.
- [ ] Status Board updated: `S14` = `[x]`.
- [ ] No open critical security issue.

---

## 6. Cross-Cutting Continuous Backlogs

These never "complete"; they run alongside every sprint and are reviewed at each sprint boundary.

### 6.1 Engineering Health Backlog

- [ ] Keep `main` releasable at every commit; no broken windows.
- [ ] Pay down TODOs with linked issues; no untracked debt.
- [ ] Update dependencies on a schedule; security patches within SLA.
- [ ] Keep coverage ≥ 90% on core modules.
- [ ] Add a property test whenever a new core algorithm appears.
- [ ] Benchmarks tracked over time to catch performance regressions.
- [ ] ADRs written for every significant decision, no exceptions.

### 6.2 Product/Docs Backlog

- [ ] Every new config option documented with a safe default.
- [ ] Every new flag type documented with an example and remediation guidance.
- [ ] Quickstart re-verified after any public API change.
- [ ] Changelog reviewed for user-facing clarity each release.

### 6.3 Security Backlog

- [ ] Dependency audit green; no high/critical.
- [ ] Secret scanning clean.
- [ ] Threat model revisited after any architecture change.
- [ ] Disclosure inbox monitored; response within the published window.

### 6.4 Data/Privacy Backlog

- [ ] Redaction defaults reviewed as new event types are added.
- [ ] Retention policy tested end-to-end before each release.
- [ ] No external egress by default; any opt-in external call reviewed and documented.

---

## 7. Risk Register

Severity: H/M/L · Likelihood: H/M/L. Reviewed every sprint; owners assigned as the team grows.

| ID | Risk | Sev | Lik | Mitigation | Trigger to act |
|---|---|---|---|---|---|
| R1 | Probabilistic modules (faithfulness, eval-awareness) too noisy to be useful | H | M | Measure FP/FN; observe-mode default; advisory labeling; honest docs; tune per deployment | FP/FN miss budget in `S5`/`S13` |
| R2 | Capture loses events or blocks the host agent | H | L | Append-only store, losslessness tests, fail-open capture, queue overflow recording | Any gap detected |
| R3 | Framework instrumentors break on upgrades | M | H | Isolated modules, pinned compat tests, version matrix CI, generic decorator fallback | CI matrix failure |
| R4 | Gating friction leads to disabling the product | H | M | Configurable thresholds, observe mode, low-friction defaults, reviewer ergonomics | Partner disables in `S13` |
| R5 | Critical security vuln found late | H | M | Dedicated `S9` hardening, fuzzing, audits, SBOM, signed releases | Any critical finding |
| R6 | Compliance overclaim damages trust | M | M | Support-language mappings, legal review, no certification claims | Review flags overclaim |
| R7 | Scale/perf budgets unmeetable without redesign | M | M | Measure in `S10`; ADR + renegotiate or redesign before GA | Budget miss in `S10` |
| R8 | DR untested; audit trail lost after failure | H | L | Mandatory recorded restore drill in `S10` | Drill fails |
| R9 | Solo-maintainer burnout / bus factor | M | M | Docs-as-code, ADRs, automation, bounded support policy, optional team ramp | Velocity drop |
| R10 | Judge/embedding model drift changes results | M | M | Pin model versions, cache, record versions in flags, backtest on upgrade | Model upgrade |
| R11 | Scope creep delays GA | M | H | Phase gates, ADR-based descoping, protect critical path | Sprint slips > 50% |
| R12 | Public API churn breaks early adopters | M | M | API freeze in `S11`, deprecation policy, SemVer | Breaking change requested |

---

## 8. Definition of Done at `101`

The final sellable/complete checklist. Every box must be tickable (or deliberately waived with an ADR) for the project to be considered at maturity `101`.

### 8.1 Product & Engineering

- [ ] Public, typed, documented API frozen under SemVer with a deprecation policy.
- [ ] Instrumentation captures LLM, tool, and memory boundaries for LangChain, LangGraph, and raw Ollama with lossless replay.
- [ ] All five modules implemented, deterministic where claimed, with published FP/FN and honest admissibility labeling.
- [ ] Policy/gating engine auditable in under an afternoon, fail-safe, and configurable per deployment.
- [ ] Review API/UI lets a human adjudicate a flag from evidence efficiently.
- [ ] Performance budgets met or renegotiated via ADR.
- [ ] Adversarial suite green and reproducible; results published.
- [ ] Security dossier: threat model, fuzzing, audits, SBOM, signed artifacts, disclosure process.

### 8.2 Operations

- [ ] One-command self-hosted deployment documented and tested.
- [ ] SLO dashboards + alerting; every alert links a runbook.
- [ ] Backup/restore and DR drill executed and recorded.
- [ ] Capture-completeness monitoring live (the system watches itself).
- [ ] Supported-version policy and support channels live.

### 8.3 Distribution & Trust

- [ ] Quickstart verified at ≤ 15 minutes in a clean environment.
- [ ] Docs site complete: quickstart, concepts, integration, module guides, operations, security, API.
- [ ] Release pipeline: versioned PyPI package + container, changelog, SBOM, signatures, trusted publishing.
- [ ] License, terms, privacy policy, `SECURITY.md`/`security.txt`, responsible disclosure live.
- [ ] Compliance mappings (EU AI Act, NIST AI RMF) published as support, not certification.
- [ ] Exportable, verifiable audit pack.

### 8.4 Evidence & Traction

- [ ] At least one design partner running it in production with measured reviewer-time-per-flag.
- [ ] Case study (with permission) and demo available.
- [ ] Adoption metrics flowing; post-GA roadmap driven by real usage.
- [ ] Regular release cadence operational.

---

## 9. Appendices

### 9.1 Initial ADR Index

| ADR | Title | Sprint |
|---|---|---|
| 0001 | Python 3.12+ async-first SDK | `S-1` |
| 0002 | Postgres reference store, SQLite dev-only | `S-1` |
| 0003 | Capture/evaluation separation | `S-1` |
| 0004 | Evaluators as independent stream workers | `S-1` |
| 0005 | Minimal, auditable gating rules engine | `S-1` |
| 0006 | Self-hosted core path; opt-in external models | `S-1` |
| 0007 | Versioned event schemas + backward-compatible readers | `S-1` |
| 0008 | Local Ollama defaults for embeddings/judge; pluggable | `S-1` |
| 0009 | Small stable public API; SemVer/deprecation policy | `S-1` |
| 0010 | ULID event IDs + monotonic sequence for replay/idempotency | `S-1` |
| 0011 | Retention & legal-hold semantics | `S2` |
| 0012 | License selection (core) | `S-1`/`S12` |
| 0013 | Descope policy for probabilistic modules | `S4`/`S5`/`S8` |
| 0014 | Pricing/packaging (open core vs enterprise) | `S12` |

> Add ADRs as decisions arise during execution. Never let a significant decision live only in code or chat.

### 9.2 Glossary

- **Event** — an immutable, sequence-numbered record of a boundary crossing (LLM call, tool call/result, memory op, agent step, error).
- **Session** — a bounded sequence of events for one agent run, identified by `session_id`.
- **Ref / call graph** — explicit foreign-key links between events (tool result ↔ tool call ↔ dependent LLM call) enabling structural provenance.
- **Flag** — a structured finding from an evaluator: category, severity, confidence, evidence refs, adjudication state.
- **Module** — one evaluator implementing one named failure mode.
- **Checkpoint** — a host-agent integration point where the gate can hold an action (`before_tool`, `before_response`, `before_state_commit`).
- **Gate / policy engine** — the thin rules engine that maps flags + stakes to `proceed`/`hold`/`block`.
- **Observe mode** — gating configured to record flags without enforcing, used for tuning and trust-building.
- **FP/FN** — false-positive / false-negative rate of a module against its adversarial corpus.
- **Reviewer time per flag** — the human cost of adjudicating a flag; a first-class product metric.
- **Fail-open (capture)** — capture errors never crash the host agent. **Fail-safe (gate)** — gate errors hold rather than approve.

### 9.3 Suggested Repository Layout (target)

```
sentinel/
├─ src/sentinel/
│  ├─ __init__.py                 # public surface
│  ├─ config.py                   # pydantic-settings
│  ├─ instrument/                 # INV-1: capture only
│  │  ├─ session.py
│  │  ├─ registry.py
│  │  ├─ langchain.py
│  │  ├─ langgraph.py
│  │  ├─ http.py                  # Ollama / OpenAI-compatible
│  │  ├─ memory.py
│  │  └─ generic.py
│  ├─ store/                      # event store + migrations
│  │  ├─ protocol.py
│  │  ├─ postgres.py
│  │  ├─ sqlite.py
│  │  └─ migrations/
│  ├─ eval/                       # INV-1 boundary: no imports from instrument
│  │  ├─ worker.py                # EvaluatorWorker base
│  │  ├─ provenance_core.py       # shared by S3 & S4
│  │  ├─ provenance.py
│  │  ├─ memory_integrity.py
│  │  ├─ faithfulness.py
│  │  ├─ spec_gaming.py
│  │  └─ evaluation_awareness.py
│  ├─ gate/                       # INV-4: minimal, auditable
│  │  ├─ engine.py
│  │  ├─ rules.py
│  │  └─ service.py               # FastAPI review API
│  ├─ models/                     # event + flag schemas
│  └─ cli.py                      # replay, sessions, eval-fixtures, doctor, export
├─ tests/
│  ├─ unit/  integration/  property/  contract/  adversarial/  e2e/  perf/
├─ examples/
├─ docs/
│  ├─ adr/  modules/  operations/  security/  runbooks/
│  └─ design/SENTINEL_TDD.md       # this document
├─ deploy/                         # docker compose, future helm
├─ pyproject.toml
├─ uv.lock
└─ README.md
```

### 9.4 Change Log of This Document

| Version | Date | Change |
|---|---|---|
| v1.5 | 2026-09-24 | Sprint `S2` gate passed: event store green (190 tests / 92.4% coverage, `mypy --strict`/ruff/bandit clean). All `S2-T1…T19` tasks checked off; `S1-T15` sampling and `S1-T16` truncation landed as part of `S2` (previously deferred). Added ADR-0011 (least-privilege DB roles), `deploy/roles.sql`, `deploy/compose.postgres.yml` + migration runner + demo capture worker, CI integration/load jobs. `S2-T19` backup/restore smoke skips without `pg_dump`/`psql` (runs in CI with tooling). |
| v1.4 | 2026-09-23 | Sprint `S1` gate passed: instrumentation layer core green on `main` (taxonomy/refs/registry/config, capture worker, langchain/langgraph/openai-compat/memory/generic instrumentors, call-graph helper; 115 tests / 95.3% coverage, `mypy --strict`), `v0.0.3` GPG-signed tag. `S1-T15`/`S1-T16` deferred to `S2`. |
| v1.3 | 2026-09-23 | Sprint `S0` gate passed: vertical slice green on `main` (31 tests / 99.3% coverage, `mypy --strict`), `v0.0.2` GPG-signed tag. |
| v1.2 | 2026-09-23 | Sprint `S-1` gate passed: CI green on `main`, branch protection enabled, `v0.0.1` GPG-signed tag pushed. Sprint `S0` (vertical slice) in progress. |
| v1.1 | 2026-09-21 | Sprint `S-1` scaffolding executed: repo tree, governance files, pyproject/uv, tooling, CI, ADR 0001–0010. |
| v1.0 | 2026-09-21 | Initial master engineering plan derived from `safety_sdk.tex`; 15-sprint roadmap `-1 → 101`, global standards, risk register, DoD@101. |

---

*End of document. Update the Status Board and this changelog at every sprint boundary.*
