# Sentinel Documentation

Documentation home. The canonical engineering plan is
[`SENTINEL_TDD.md`](https://github.com/uzairlol/sentinel/blob/main/docs/design/SENTINEL_TDD.md)
at `docs/design/`.

## Contents

- [Architecture Decision Records](adr/README.md) — every significant decision, ADR-0001+.
- [Design](design/) — the original conceptual design documents (`.tex` / `.pdf`).
- [Event Schema](event-schema.md) — the payload contract every captured event carries (`S1-T17`).
- [Integration Guide](integration-guide.md) — ways to instrument an agent and query the trace (`S1-T19`).
- Modules — per-evaluator methodology and configuration guides (written during Sprints `S3`–`S8`).
- Operations — deployment, SLOs, alerts, gatekeeping (Sprints `S7`, `S10`+).
- Runbooks — "what to do when X breaks" (Sprints `S10`+).
- Security — threat model and adversarial results (Sprint `S9`).

> The public-facing documentation site (MkDocs Material) is delivered in Sprint
> `S11`; this tree is the working source for it.
