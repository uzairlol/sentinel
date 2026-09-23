# Contributing to Sentinel

Thanks for helping build a defensive layer for autonomous agents. This document
defines how we work. It is short on purpose: the rules are few and non-negotiable
so that review time goes to substance, not process.

## The workflow, in one paragraph

Trunk-based development. `main` is always releasable. Work happens on short-lived
branches named `feat/<scope>-<task>`, `fix/...`, `refactor/...`, `docs/...`,
`chore/...`, or `ci/...`. Every change lands via a pull request — even a
one-line fix. PRs are squash-merged into `main`. Conventional Commits. Done means
green CI plus a checked-off Definition of Done (see below).

## Before you start

1. Find your place in the roadmap: [`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md).
   Only implement work belonging to the current or an older sprint. If your
   change isn't mapped to a task ID, open an issue first.
2. Check for an open ADR-able decision. If your change contains a significant
   design decision, it needs an ADR under `docs/adr/` — not a comment thread.
3. Set up the environment:
   ```bash
   uv sync --group dev
   pre-commit install
   ```

## Branch naming and commits

- Branch: `feat/s3-t4-provenance-diff`
- Commit message: Conventional Commits —
  `feat(scope): summary`

  Scopes: `instrument`, `store`, `eval.provenance`, `eval.memory`,
  `eval.faithfulness`, `eval.spec`, `eval.evalaware`, `gate`, `api`, `ui`,
  `cli`, `docs`, `ci`, `build`, `packaging`.

- Put the task ID in the commit body (e.g. `Closes #123, implements S3-T4.`).
- Keep commits atomic: one logical change per commit.

## The pull request checklist

Every PR must be self-reviewable without the author present:

- [ ] Linked to a task/issue (ID in title or description)
- [ ] What changed and why are described
- [ ] Tests added: happy path + at least one failure/adversarial path
- [ ] Docs updated if public behavior changed
- [ ] ADR added/updated if a design decision changed
- [ ] Migration included if the schema changed
- [ ] Breaking change? Called out with a semver note
- [ ] Security review done for anything touching inputs, secrets, SQL, or the network
- [ ] Rollback method stated
- [ ] CI green: lint, types, tests, coverage

## Definition of Done (per item)

1. Merged to `main` via PR; CI fully green.
2. `ruff check` + `ruff format --check` + `mypy --strict` clean on changed modules.
3. Tests cover happy and failure paths; coverage ≥ 90% on `src/sentinel/`.
4. Public functions typed and docstringed; new public API documented.
5. Logs/metrics added where the change crosses a boundary.
6. No `TODO` without an issue; no commented-out code; no secrets.
7. `main` remains installable/releasable at any commit.

## Testing conventions

- Match the pyramid in `SENTINEL_TDD.md` §2.5: unit, property, integration,
  contract, adversarial, e2e, perf.
- Each evaluator needs FP/FN measurement against its adversarial corpus before it
  is called "done".
- Mark slow or service-dependent tests with `@pytest.mark.integration`.

## Code style

- Formatting and linting are enforced by `ruff` (see `pyproject.toml`).
- Types are enforced by `mypy --strict`.
- No comments that restate the code; prefer docstrings that explain "why".
- Imports: standard library, third-party, first-party — sorted by `ruff` isort.

## Reporting issues

- Bugs: use the bug template (reproduction, expected vs actual, environment).
- Security: do **not** open a public issue. Follow [`SECURITY.md`](SECURITY.md).

## Reviewing

- Reviewer reads for correctness, safety, and evidence — not style (that's the tools' job).
- Approve only when the Definition of Done passes; ask for fixes otherwise.
- Be constructive, specific, and fast — review should not be the bottleneck.
