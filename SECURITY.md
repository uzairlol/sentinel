# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.0.x | Development milestones; fixes prioritized but not SLO-guaranteed |

Security fixes are backported to supported versions. A supported-version policy
with explicit SLAs is published at GA (see `S14` in `SENTINEL_TDD.md`).

## Reporting a vulnerability

**Do not open a public GitHub issue.** Sentinel is a safety product; a flaw in
it can undermine the agents it monitors.

Report privately instead:

- Email: **uarif2093@gmail.com** (PGP key published here once signature is required)
- GitHub private security advisories: enabled once the repository is public

Include, if possible:

1. Affected component and version
2. Description of the vulnerability and its impact
3. Steps to reproduce, ideally a minimal failing test or trace
4. Suggested fix, if you have one

## Response expectations

- **Acknowledgment** of every report within **72 hours**.
- **Assessment** of validity and severity within **10 days**.
- **Fix and release** timeline once valid, matched to severity:
  - Critical: as fast as possible, typically within days.
  - High: within one patch release.
  - Medium/Low: scheduled into the roadmap.

If the report is valid but cannot be fixed immediately, we publish an advisory
with mitigations and a timeline.

## Scope

In scope: everything in the `sentinel-sdk` package and its reference deployment
(`deploy/`). Out of scope: vulnerabilities in upstream dependencies (report those
to the upstream project); vulnerabilities in host agents that Sentinel merely
instruments (report those to the agent's maintainers).

## Safe harbor

We commit to no legal action against security researchers who:
- Report vulnerabilities through the channel above,
- Make a good-faith effort to avoid privacy violations, destruction of data, and
  interruption or degradation of services,
- Do not publicly disclose the issue before a fix is shipped.

For the full policy on handling of security-sensitive findings, see
`docs/security/threat-model.md` when published (Sprint `S9`).
