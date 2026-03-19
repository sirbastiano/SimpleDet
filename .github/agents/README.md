# Custom agent pack for coding repositories

This pack complements the root `AGENTS.md` with specialist agent profiles, repo skills, eval templates, and stricter delivery gates.

## Design rules
- `feature-builder` is the main orchestrator for non-trivial coding work.
- All review-style specialists are read-only by default and should keep narrow tool scopes.
- Only one write-capable specialist should mutate the codebase at a time.
- `instruction-maintainer` owns changes to `AGENTS.md`, skills, custom agents, and instruction evals.
- `gatekeeper` is the final pass/fail stage for non-trivial work and must check `G0` through `G9`.

## Included specialists
- `planner` — discovery, blast radius, validation planning
- `architecture-reviewer` — change schematic and architecture fit
- `implementer` — focused code changes
- `test-engineer` — tests, repro, and validation
- `clean-code-reviewer` — maintainability and schematic readability
- `reviewer` — regressions, scope, and contract drift
- `security-reviewer` — trust boundaries and misuse cases
- `docs-writer` — docs, examples, migration notes
- `skill-instantiator` — create missing skills/roles and propose specialist cohorts
- `dependency-curator` — dependency change risk analysis
- `migration-planner` — rollout/rollback planning
- `instruction-maintainer` — agent-system maintenance and routing quality
- `gatekeeper` — final gate evaluation

## Included repo skills
Canonical repo skills:
- `exec-plan` — produce `PLANS.md` / ExecPlans for large changes
- `implement` — perform bounded source-code edits
- `plan` — create explicit sequencing, invariants, and scope for non-trivial work
- `publish` — run final release-facing checks and handoff completeness
- `docs` — keep docs/examples/migration notes aligned with behavior
- `skill-instantiator` — add missing skills and specialist roles after validating repeated capability gaps

## Compatibility notes
- GitHub custom agents use Markdown profiles with YAML frontmatter in `.github/agents/`.
- Keep each custom agent focused. If an agent prompt grows too large or broad, split it into specialists or skills.
- `handoffs` can help in IDEs, but some environments ignore them. The workflow must still make sense without handoff support.
- Prefer least-privilege `tools` lists over granting every tool to every specialist.

## Recommended routes
- small bug fix -> implementer -> test-engineer -> reviewer
- cross-file bug fix -> planner -> architecture-reviewer -> implementer -> test-engineer -> clean-code-reviewer -> reviewer -> gatekeeper
- feature -> exec-plan skill -> planner -> architecture-reviewer -> implementer -> test-engineer -> clean-code-reviewer -> reviewer -> docs-writer if needed -> gatekeeper
- dependency upgrade -> planner -> dependency-curator -> architecture-reviewer -> implementer -> test-engineer -> reviewer -> gatekeeper
- migration -> migration-planner -> planner -> architecture-reviewer -> implementer -> test-engineer -> security-reviewer -> docs-writer -> clean-code-reviewer -> reviewer -> gatekeeper
- AGENTS/skill/custom-agent change -> instruction-maintainer -> reviewer -> gatekeeper
