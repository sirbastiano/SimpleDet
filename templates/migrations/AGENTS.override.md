# migrations/AGENTS.override.md

## Migration-specific rules
- Ask first before destructive or irreversible operations.
- Include forward path, rollback path, and data-backfill notes in every change.
- Prefer staged, backward-compatible rollouts.
- Run dry-runs, schema diffs, and the highest-fidelity migration validation available.
- Require `Migration Planner`, `Architecture Reviewer`, and `Gatekeeper` for non-trivial changes here.
