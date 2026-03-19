# Agent-instruction eval starter

Use this when `AGENTS.md`, skills, or custom agents change.

Minimal loop:
1. Define 4-10 must-pass prompts.
2. Include both positive and negative trigger cases.
3. Run the prompts in a clean repo checkout or scratch workspace.
4. Score the result with the schema in `result.schema.json` or an equivalent rubric.
5. Add real misses as new prompt cases.

Artifacts here:
- `prompts.csv.example` — starter routing cases
- `result.schema.json` — structured result shape for consistent scoring
