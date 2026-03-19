# architecture/AGENTS.override.md

## Architecture-sensitive subtree rules
- Start with a change schematic before editing code here.
- State allowed dependency direction for this subtree and preserve it.
- Keep orchestration, policy, I/O, and persistence roles distinct unless the local pattern says otherwise.
- Prefer explicit interfaces, typed contracts, and visible state transitions over implicit shared state.
- Update local design notes, ADRs, or architecture docs when behavior or boundaries change.
- Require `Architecture Reviewer`, `Clean Code Reviewer`, and `Gatekeeper` for non-trivial changes here.
