# Agent contract schema (template)

This file documents the optional `agent_contract` block supported by all packaged role templates.

## Required shape when present

- `contract_version` (string)
- `agent_contract.scope` (string)
- `agent_contract.risk_level` (string)
- `agent_contract.inputs` (array of strings)
- `agent_contract.outputs` (array of strings)
- `agent_contract.handoff_to` (array of strings)
- `agent_contract.artifacts` (array of strings)
- `agent_contract.must_not_change` (array of strings)
- `agent_contract.done_criteria` (array of strings)
- `agent_contract.escalation_criteria` (array of strings)

## Suggested interpretation

- `inputs`: what the next actor should provide before handoff.
- `outputs`: what this role must produce before handoff.
- `handoff_to`: likely next role names.
- `artifacts`: files, notes, or event IDs created during the stage.
- `must_not_change`: explicit boundaries for the role.
- `done_criteria`: acceptance condition list used by downstream control roles.
- `escalation_criteria`: conditions requiring explicit escalation.

## Example

```toml
contract_version = "1.0"

[agent_contract]
scope = "implementation"
risk_level = "medium"
inputs = ["Task handoff", "Acceptance criteria"]
outputs = ["Minimal patch", "Changed file list", "Validation notes"]
handoff_to = ["test-engineer", "reviewer"]
artifacts = ["diff_summary.md", "validation_notes.md"]
must_not_change = ["Files outside scope", "Unrelated refactors"]
done_criteria = ["Changes are minimal", "Scope was respected"]
escalation_criteria = ["Cannot isolate risk", "New dependency path discovered"]
```
