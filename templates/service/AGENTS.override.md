# service/AGENTS.override.md

## Service-local rules
- Use service-local build and test commands instead of repo-wide defaults when they exist.
- Treat all service boundaries, auth checks, and configuration surfaces as high risk.
- Keep handlers schematic: validate near the boundary, delegate policy, isolate side effects, and keep persistence explicit.
- Update service-local docs and examples when behavior changes.
- Keep overrides narrow: only add what differs from the root file.
