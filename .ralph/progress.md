# Progress Log
Started: Mon May 18 18:34:17 UTC 2026

## Codebase Patterns
- (add reusable patterns here)

---
## [2026-05-18 19:52:23 UTC] - US-002: Normalize package metadata and extras
Thread:
Run: 20260518-183418-2827287 (iteration 2)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-2.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-2.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 95bdd84 feat(packaging): normalize optional extras
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest tests.test_packaging` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_packaging` -> PASS (8 tests)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (105 tests, 9 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - Makefile
  - README.md
  - docs/installation.html
  - docs/package-surface-audit.html
  - docs/quickstart.html
  - docs/troubleshooting.html
  - pyproject.toml
  - simpledet/simpledet/cli.py
  - simpledet/simpledet/detectors/_deps.py
  - tests/test_packaging.py
  - uv.lock
- What was implemented
  - Split optional extras into explicit `cpu`, `dev`, `docs`, `geo`, `plots`, and `timm` groups while keeping base `dependencies = []`.
  - Removed TIMM from CPU runtime checks so `simpledet[cpu]` and base imports do not require TIMM; TIMM-backed backbones still require the TIMM extra at use.
  - Updated install docs and Makefile editable install guidance for `simpledet`, `simpledet[cpu]`, `simpledet[timm]`, and `.[cpu,timm,dev]`.
  - Added packaging tests for explicit extras, no MMDetection/MMCV metadata, built wheel metadata, and base import without optional `torch`, `timm`, `rasterio`, or `matplotlib`.
  - Security/performance/regression review: metadata/docs/import-boundary changes only, no new external input handling, no added hot-path work, and full regression tests passed via the repo-supported `python3` runner.
- **Learnings for future iterations:**
  - `uv lock` updates only the SimpleDet extra metadata for this story; no dependency version churn was needed.
  - Empty `docs = []` still emits `Provides-Extra: docs` in wheel metadata.
  - This environment still has no bare `python`; use `python3` or Makefile defaults for executable validation.
---
## [2026-05-18 19:08:39 UTC] - US-001: Audit current package surface
Thread:
Run: 20260518-183418-2827287 (iteration 1)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-1.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: fa24f63 docs(audit): add package surface audit
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (103 tests, 9 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
- Files changed:
  - .github/PULL_REQUEST_TEMPLATE.md
  - docs/developer-guide.html
  - docs/package-surface-audit.html
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
  - Added a static package surface audit covering modules, APIs, optional extras, tests, docs pages, legacy MMDetection/MMEngine references, detector alias status, and minimum PRD story order.
  - Linked the audit from the developer guide and tightened detector-family guidance so catalog/config-only entries are not documented as support.
  - Ran security, performance, and regression review: static docs only, no new secret handling or dynamic execution, no runtime code path changes, and regression gates passed through the repo-supported `python3` runner.
- **Learnings for future iterations:**
  - Static docs are HTML-only; `docs-check` rejects Markdown docs in `docs/`.
  - This environment has `python3` but no bare `python`; the Makefile uses `python3`.
  - Several aliases are native-compatible scaffolds, not architecture-faithful detector implementations yet.
---
