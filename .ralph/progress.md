# Progress Log
Started: Mon May 18 18:34:17 UTC 2026

## Codebase Patterns
- (add reusable patterns here)

---
## [2026-05-18 21:30:47 UTC] - US-004: Replace legacy fallback assumptions
Thread:
Run: 20260518-183418-2827287 (iteration 4)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-4.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-4.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: c379d37 fix(runtime): remove legacy fallback assumptions
- Post-commit status: `clean` after committing this progress entry
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_api_model_resolution tests.test_public_api tests.test_data tests.test_repo_audit tests.test_native_components tests.test_native_api_routing tests.test_suite` -> PASS (64 tests, 9 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (117 tests, 9 skipped)
  - Command: `make test` -> PASS (117 tests, 9 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/package-surface-audit.html
  - simpledet/simpledet/_legacy.py
  - simpledet/simpledet/_model_resolution.py
  - simpledet/simpledet/api.py
  - simpledet/simpledet/detectors/data.py
  - tests/test_api_model_resolution.py
  - tests/test_data.py
  - tests/test_native_components.py
  - tests/test_public_api.py
  - tests/test_repo_audit.py
- What was implemented
  - Removed hardcoded native head/neck fallback lists so listing comes from registries and returns empty instead of masking missing optional native dependencies.
  - Added a shared legacy config boundary error and rejected MMDetection-style `.py` config paths in project config loading, project config initialization, and detector config loading.
  - Strengthened maintained package audit coverage with lower-case legacy runtime text checks plus AST checks for static and dynamic imports of the legacy runtime modules.
  - Added focused tests that RetinaNet native construction returns SimpleDet-owned torch module wrappers under the native build path.
  - Updated the package surface audit to document explicit unsupported import/conversion behavior for legacy `.py` config paths.
  - Security/performance/regression review: no new secret handling or code execution was added, legacy path handling fails closed, registry listing remains bounded to in-memory names, and review blockers were fixed before full validation.
- **Learnings for future iterations:**
  - The environment still has no bare `python`; use `python3` or Makefile defaults for runnable validation.
  - Importing full native registries for metadata can regress base installs unless optional dependency errors are handled as absence, not fallback support.
  - The repo audit should keep both text-level legacy references and AST import checks so production fallback assumptions cannot hide in strings or dynamic imports.
---
## [2026-05-18 20:42:38 UTC] - US-003: Define registry contract
Thread:
Run: 20260518-183418-2827287 (iteration 3)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-3.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-3.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 401b63b feat(registry): define component contract
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest tests.test_native_backend_plan tests.test_suite tests.test_native_components tests.test_native_api_routing tests.test_api_model_resolution tests.test_deps` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_native_backend_plan tests.test_suite tests.test_native_components tests.test_native_api_routing tests.test_api_model_resolution tests.test_deps` -> PASS (56 tests)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (113 tests, 9 skipped)
  - Command: `make test` -> PASS (113 tests, 9 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
  - Command: `PYTHONPATH=simpledet python3 - <<'PY' ... registry alias smoke ... PY` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/api-reference.html
  - simpledet/simpledet/extensions/__init__.py
  - simpledet/simpledet/extensions/registry.py
  - simpledet/simpledet/native/assemblers.py
  - simpledet/simpledet/native/backbones.py
  - simpledet/simpledet/native/heads.py
  - simpledet/simpledet/native/modeling.py
  - simpledet/simpledet/native/necks.py
  - simpledet/simpledet/suite/catalog.py
  - tests/test_native_backend_plan.py
- What was implemented
  - Added `ComponentMetadata`, `DependencyRequirement`, alias-aware lookup, duplicate alias/name rejection, dependency requirement checks, and actionable kind-scoped unknown lookup errors to the extension registry.
  - Registered metadata for native backbones, heads, necks, and detector families including aliases for VFNet, FOVEA/FoveaBox, RepPoints, YOLOF, CenterNet, Grid R-CNN, Cascade R-CNN, Faster R-CNN, and Mask R-CNN.
  - Updated suite inspection/resolution and native model lookup to use registry resolution instead of scattered case-only maps.
  - Added tests for alias normalization, duplicate alias and normalized name rejection, missing dependency messages, unknown lookup guidance, and inherited metadata on exact alias names.
  - Updated the API reference with the registry metadata fields, alias lookup behavior, dependency error intent, and test coverage anchor.
  - Security/performance/regression review: registry remains stdlib-only, optional dependency imports stay explicit/lazy, lookup work is bounded to small in-memory registries, review findings on alias metadata precedence were fixed, and full regression gates passed.
- **Learnings for future iterations:**
  - `python` is still unavailable; the Makefile and successful checks use `python3`.
  - Stacked decorator registrations need metadata propagation so exact public aliases do not lose dependency or validation contract details.
  - Alias resolution must prefer explicit aliases before compact canonical-name matching to avoid sparse secondary metadata entries.
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
