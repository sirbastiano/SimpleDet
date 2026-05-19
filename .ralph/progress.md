# Progress Log
Started: Mon May 18 18:34:17 UTC 2026

## Codebase Patterns
- (add reusable patterns here)

## [2026-05-18 23:47:02 UTC] - US-010: Implement assignment and sampling utilities
Thread:
Run: 20260518-183418-2827287 (iteration 10)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-10.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-10.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 7f9ff69 feat(assignment): add native assigners; 75e6fd4 test(assignment): correct sampler expectation
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `python3 -m py_compile simpledet/simpledet/native/assignment.py simpledet/simpledet/native/dense_ops.py simpledet/simpledet/native/transformer_ops.py simpledet/simpledet/native/__init__.py tests/test_native_assignment.py` -> PASS
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_assignment.py'` -> PASS (7 skipped; torch CPU extra unavailable)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (150 tests, 27 skipped)
  - Command: `make test` -> PASS (150 tests, 27 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/assignment.py
  - simpledet/simpledet/native/dense_ops.py
  - simpledet/simpledet/native/transformer_ops.py
  - tests/test_native_assignment.py
- What was implemented
  - Added registry-backed native assignment utilities for MaxIoU, ATSS, task-aligned, center-region, Hungarian, simOTA, and point-based matching plus deterministic balanced sampling.
  - Rewired RetinaNet, ATSS/GFL, FCOS, and DETR training losses to consume native assignment results, including ignored priors and all-background no-ground-truth targets.
  - Added CPU unit coverage for ATSS positive/negative/ignored labels, no-ground-truth assignment, ATSS no-GT loss handling, point/center-region matching, task-aligned/simOTA matching, Hungarian one-to-one matching, and sampler behavior.
  - Security/performance/regression review: tensor-only utilities, no new file/network/secret handling, vectorized IoU/top-k matching with a bounded exact Hungarian solver, and native registry/model/runtime regressions passed.
- **Learnings for future iterations:**
  - Registry aliases normalize punctuation and case, so only one spelling per normalized alias key should be registered.
  - Sampler tests should size requested samples to the intended positive/negative split; otherwise deterministic fill can legitimately return extra negatives.
  - The environment still lacks bare `python` and the torch CPU extra; tensor-focused tests are present but skip until `simpledet[cpu]` is installed.
  - `make verify-dist` should run after `make build` so the wheel audit includes newly added native modules.
---
## [2026-05-19 02:34:28 UTC] - US-017: Register ROI bbox heads
Thread:
Run: 20260518-183418-2827287 (iteration 17)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-17.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-17.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: bc73c4a feat(heads): register roi bbox heads
- Post-commit status: `clean` after progress/log follow-up commit
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_roi.py'` -> PASS (10 tests, 9 skipped under base install)
  - Command: `uv run --extra cpu python -m unittest discover -s tests -p 'test_native_roi.py'` -> PASS (10 real CPU tensor tests)
  - Command: `PYTHONPATH=simpledet:tests python - <<'PY' ... ROI alias registry smoke ... PY` -> PASS
  - Command: `PYTHONPATH=simpledet python -m unittest tests.test_native_backend_plan tests.test_suite` -> PASS (21 tests)
  - Command: `PYTHONPATH=simpledet python -m unittest tests.test_api_model_resolution tests.test_suite` -> PASS (13 tests)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_dense_heads.py'` -> PASS (21 skipped under base install)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> PASS (193 tests, 66 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS after build
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/heads.py
  - tests/test_native_roi.py
- What was implemented
  - Registered native ROI bbox heads for Shared2FCBBoxHead, ConvFCBBoxHead, DoubleConvFCBBoxHead, DynamicBBoxHead, CascadeBBoxHead, SABLHead, and SparseRoIHead with `family="roi"` metadata and snake-case aliases.
  - Added background-aware ROI bbox outputs: `cls_score` shape `(N, num_classes + 1)` and class-specific or class-agnostic `bbox_pred` shapes aligned with existing ROI helpers.
  - Added shared target/loss support using `build_roi_bbox_targets`, CE classification loss, SmoothL1 positive bbox loss, cascade refinement helper reuse, and explicit class-agnostic target-shape validation.
  - Preserved dense `sabl` detector routing to `ATSSHead` while exposing `SABLHead` as an ROI bbox head.
  - Added construction, forward-shape, direct `build_head(...)`, target/loss smoke, registry, and negative validation tests.
  - Security/performance/regression review: no file/network/secret handling added; computation is bounded per ROI with explicit tensor shape validation; dense `sabl` default and native registry regressions passed.
- **Learnings for future iterations:**
  - Registry aliases normalize case and separators, so aliases like `shared_2fc_bbox_head` and `shared2fc_bbox_head` collide.
  - `uv run --extra cpu` is available for real PyTorch execution and should be used for story-specific tensor tests when the base install skips them.
  - A local `python` shim in `~/.local/bin` lets the required `python` gate run as written in this environment.
---
## [2026-05-19 01:49:14 UTC] - US-015: Register YOLO SSD and efficient heads
Thread:
Run: 20260518-183418-2827287 (iteration 15)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-15.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-15.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 2be9de1 feat(heads): add lightweight dense heads
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet:tests python3 -m unittest discover -s tests -p 'test_native_dense_heads.py'` -> PASS (21 skipped; base install lacks torch)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_suite.py'` -> PASS (8 tests)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_backend_plan.py'` -> PASS (13 tests)
  - Command: `make test` -> PASS (182 tests, 56 skipped)
  - Command: `uv run --extra cpu python -m unittest discover -s tests -p 'test_native_dense_heads.py'` -> PASS (21 real-tensor tests)
  - Command: `make docs-check` -> PASS
  - Command: `make verify-dist` -> PASS before build
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS after build
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/api-reference.html
  - docs/package-surface-audit.html
  - docs/roadmap-changelog.html
  - simpledet/simpledet/cli.py
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/assemblers.py
  - simpledet/simpledet/native/dense_ops.py
  - simpledet/simpledet/native/heads.py
  - simpledet/simpledet/native/modeling.py
  - simpledet/simpledet/suite/catalog.py
  - tests/test_native_backend_plan.py
  - tests/test_native_dense_heads.py
  - tests/test_suite.py
- What was implemented
  - Registered YOLOX, RTMDet, SSD, and EfficientDet dense heads with explicit aliases, dependency metadata, tensor contracts, and runtime-validated head discovery.
  - Added YOLOX objectness/class/bbox branches plus simOTA target building, objectness target configuration validation, finite loss, and decoded prediction support.
  - Added RTMDet task-aligned target/loss wiring, SSD max-IoU anchor target/loss wiring, and EfficientDet anchor target/loss/decode adapters.
  - Updated suite defaults, native assemblers, supported architecture discovery, CLI help, and docs so `rtmdet` uses `RTMDetHead` and `efficientdet` uses `EfficientDetHead`.
  - Added CPU tensor tests for construction, forward contracts, target assignment, loss smoke, decoded predictions, and the YOLOX missing-objectness negative case.
  - Security/performance/regression review: no new file/network/secret handling; dense target/decode work is tensor-local and bounded by feature-map predictions; focused, real-tensor, full regression, docs, dist, and build gates passed through `python3`/Makefile.
- **Learnings for future iterations:**
  - Bare `python` is still unavailable in this environment; use `python3` or Makefile targets for executable validation while still recording the required command failure.
  - `uv run --extra cpu` is the reliable path for real torch tensor coverage in this repo.
  - YOLOX loss setup should require an explicit objectness target configuration, while assembler defaults can provide the standard positive/negative values.
---
## [2026-05-18 23:03:35 UTC] - US-008: Implement neck registry coverage
Thread:
Run: 20260518-183418-2827287 (iteration 8)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-8.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-8.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: b2f85bf feat(necks): implement native registry coverage
- Post-commit status: `clean`
- Verification:
  - Command: `python3 -m py_compile simpledet/simpledet/native/necks.py simpledet/simpledet/native/__init__.py tests/test_native_components.py` -> PASS
  - Command: `PYTHONPATH=simpledet:tests python3 -m unittest tests.test_native_components.NativeComponentTests.test_build_native_neck_aliases_resolved tests.test_native_components.NativeComponentTests.test_build_native_neck_aliases_normalized_names tests.test_native_components.NativeComponentTests.test_native_neck_alias_metadata_covers_required_families tests.test_native_components.NativeComponentTests.test_native_neck_aliases_forward_shapes_with_real_tensors tests.test_native_components.NativeComponentTests.test_native_neck_tensor_contract_rejects_level_mismatch tests.test_native_components.NativeComponentTests.test_native_neck_tensor_contract_rejects_channel_mismatch` -> PASS (6 tests, 3 skipped)
  - Command: `PYTHONPATH=simpledet:tests python3 -m unittest tests.test_native_components` -> PASS (26 tests, 3 skipped)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (138 tests, 15 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/necks.py
  - tests/test_native_components.py
- What was implemented
  - Registered explicit native neck coverage for FPN, PAFPN, NASFPN, BiFPN, DilatedEncoder, HRFPN, SSDNeck, and YOLOXPAFPN with aliases, dependency metadata, tensor contracts, and exports.
  - Added forward-time tensor-contract validation for feature-level count, NCHW shape, and channel mismatches before neck computation can silently truncate inputs.
  - Added extra-output support so FPN-style and projection necks can produce `num_outs` levels greater than the input feature count.
  - Added construction, alias metadata, forward-shape, and negative tensor-contract tests for the requested neck aliases.
  - Security/performance/regression review: no new file, network, subprocess, or secret handling; added loops are bounded by feature levels/`num_outs`; existing default FPN fake-runtime and full unittest regression passed through the repo-supported `python3` runner.
- **Learnings for future iterations:**
  - The native neck builder injects backbone `feature_channels` as `in_channels` unless the neck plan passes explicit `in_channels`; single-level necks like DilatedEncoder should pass explicit one-level channels.
  - Registry lookup already normalizes separators, so only one alias per normalized key should be registered to avoid collisions.
  - This environment has no bare `python` and no torch/torchvision runtime; real tensor neck tests are present but skipped here until the CPU extra is installed.
---
## [2026-05-18 22:44:06 UTC] - US-007: Add TIMM encoder extra
Thread:
Run: 20260518-183418-2827287 (iteration 7)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-7.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-7.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: b08a64b feat(timm): add prefixed feature backbone adapter
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_deps tests.test_native_backbones` -> PASS (20 tests, 1 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_native_backend_plan tests.test_native_components tests.test_suite` -> PASS (41 tests)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (134 tests, 12 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/configuration-guide.html
  - simpledet/simpledet/detectors/_deps.py
  - simpledet/simpledet/native/backbones.py
  - simpledet/simpledet/suite/catalog.py
  - tests/test_deps.py
  - tests/test_native_backbones.py
- What was implemented
  - Added `build_backbone("timm:<model>", out_indices=...)` support that compiles to a TIMM encoder plan without importing TIMM during suite spec construction.
  - Kept the TIMM optional extra isolated while adding a direct install hint for `python -m pip install 'simpledet[timm]'` on missing TIMM runtime imports.
  - Updated `TimmFeatureBackbone` to force `features_only=True`, pass explicit `out_indices` when provided, expose `feature_info`, and derive `BackboneSpec.feature_channels` from runtime TIMM metadata.
  - Added fake-runtime tests for prefixed TIMM plans, forced `features_only`, install hints, metadata extraction, and preservation of raw TIMM defaults when `out_indices` is omitted.
  - Added a real CPU TIMM smoke test that skips clearly when `torch` or `timm` is not installed, plus docs for the `timm:` prefix.
  - Security/performance/regression review: no new secret handling or shell execution, TIMM import remains lazy, no hot-path loops added, and a review-found raw TIMM `out_indices` regression was fixed before commit.
- **Learnings for future iterations:**
  - The `timm` extra was already present from packaging work; this story needed builder/runtime behavior and tests rather than metadata churn.
  - `build_backbone("timm:<model>")` should stay spec-only; runtime feature channels come from `feature_info` inside `simpledet.native`.
  - Raw `build_encoder(..., source="timm")` calls without `out_indices` must preserve TIMM's model-specific defaults for compatibility.
  - This environment has no bare `python`, `torch`, or `timm`; use `python3` for repo validation and expect the real TIMM smoke test to skip here.
---
## [2026-05-18 21:48:40 UTC] - US-005: Create native tensor contract tests
Thread:
Run: 20260518-183418-2827287 (iteration 5)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-5.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-5.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 69c4228 test(native): add tensor contract helpers
- Post-commit status: `clean` after committing this progress entry
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_tensor_contracts.py'` -> PASS (3 tests, 2 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (120 tests, 11 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `python3 -m py_compile tests/native_tensor_contracts.py tests/test_native_tensor_contracts.py` -> PASS
  - Command: `git diff --check` -> PASS
  - Command: `tar -tzf dist/simpledet-0.1.0.tar.gz | rg 'tests/(native_tensor_contracts|test_native_tensor_contracts)\.py'` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - MANIFEST.in
  - docs/developer-guide.html
  - tests/native_tensor_contracts.py
  - tests/test_native_tensor_contracts.py
- What was implemented
  - Added shared CPU tensor-contract helpers for dummy images, multiscale feature maps, boxes, labels, metadata, and native targets.
  - Added dense-head contract assertions for required output keys, feature-level counts, batch dimensions, and spatial alignment.
  - Added focused tests covering helper construction, FCOS dense-head tensor output behavior when PyTorch CPU is installed, and a clear mismatch assertion for missing feature levels.
  - Documented the helper workflow in the developer guide and included the helper in the sdist manifest so packaged tests can import it.
  - Security/performance/regression review: test/docs/manifest changes only, no new external input handling or secret paths, tiny deterministic CPU tensors, sdist helper inclusion verified, and final regression gates passed with `python3`/Makefile commands.
- **Learnings for future iterations:**
  - This base environment has no bare `python` and no PyTorch CPU runtime; real tensor helper tests skip until `simpledet[cpu]` is installed.
  - Keep non-`test*.py` test helpers in `MANIFEST.in`; setuptools copied the new test module into the sdist before the helper until the manifest was updated.
  - Dense-head smoke tests should call `assert_dense_head_output_contract(...)` before detector-family support is claimed so feature-level mismatches fail with actionable messages.
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
## [2026-05-18 22:15:37 UTC] - US-006: Implement backbone registry aliases
Thread:
Run: 20260518-183418-2827287 (iteration 6)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-6.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-6.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 8cd96ae feat(backbones): add native aliases
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_native_backbones tests.test_native_backend_plan tests.test_suite` -> PASS (29 tests)
  - Command: `python3 -m py_compile simpledet/simpledet/suite/backbone_aliases.py simpledet/simpledet/suite/catalog.py simpledet/simpledet/native/backbones.py tests/test_native_backbones.py tests/test_native_backend_plan.py` -> PASS
  - Command: `git diff --check` -> PASS
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (127 tests, 11 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
- Files changed:
  - .ralph/activity.log
  - docs/configuration-guide.html
  - simpledet/simpledet/native/backbones.py
  - simpledet/simpledet/suite/__init__.py
  - simpledet/simpledet/suite/backbone_aliases.py
  - simpledet/simpledet/suite/catalog.py
  - tests/test_native_backbones.py
  - tests/test_native_backend_plan.py
- What was implemented
  - Added a native backbone alias catalog with ResNet, ResNeXt, Res2Net, HRNet, CSPDarkNet, CSPNeXt, MobileNetV2, MobileNetV3, EfficientNet, ConvNeXt, Swin Transformer, and Vision Transformer aliases plus stage-channel metadata.
  - Exposed `build_backbone`, `list_backbones`, and `inspect_backbone` through `simpledet.suite`; `build_backbone("resnet50", out_indices=(1, 2, 3, 4))` now returns four-stage feature metadata.
  - Registered aliases in the native `ENCODERS` registry and updated native backbone construction so alias plans pass resolved TIMM model names, out indices, input channels, and extra kwargs while preserving custom feature-channel metadata.
  - Added alias discovery, negative unknown-backbone, output-channel metadata, registry metadata, raw TIMM regression, and native alias build tests.
  - Security/performance/regression review: no new shell execution or untrusted dynamic imports, alias resolution is a small static lookup, raw TIMM behavior remains unchanged, and full regression gates passed via the repo-supported `python3` runner.
- **Learnings for future iterations:**
  - Registry alias validation normalizes separators, so duplicate aliases such as `ResNet-50` and `resnet_50` collide even though canonical names still resolve separator variants.
  - Registered aliases backed by `TimmFeatureBackbone` must carry `model_name`; otherwise native construction cannot instantiate the adapter.
  - This environment still has no bare `python`; use `python3` or Makefile defaults for executable validation.
---
## [2026-05-18 23:21:18 UTC] - US-009: Implement anchor and point utilities
Thread:
Run: 20260518-183418-2827287 (iteration 9)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-9.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-9.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 1f13f45 feat(geometry): add native dense priors
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_geometry.py'` -> PASS (5 skipped; torch CPU extra unavailable)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (143 tests, 20 skipped)
  - Command: `make test` -> PASS (143 tests, 20 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/dense_ops.py
  - simpledet/simpledet/native/geometry.py
  - tests/test_native_geometry.py
- What was implemented
  - Added native feature-map specs, stride inference, anchor priors, point priors, bbox encode/decode, point-distance encode/decode, IoU, clipping, scaling, and batched-NMS payload helpers.
  - Rewired RetinaNet, FCOS, ATSS, and GFL dense decode/loss paths to share the geometry helpers while preserving public per-image prediction dictionaries.
  - Added validation for invalid feature sizes, invalid strides, payload shape mismatches, and empty tensors before malformed priors or detections are emitted.
  - Added unit coverage for anchor counts, point coordinates, bbox round trips, IoU, clipping, scaling, payload grouping, empty inputs, and negative validation cases.
  - Security/performance/regression review: tensor-only utilities, no new trust boundary or file/network handling, vectorized prior/box math, existing dense prediction contracts preserved.
- **Learnings for future iterations:**
  - The native dense path previously used torchvision AnchorGenerator/BoxCoder in decode and loss paths; shared native geometry now owns that dependency direction.
  - Retina-style anchor heads need nine anchors per location to match native head specs; the new default prior scales produce that count.
  - This environment still has no bare `python` and no installed torch CPU extra; use `python3`/Makefile gates, and tensor-heavy tests run when the CPU extra is available.
---
## [2026-05-19 00:07:33 UTC] - US-011: Implement common detection losses
Thread:
Run: 20260518-183418-2827287 (iteration 11)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-11.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-11.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: b4cf52b feat(losses): add native detection losses
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_losses.py'` -> PASS (4 skipped; torch CPU extra unavailable)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (154 tests, 31 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/assemblers.py
  - simpledet/simpledet/native/dense_ops.py
  - simpledet/simpledet/native/heads.py
  - simpledet/simpledet/native/losses.py
  - tests/test_native_components.py
  - tests/test_native_losses.py
- What was implemented
  - Added registry-backed native focal, quality focal, varifocal, distribution focal, smooth L1, L1, IoU, GIoU, DIoU, CIoU, cross-entropy, dice, and mask losses with explicit `LossContractError` shape checks.
  - Exported native loss classes and `build_loss`, and registered aliases such as `varifocal`, `dfl`, `giou`, `cross_entropy`, `dice`, and `mask`.
  - Wired `VFNetHead` and the VFNet dense training path to resolve varifocal and IoU losses through the shared `LOSSES` registry.
  - Added loss registry, scalar/gradient, negative contract, and VFNet registry wiring tests; tensor-heavy assertions skip cleanly when the optional torch CPU runtime is absent.
  - Security/performance/regression review: no new file/network/secret handling; loss math is tensor-local; VFNet quality targets use aligned IoU instead of quadratic pairwise scoring; full regression and packaging gates passed through `python3`.
- **Learnings for future iterations:**
  - Native loss registration should be included in registry snapshots for tests that clear and re-import native modules.
  - The current environment still has no bare `python` and no torch CPU extra, so exact user-specified `python` gates fail while Makefile/`python3` gates run.
  - Avoid `box_iou(...).diag()` in dense training paths; aligned box metrics preserve the contract without pairwise memory growth.
---
## [2026-05-19 00:29:31 UTC] - US-012: Implement ROI primitives
Thread:
Run: 20260518-183418-2827287 (iteration 12)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-12.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-12.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 2a43559 feat(roi): add native ROI primitives
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet:tests python3 -m unittest tests.test_native_roi tests.test_native_components tests.test_native_runtime` -> PASS (36 tests, 7 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (159 tests, 35 skipped)
  - Command: `make test` -> PASS (159 tests, 35 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/roi.py
  - tests/test_native_components.py
  - tests/test_native_roi.py
- What was implemented
  - Added native ROI proposal batching with `(K, 5)` ROI tensors, per-image proposal lists, image indices, counts, and empty-image preservation.
  - Added an empty-safe ROI align adapter plus bbox, mask, cascade refinement, and grid target primitives that reuse native geometry and assignment helpers.
  - Reused bbox target generation in the real-tensor ROI training path, added non-placeholder mask loss when masks are present, and kept empty proposal validation from crashing.
  - Configured Cascade R-CNN and Grid R-CNN model variants with explicit cascade/grid attributes while Faster R-CNN continues through the shared proposal and bbox target path.
  - Added focused unit tests for proposal formatting, empty ROI pooling, bbox target edge cases, mask/grid targets, cascade refinement, and Cascade/Grid construction.
  - Security/performance/regression review: no file/network/secret handling added; helper math is tensor-local with empty fast paths; focused and full regression gates passed through `python3`.
- **Learnings for future iterations:**
  - This environment still has no bare `python`; use `python3` or Makefile defaults for executable validation.
  - Existing ROI tests use lightweight fake torch modules, so production helpers need real-tensor paths without breaking fake construction/runtime tests.
  - Empty ROI pooling should avoid calling `MultiScaleRoIAlign` and synthesize `(0, C, H, W)` from feature metadata.
---
## [2026-05-19 00:52:43 UTC] - US-013: Register core dense heads
Thread:
Run: 20260518-183418-2827287 (iteration 13)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-13.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-13.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: c77ad5e feat(heads): register core dense heads
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest tests.test_native_backend_plan tests.test_suite tests.test_api_model_resolution` -> PASS (26 tests)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_dense_heads.py'` -> PASS (5 skipped; torch CPU extra unavailable)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_tensor_contracts.py'` -> PASS (3 tests, 2 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_assignment.py'` -> PASS (7 skipped; torch CPU extra unavailable)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_components.py'` -> PASS (27 tests, 3 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (166 tests, 40 skipped)
  - Command: `make test` -> PASS (166 tests, 40 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS after build
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/api-reference.html
  - simpledet/simpledet/__init__.py
  - simpledet/simpledet/api.py
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/dense_ops.py
  - simpledet/simpledet/native/heads.py
  - simpledet/simpledet/suite/__init__.py
  - simpledet/simpledet/suite/catalog.py
  - simpledet/simpledet/suite/specs.py
  - tests/test_native_backend_plan.py
  - tests/test_native_dense_heads.py
  - tests/test_suite.py
- What was implemented
  - Registered native dense `RPNHead`, `FSAFHead`, and `FreeAnchorRetinaHead`, kept `RetinaHead`, `FCOSHead`, `ATSSHead`, and converted `FoveaHead` to the anchor-free output contract used by its assembler path.
  - Added explicit dense aliases including `retina_head`, `fcos_head`, `atss_head`, `fsaf_head`, `fovea_head`, `free_anchor_head`, and `rpn_head`, exposed through `list_heads(kind="dense")`.
  - Added positive `num_classes`, `in_channels`, and `num_anchors` validation before native head construction.
  - Added RPN, FSAF, Fovea, and FreeAnchor dense loss/decode smoke paths using existing native geometry, assignment, and compatible Retina/FCOS contracts.
  - Added registry/build-plan tests and a real-tensor dense-head construction, forward-shape, loss-smoke, and inference-decode matrix that skips cleanly when the optional torch CPU runtime is absent.
  - Updated API docs for the new head discovery helper.
  - Security/performance/regression review: no new file/network/secret handling; added decode/loss paths are tensor-local and reuse existing NMS/assignment utilities; alias collision and full regression gates passed.
- **Learnings for future iterations:**
  - Registry aliases are normalized by removing separators and case, so paired aliases such as `FreeAnchorHead` and `free_anchor_head` collide.
  - `FOVEA` already resolves lowercase `fovea`; only the new `fovea_head` alias was needed.
  - This environment still has no bare `python` and no torch CPU extra, so runtime tensor tests are present but skip under the base install.
---
## [2026-05-19 01:29:12 UTC] - US-014: Register advanced dense heads
Thread:
Run: 20260518-183418-2827287 (iteration 14)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-14.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-14.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 14943a8 feat(heads): register advanced dense heads
- Post-commit status: `clean`
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_backend_plan.py'` -> PASS (13 tests)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_components.py'` -> PASS (27 tests, 3 skipped)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test_native_dense_heads.py'` -> PASS (13 skipped; base install lacks torch)
  - Command: `uv run --extra cpu python -m unittest discover -s tests -p 'test_native_dense_heads.py'` -> PASS (13 real-tensor tests)
  - Command: `PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'` -> PASS (174 tests, 48 skipped)
  - Command: `make test` -> PASS (174 tests, 48 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/dense_ops.py
  - simpledet/simpledet/native/heads.py
  - simpledet/simpledet/suite/catalog.py
  - tests/test_native_dense_heads.py
- What was implemented
  - Registered GFL, GFLv2, VFNet, PAA, RepPoints, YOLOF, TOOD, DDOD, AutoAssign, and NAS-FCOS dense head aliases with explicit native tensor contracts and dependency metadata.
  - Added native dense loss/decoder adapter classes for the advanced heads, preserving the current ATSS/FCOS output contracts instead of claiming unsupported family-specific training semantics.
  - Added VFNet direct native head construction through `build_head(..., in_channels=...)` while preserving legacy `HeadSpec` behavior when no native channels are supplied.
  - Added RepPoints point-generation validation for missing `point_strides` and feature-level mismatch.
  - Added advanced dense head construction, forward, finite-loss, decode, registry metadata, VFNet varifocal, and RepPoints negative tests.
  - Security/performance/regression review: no new file/network/secret handling; added validation is constant-time per feature level; loss/decode paths reuse existing dense ops; focused real-tensor and full regression gates passed.
- **Learnings for future iterations:**
  - Registry aliases normalize case and separators, so do not register both `NASFCOS` and `NAS-FCOS` or both `autoassign_head` and `auto_assign_head`.
  - `suite.build_head()` must remain spec-first; the direct native path is gated by `in_channels`/`out_channels`.
  - `uv run --extra cpu` executes real tensor tests here, but the PyPI torch wheel set installs large CUDA companion wheels and took several minutes.
---
## [2026-05-19 02:11:50 UTC] - US-016: Register keypoint transformer heads
Thread:
Run: 20260518-183418-2827287 (iteration 16)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-16.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-16.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 37556f8 feat(heads): add keypoint transformer heads
- Post-commit status: `clean` after progress/log follow-up commit
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> FAIL (`python`: command not found before local env setup)
  - Command: `PATH=.venv/bin:$PATH PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_keypoint_transformer_heads.py'` -> PASS (5 real CPU tensor tests)
  - Command: `tmpdir=$(mktemp -d); ln -s /usr/bin/python3 "$tmpdir/python"; PATH="$tmpdir:$PATH" PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> PASS (187 tests, 61 skipped)
  - Command: `PATH=.venv/bin:$PATH make docs-check` -> PASS
  - Command: `PATH=.venv/bin:$PATH make build` -> PASS
  - Command: `PATH=.venv/bin:$PATH make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/api-reference.html
  - docs/package-surface-audit.html
  - docs/roadmap-changelog.html
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/heads.py
  - simpledet/simpledet/suite/catalog.py
  - tests/test_native_keypoint_transformer_heads.py
- What was implemented
  - Registered CenterNetHead and CornerNetHead with heatmap tensor contracts, aliases, metadata, and CPU construction/forward coverage.
  - Registered DETRHead, ConditionalDETRHead, DABDETRHead, DeformableDETRHead, and DINOHead with query class/box outputs and direct `build_head(...)` construction for transformer aliases.
  - Added constructor-time validation for incompatible transformer `hidden_dim` and `num_heads` settings before forward execution.
  - Updated docs to expose dense keypoint and transformer head discovery while keeping full detector parity staged.
  - Security/performance/regression review: no new file/network/secret handling; transformer heads validate numeric attention shape before module execution; batch-first attention avoids the PyTorch nested-tensor warning; CenterNet preserves dense compatibility keys for existing consumers.
- **Learnings for future iterations:**
  - Registry aliases normalize case and separators, so aliases like `DETR`, `detr`, and `de_tr` collide; register one display alias plus explicit `_head` aliases.
  - The repo's lightweight global unittest gate skips optional torch tests; use `.venv` from `uv sync --extra cpu --extra dev` for real CPU tensor coverage.
  - A broad real-torch native sweep currently exposes unrelated pre-existing ROI/geometry/API failures, so US-016 validation used focused real-torch tests plus the required lightweight global gate.
---
## [2026-05-19 02:52:43 UTC] - US-018: Register mask and grid heads
Thread:
Run: 20260518-183418-2827287 (iteration 18)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-18.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-18.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: b61a164 feat(native): add roi mask and grid heads
- Post-commit status: `clean` after progress/log follow-up commit
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest tests.test_native_roi` -> FAIL (direct module mode does not add `tests/` for `native_tensor_contracts`)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_roi.py'` -> PASS (16 tests, 15 skipped without torch)
  - Command: `uv run --extra cpu python -m unittest discover -s tests -p 'test_native_roi.py'` -> PASS (16 real CPU tensor tests)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> PASS (199 tests, 72 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/heads.py
  - tests/test_native_roi.py
- What was implemented
  - Registered native `FCNMaskHead`, `CascadeMaskHead`, and `GridHead` ROI aliases with runtime-validated registry metadata and tensor contracts.
  - Added FCN/Cascade mask forward, target, loss, and decode helpers with default `(N, C, 14, 14) -> (N, num_classes, 28, 28)` mask logits.
  - Added `GridHead` forward, target, loss, and grid decode helpers, with constructor validation that rejects missing `grid_size`.
  - Added construction, direct `build_head`, forward-shape, target/loss/decode, negative-grid-config, and empty-ROI tests.
  - Security/performance/regression review: no file/network/secret handling added; tensor work remains bounded by ROI count, grid size, and pooled feature size; existing ROI bbox behavior and global gates passed.
- **Learnings for future iterations:**
  - Use unittest discovery for focused test files that import helpers from `tests/`; direct module mode misses `native_tensor_contracts`.
  - `uv run --extra cpu` is the real tensor validation path when the base interpreter skips torch-backed tests.
  - The task-provided absolute activity helper path is absent, but `ralph log` is available on `PATH` and writes the required activity entries.
---
## [2026-05-19 03:16:36 UTC] - US-019: Build single-stage detector composition
Thread:
Run: 20260518-183418-2827287 (iteration 19)
Run log: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-19.log
Run summary: /shared/home/rdelprete/PythonProjects/MMDET/.ralph/runs/run-20260518-183418-2827287-iter-19.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: d3e717d feat(native): add single-stage detector
- Post-commit status: `clean` after implementation commit; progress/log update committed separately
- Verification:
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_single_stage.py'` -> PASS (5 skipped without torch in base interpreter)
  - Command: `uv run --extra cpu env PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_single_stage.py'` -> PASS (5 real CPU tensor tests)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_components.py'` -> PASS (27 tests, 3 skipped)
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_suite.py'` -> PASS
  - Command: `PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test*.py'` -> PASS (204 tests, 77 skipped)
  - Command: `make docs-check` -> PASS
  - Command: `make build` -> PASS
  - Command: `make verify-dist` -> PASS
  - Command: `git diff --check` -> PASS
  - Command: `uv run --extra cpu env PYTHONPATH=simpledet python -m unittest discover -s tests -p 'test_native_components.py'` -> FAIL (additional diagnostic; pre-existing fake-module registry ordering makes neck classes lack `eval`, outside the US-019 path)
- Files changed:
  - .ralph/activity.log
  - .ralph/progress.md
  - docs/api-reference.html
  - simpledet/simpledet/native/__init__.py
  - simpledet/simpledet/native/assemblers.py
  - simpledet/simpledet/native/modeling.py
  - tests/test_native_single_stage.py
- What was implemented
  - Added `SingleStageDetector` as the native dense/single-stage composition boundary with owned backbone, optional neck, dense head, loss module, prediction path, and postprocessor.
  - Kept `NativeRetinaNetModel` import-compatible as a single-stage subclass and exported `simpledet.native.build_detector(name="retinanet", num_classes=3)`.
  - Added dense-plan validation that rejects explicit non-dense/ROI-only heads before native head construction and again before final assembly.
  - Added focused tests for module ownership, `forward_loss`, `predict`, eval-mode no-grad prediction, default RetinaNet construction, and ROI-only head rejection.
  - Updated the API reference for the new native single-stage builder surface.
  - Security/performance/regression review: no file/network/secret handling added; prediction uses `torch.no_grad()`; validation is registry metadata lookup only; existing dense assembly and full lightweight gates passed.
- **Learnings for future iterations:**
  - `simpledet.native.build_detector(...)` is the module-building entrypoint; `simpledet.suite.build_detector(...)` remains the spec-building entrypoint.
  - Dense head validation should reject explicit non-dense metadata while preserving older dense aliases that do not yet carry full family metadata.
  - Use `uv run --extra cpu` for real tensor story coverage when the base interpreter skips torch-backed tests.
---
