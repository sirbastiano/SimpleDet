# PLANS.md

## Goal

Replace the current `mmdet` / `mmengine` runtime backend with a native PyTorch Lightning backend while preserving SimpleDet's high-level workflow quality:
- detector authoring through `simpledet.suite`
- project-layout-oriented training / inference / evaluation
- direct execution and config-driven execution
- explicit custom network component extensibility
- clear checkpoint, metrics, and manifest artifacts

## Scope / non-goals

### In scope
- backend-neutral rewrite of the detector suite IR
- native PyTorch / Lightning model assembly and runtime
- native dataset, transform, and datamodule layer
- native training, inference, and evaluation orchestration
- native custom-component registration and import mechanism
- CLI and docs migration to the new backend
- removal of `mmdet` / `mmengine` from the primary package runtime path

### Non-goals
- preserving MMDetection config dictionaries as the primary internal source of truth
- preserving MMEngine runner semantics
- preserving MMDetection registry semantics as the main extension model
- achieving one-shot feature parity for every legacy architecture before shipping any native milestone
- maintaining old handwritten MMDet config files as first-class authoring artifacts

## Invariants and contracts to preserve

### Public API invariants
- `build_detector(...)` remains the main high-level model authoring entrypoint
- `create_project_pipeline(...)`, `create_project_inference_pipeline(...)`, `run_training(...)`, `run_inference(...)`, and `run_evaluation(...)` remain first-class workflows
- non-config execution remains supported
- config-driven execution remains supported
- custom component support remains explicit and documented

### Behavioral invariants
- project layout conventions remain stable by default:
  - `imgs/`
  - `Annotations/train_annotations.json`
  - `Annotations/val_annotations.json`
  - `Annotations/test_annotations.json`
- checkpoints, metrics, manifests, and logs remain operationally visible artifacts
- backend internals should not depend on hidden global registries or implicit runtime mutation

### Migration invariants
- the migration is staged, not a destructive in-place rewrite
- the new backend must be able to coexist with the legacy backend until parity gates are met
- every preserved public API must either:
  - keep behavior, or
  - declare a documented compatibility change
- no new native family is added on top of a backend core that fails clean-code gates
- native model parts must have explicit ownership boundaries:
  - backbone composition
  - neck composition
  - head composition
  - target assignment / loss
  - decoding / post-processing
  - runtime orchestration
- trainable native components must be real `torch.nn.Module` objects; wrapper-only parameter aggregation is not acceptable as an end state

## Files / layers likely to change

### Existing files that will be heavily modified
- `simpledet/simpledet/api.py`
- `simpledet/simpledet/__init__.py`
- `simpledet/simpledet/cli.py`
- `simpledet/simpledet/suite/specs.py`
- `simpledet/simpledet/suite/catalog.py`
- `simpledet/simpledet/suite/compiler.py`
- `pyproject.toml`
- `setup.py`

### Existing files or directories likely to be deprecated, isolated, or removed
- `simpledet/simpledet/_model_resolution.py`
- `simpledet/simpledet/src/base_config.py`
- `simpledet/simpledet/src/configs/**`
- `simpledet/simpledet/src/custom_components/**`
- legacy MMDet-specific runtime logic embedded in `simpledet/simpledet/api.py`

### New modules and directories likely to be added
- `simpledet/simpledet/models/`
- `simpledet/simpledet/data/`
- `simpledet/simpledet/engine/`
- `simpledet/simpledet/runtime/`
- `simpledet/simpledet/extensions/`
- `simpledet/simpledet/legacy_mmdet/` or equivalent isolation layer if coexistence is needed
- `simpledet/simpledet/native/`

### Tests likely to be rewritten or split
- `tests/test_suite.py`
- `tests/test_public_api.py`
- `tests/test_cli.py`
- new native runtime tests
- new model composition tests
- new datamodule / transform tests

### Docs likely to change materially
- `README.md`
- `docs/quickstart.html`
- `docs/api-reference.html`
- `docs/core-concepts.html`
- `docs/training.html`
- `docs/inference.html`
- `docs/evaluation.html`
- `docs/cli-reference.html`

## Ordered steps

1. Define the v2 migration contract
   - Freeze the target public API contract for the native backend.
   - Mark every current public symbol as:
     - preserved
     - adapted
     - deprecated
     - removed
   - Decide the compatibility policy for:
     - old checkpoints
     - old config files
     - old custom component modules

2. Introduce a backend-neutral suite IR
   - Refactor `DetectorSpec`, `EncoderSpec`, `NeckSpec`, `HeadSpec`, and `DecoderSpec` so they describe model intent rather than MMDet config fragments.
   - Change `compile_detector_spec(...)` from “compile to MMDet dict” into “compile to native structured build plan”.
   - Keep current builder ergonomics where possible.

3. Build a native extension and registration mechanism
   - Replace MMDet `custom_imports` and registry assumptions with SimpleDet-native factories or registries.
   - Define explicit registration for:
     - encoders
     - necks
     - heads
     - decoders
     - detector assemblers
   - Preserve explicit import-based extension points.

4. Refactor the native dense core to pass architecture hygiene gates
   - Stop adding detector families until the current dense path is split into explicit modules.
   - Require:
     - real `nn.Module` ownership for backbone / neck / head wrappers
     - a dedicated dense loss / target-assignment module
     - a dedicated dense decode / post-processing module
     - a composed model class that only orchestrates these units
   - Replace protocol-driven fallback branches with an explicit tensor-batch input contract.
   - Keep test doubles in tests by patching loss / decode collaborators rather than routing through production fallback behavior.
   - Ensure the Lightning step uses one authoritative total loss, not recomposed sums of duplicate values.

5. Implement native model composition
   - Add native PyTorch modules under `simpledet.simpledet.models` or `simpledet.simpledet/native/` with clear submodule boundaries.
   - Move detector construction behind a detector-assembler layer so new architectures are registered instead of added as hardcoded branches.
   - Expose a component-first custom-detector path so new models can be created from encoder / neck / head / decoder specs plus a registered assembler.
   - Start with a constrained architecture set:
     - one dense detector
     - one ROI detector
     - one transformer detector
   - The first ROI milestone may use a native `torchvision` ROI detector factory behind the Lightning runtime while explicit ROI component composition is still being built.
   - The first transformer milestone may use a minimal native transformer composition with explicit backbone / neck / decoder / post-processing ownership before full parity work.
   - Do not claim full architecture parity until each family has explicit implementation coverage.
   - Dense-family expansion is blocked until step 4 is complete.

6. Implement a native data layer
   - Add datasets, transforms, collate logic, and a Lightning `DataModule`.
   - Port current project-layout assumptions into this layer.
   - Replace MMDet pipeline mutation with explicit transform assembly.

7. Implement the Lightning runtime engine
   - Add `LightningModule` wrappers for training / validation / test.
   - Implement optimizer and scheduler creation natively.
   - Implement checkpointing, metrics, logging, and artifact emission.
   - Define and document the native artifact contract.
   - Keep model math out of the Lightning wrapper; it should orchestrate, not own detection policy.

8. Rewrite the pipeline layer on top of the native backend
   - Rebuild `ObjectDetectionPipeline` around the native engine and data layer.
   - Preserve high-level helpers:
     - `create_project_pipeline(...)`
     - `create_project_inference_pipeline(...)`
     - `run_training(...)`
     - `run_inference(...)`
     - `run_evaluation(...)`
   - Remove backend coupling from the API layer.
   - Route to native only for families that meet the architecture and validation gates.

9. Rewrite config-driven execution
   - Keep `ProjectConfig` and direct/config workflows.
   - Change config loading to target native runtime structures rather than MMDet runtime structures.
   - Define migration handling for old config payloads if compatibility is required.

10. Rewrite the CLI
   - Update direct-run commands to use the native backend.
   - Keep discovery, help, config execution, and direct execution flows.
   - Ensure CLI output remains structured and operational.

11. Isolate, deprecate, and remove the legacy MMDet backend
   - Move any still-needed legacy backend code behind an explicit compatibility layer.
   - Remove `mmdet` / `mmengine` from default runtime dependencies once native parity gates are met.
   - Remove or archive MMDet-specific configs and custom component modules once migration is complete.

12. Final parity and cleanup pass
   - Update notebooks, docs, packaging metadata, and examples.
   - Remove stale MMDet references from primary user-facing docs.
   - Finalize deprecation notes and release migration guidance.

## Clean-code gating rules for the native backend

Before adding a new detector family or cutting a public API over to native by default, the current native slice must satisfy all of these:
- wrappers are real `nn.Module`s rather than plain Python holders around trainable submodules
- training loss is computed in one place and returned once
- decode / NMS logic is separated from core model composition
- target assignment is not embedded inside the top-level model class
- runtime wrappers do not manually control train/eval mode unless justified
- test-only fallback behavior does not drive production control flow

If any of the above are false, family expansion is blocked and the next implementation step must be cleanup, not feature growth.

## Public API deprecation map

### Preserve with native implementation
- `build_detector(...)`
- `create_project_pipeline(...)`
- `create_project_inference_pipeline(...)`
- `create_training_pipeline(...)`
- `create_inference_pipeline(...)`
- `run_training(...)`
- `run_inference(...)`
- `run_evaluation(...)`
- `ProjectConfig`
- `load_project_config(...)`
- `run_project(...)`

### Adapt
- `compile_detector_spec(...)`
  - current meaning: MMDet config compiler
  - target meaning: native model build-plan compiler
- `ObjectDetectionPipeline`
  - current meaning: MMEngine/MMDet-backed operational pipeline
  - target meaning: Lightning-backed operational pipeline

### Deprecate
- direct reliance on raw MMDet-shaped `model_cfg` dictionaries as the preferred advanced path
- MMDet-style `custom_imports` as a public extension mechanism
- MMDet-specific config files under `simpledet/simpledet/src/configs/**` as first-class runtime artifacts

### Remove or isolate
- `_model_resolution.py` MMDet patching logic
- MMDet registry-driven custom component assumptions
- MMEngine runner-specific orchestration

## File-by-file migration map

### `simpledet/simpledet/suite/specs.py`
- convert spec classes to backend-neutral intent models
- remove MMDet-shape assumptions
- define native extension metadata for component registration

### `simpledet/simpledet/suite/catalog.py`
- keep public builders stable
- route builders into native component registries and defaults
- provide native custom-component builders

### `simpledet/simpledet/suite/compiler.py`
- replace MMDet config compilation with native build-plan compilation
- stop importing `simpledet.simpledet.src.models` as the source of truth

### `simpledet/simpledet/api.py`
- split current all-in-one pipeline responsibilities into runtime/data/engine integrations
- remove MMEngine/MMDet imports
- preserve high-level pipeline and direct-run helpers
- keep backend routing policy shallow; do not embed family-specific model math here

### `simpledet/simpledet/native/`
- split into explicit submodules for:
  - backbones
  - necks
  - heads
  - loss / assignment
  - decode / post-processing
  - runtime orchestration
- avoid letting `modeling.py` become the new monolith

### `simpledet/simpledet/cli.py`
- keep the current user-oriented command surface
- replace backend implementation with native runtime calls

### `simpledet/simpledet/_model_resolution.py`
- either delete or reduce to compatibility shims during transition

### `simpledet/simpledet/src/custom_components/**`
- migrate reusable ideas into native PyTorch modules or extension hooks
- isolate MMDet-only components under a legacy namespace if short-term coexistence is needed

### `tests/test_suite.py`
- rewrite compiler expectations around native build plans instead of MMDet dict fields

### `tests/test_public_api.py`
- preserve public API coverage while changing backend expectations

### `tests/test_cli.py`
- preserve command surface tests
- update assertions for native backend behavior and outputs

## Validation plan

### Phase-by-phase validation

#### Phase 1: suite IR
- unit tests for builder outputs
- unit tests for compiled build-plan structure
- explicit tests for custom component registration metadata

#### Phase 2: native models
- construction tests
- forward-pass shape tests
- loss computation smoke tests
- explicit clean-code gate review before adding the next detector family
- optimizer parameter coverage tests to ensure all trainable modules are registered through `nn.Module`

#### Phase 3: data layer
- dataset parsing tests
- transform correctness tests
- datamodule split tests

#### Phase 4: engine
- Lightning training smoke test
- validation/test loop smoke tests
- optimizer/scheduler integration tests
- checkpoint and metrics artifact tests
- assert the authoritative optimization loss matches the model-reported total loss

#### Phase 5: runtime / pipeline
- pipeline build / train / test smoke tests
- direct execution helper tests
- config-driven execution tests

#### Phase 6: CLI
- direct CLI train / infer / eval tests
- config CLI tests
- discovery/help command tests

### Final acceptance gates
- package installs and runs without `mmdet` or `mmengine`
- primary public API flows execute on Lightning
- custom components work through native extension hooks
- docs and notebooks no longer require MMDet/MMEngine as the primary path
- legacy backend is either removed or isolated behind an explicit compatibility boundary
- native backend passes clean-code-gates at `PASS` or explicit `WARN` with documented debt before it becomes the default for a family

## Rollback / recovery notes

- Do not perform this rewrite as a single in-place cutover.
- Keep the legacy backend isolated during migration, preferably under a clearly named compatibility namespace.
- Only switch the default backend after the native backend meets the acceptance gates for the selected supported architecture set.
- If a migration phase fails, rollback should happen by:
  - reverting that phase’s branch or commit series
  - preserving the legacy backend as the default path
- No stateful migration is required at the database level, but checkpoint and config compatibility need explicit rollback notes in release docs.

## Risks / blockers

### High-risk blockers
- reproducing enough detector-family coverage to replace current user workflows
- replacing evaluator behavior currently delegated to MMDet
- checkpoint compatibility expectations
- migrating custom components away from MMDet registries without breaking extension workflows
- allowing the current native dense path to grow into a new monolith before the architecture boundaries are cleaned up

### Medium-risk blockers
- transform and dataset behavior drift
- optimizer / scheduler drift
- artifact naming or metric format drift

### Low-risk blockers
- CLI migration
- config loader migration
- doc and notebook rewrite

### Assumptions that could invalidate the plan
- if users depend heavily on direct MMDet config authoring, deprecation scope becomes larger
- if full architecture parity is required before any rollout, timeline and validation scope expand materially
- if legacy checkpoints must remain natively loadable without translation, model interface constraints tighten
- if native family growth proceeds before the dense core is modularized, later ROI/transformer work will require rework rather than extension

## Definition of done

- `mmdet` and `mmengine` are no longer required for the main package runtime
- `simpledet.suite` compiles to a native build plan, not an MMDet config
- the main training / inference / evaluation workflows run through Lightning
- direct execution and config-driven execution remain first-class
- custom network components are supported through a native extension mechanism
- the CLI and docs reflect the native backend as the primary workflow
- tests cover suite, models, data, runtime, CLI, and docs at the appropriate depth
- any remaining legacy backend is clearly isolated, deprecated, and not the default
