# SimpleDet

<p align="center">
  <img src="https://raw.githubusercontent.com/sirbastiano/SimpleDet/main/assets/simpledet-logo.svg" alt="SimpleDet logo" width="620" />
</p>

<p align="center">
  <a href="https://pypi.org/project/simpledet/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/PyPI-0A66C2?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://www.python.org/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python versions">
  </a>
  <a href="LICENSE" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/License-MIT-0D9488?style=for-the-badge" alt="MIT license">
  </a>
  <a href="https://github.com/sirbastiano/SimpleDet" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/GitHub-0f172a?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<p align="center">
  <a href="#quick-installation">
    <img src="https://img.shields.io/badge/Start%20in%20a%20Minute-0f766e?style=for-the-badge" alt="Get started">
  </a>
  <a href="#documentation-and-links">
    <img src="https://img.shields.io/badge/Explore%20Docs-2563eb?style=for-the-badge&logo=readme&logoColor=white" alt="Documentation">
  </a>
  <a href="https://pypi.org/project/simpledet/#history" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Release%20History-7c3aed?style=for-the-badge&logo=pypi&logoColor=white" alt="Release history">
  </a>
</p>

<p align="center"><strong>Native object-detection toolkit with registry-first model composition and a PyTorch Lightning runtime.</strong></p>

<hr />

## Overview

SimpleDet is designed to make detector development predictable and production-friendly:

- Build detector definitions with `simpledet.suite` (Encoder, Neck, Head, Decoder, Detector specs).
- Compile runtime plans that adapt layer dimensions automatically from actual encoder outputs.
- Run training, inference, and evaluation through native project-oriented workflows.
- Keep experimentation reproducible with explicit CLI and config-based project execution.

## Professional capabilities

| Area | What you get |
| --- | --- |
| Model authoring | Declarative detector specs + registry-backed component assembly |
| Execution | Native PyTorch Lightning training, inference, and evaluation path |
| Adaptation | Automatic channel/shape alignment during model compilation |
| Packaging | Project-level run orchestration and validation commands |
| Catalog support | `retinanet`, `retina`, `fcos`, `atss`, `gfl`, `vfnet`, `fovea`, `foveabox`, `reppoints`, `yolof`, `centernet`, `rpn`, `fast_rcnn`, `faster_rcnn`, `mask_rcnn`, `cascade_rcnn`, `cascade_mask_rcnn`, `grid_rcnn`, `libra_rcnn`, `double_head_rcnn`, `dynamic_rcnn` |

## Quick installation

Choose the smallest install path that matches the workflow you want to run. The base package is lightweight; runtime and tooling dependencies live behind extras so imports stay fast and environments stay focused.

| Workflow | Command |
| --- | --- |
| Base package and CLI metadata | `python -m pip install simpledet` |
| CPU training, inference, and evaluation runtime | `python -m pip install 'simpledet[cpu]'` |
| TIMM encoder catalog support | `python -m pip install 'simpledet[timm]'` |
| Documentation workflow setup | `python -m pip install 'simpledet[docs]'` |
| Geospatial dataset helpers | `python -m pip install 'simpledet[geo]'` |
| Plotting utilities | `python -m pip install 'simpledet[plots]'` |
| Geospatial plus plotting workflows | `python -m pip install 'simpledet[geo,plots]'` |
| Development tooling for build, lint, and release checks | `python -m pip install 'simpledet[dev]'` |
| Local development with CPU runtime, TIMM validation, and release tooling | `python -m pip install -e '.[cpu,timm,dev]'` |

```bash
python -m pip install simpledet
```

```bash
python -m pip install 'simpledet[cpu]'
```

```bash
python -m pip install 'simpledet[timm]'
```

```bash
python -m pip install 'simpledet[docs]'
```

```bash
python -m pip install 'simpledet[geo,plots]'
```

```bash
python -m pip install 'simpledet[dev]'
```

```bash
git clone https://github.com/sirbastiano/SimpleDet.git
cd SimpleDet
python -m pip install -e '.[cpu,timm,dev]'
```

```bash
python -m simpledet --version
python -m simpledet --check-runtime
```

Troubleshooting starts with the installed extra set: use `simpledet[cpu]` before runtime imports, combine `cpu,timm` when validating TIMM-backed detector construction, and prefer a fresh virtual environment if wheel resolution leaves incompatible PyTorch or TorchVision builds. SimpleDet does not require MMDetection, MMEngine, or MMCV as runtime dependencies.

## Supported environments

- Platforms: Linux, macOS, Windows
- Python: 3.10, 3.11, 3.12
- Runtime support: CPU stack via `simpledet[cpu]`, TIMM encoders via `simpledet[timm]`, geospatial helpers via `simpledet[geo]`, plotting via `simpledet[plots]`, docs workflow setup via `simpledet[docs]`, and development tooling via `simpledet[dev]` (wheel-compatible dependencies are required for supported OS/Python pairs)

## Core workflow

- `simpledet.suite`: author specs, compose detector graphs, and compile native plans.
- `simpledet.native`: run native PyTorch Lightning model loops.
- `simpledet.detectors`: compatibility helpers for lightweight flows.
- `simpledet._model_resolution`: runtime shape and channel adaptation internals.

## Quick references

### Documentation and links

- [Installation guide](docs/installation.html)
- [Quickstart](docs/quickstart.html)
- [CLI reference](docs/cli-reference.html)
- [Core concepts](docs/core-concepts.html)
- Executed showcase notebook: `notebooks/Tools/simpledet_showcase.ipynb`
- [Roadmap / changelog](docs/roadmap-changelog.html)

### Repository workflow commands

```bash
make venv
make sync
make sync-cpu
make build
make docs-audit
make check
make release-ready
make test-cpu-smoke
```

`make release-ready` is the final handoff gate. It rebuilds the package, runs
the unit, docs, and distribution checks, then audits release claims so detector
and head counts stay above 31 and docs stay aligned with package extras.

`make test-cpu-smoke` is the optional slow CPU workflow gate for checkouts with
`simpledet[cpu]` installed. It runs the tiny project-config smoke path through
build, one training epoch, test/evaluation, inference, manifest writing, and
prediction export.

## Repository structure

- `simpledet/`: Python source root used by packaging
- `simpledet/simpledet/`: installable package
- `simpledet/simpledet/native/`: native Lightning runtime and model execution components
- `simpledet/simpledet/suite/`: detector specs, builders, and planning helpers
- `simpledet/simpledet/_model_resolution.py`: model-adaptation helpers
- `simpledet/simpledet/detectors/`: compatibility path for lightweight training/eval flows
- `docs/`: static documentation site
- `scripts/`: repository verification and docs audit helpers
- `tests/`: unit and packaging checks

## Publishing

```bash
python3 -m build
python3 -m twine check dist/*
PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'
```

## License

SimpleDet is released under the [MIT License](LICENSE).
