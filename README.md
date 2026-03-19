# SimpleDet

[![PyPI](https://img.shields.io/pypi/v/simpledet.svg)](https://pypi.org/project/simpledet/)
[![Python](https://img.shields.io/pypi/pyversions/simpledet.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SimpleDet is a custom object-detection toolkit with a native PyTorch Lightning execution path for training, inference, evaluation, and experiment packaging.

The maintained native catalog currently supports `retinanet`, `retina`, `fcos`, `atss`, `gfl`, `vfnet`, `fovea`, `foveabox`, `reppoints`, `yolof`, `centernet`, `faster_rcnn`, `mask_rcnn`, `grid_rcnn`, and `cascade_rcnn`.

## Installation

Base package only:

```bash
python -m pip install simpledet
```

Supported CPU runtime for the public detection APIs:

```bash
python -m pip install "simpledet[cpu]"
```

Optional workflow extras:

```bash
python -m pip install "simpledet[geo,plots]"
```

From source while iterating locally:

```bash
git clone https://github.com/sirbastiano/SimpleDet.git
cd SimpleDet
python -m pip install -e ".[cpu]"
```

Sanity checks:

```bash
python -m simpledet --version
python -m simpledet --check-runtime
```

## Supported Publish Matrix

The package is published as a pure-Python wheel. The supported runtime contract is:

- Linux, macOS, and Windows
- Python 3.10, 3.11, and 3.12
- CPU-only dependency stack for `simpledet[cpu]`
- wheel-only dependency resolution for release verification

For a supported OS/Python pair, all dependencies must install from wheels; otherwise that matrix entry is not supported.

## Repository Shortcuts

```bash
make venv
make sync
make sync-cpu
make build
make check
```

## Package Layout

- `simpledet/`: installable package
- `simpledet/native/`: native PyTorch Lightning runtime, components, and execution helpers
- `simpledet/suite/`: detector specs, planners, and authoring helpers
- `tests/`: unit and packaging checks
- `docs/`: project documentation site sources

## Publishing

Local release verification:

```bash
python3 -m build
python3 -m twine check dist/*
PYTHONPATH=simpledet python3 -m unittest discover -s tests -p 'test*.py'
```

The CI release workflow builds the sdist/wheel, runs the test suite, and verifies wheel-only installation of the built artifacts on Linux, macOS, and Windows for Python 3.10-3.12.

## License

This project is licensed under the [MIT License](LICENSE).
