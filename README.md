# SimpleDet

[![PyPI](https://img.shields.io/pypi/v/simpledet.svg)](https://pypi.org/project/simpledet/)
[![Python](https://img.shields.io/pypi/pyversions/simpledet.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SimpleDet is a custom object-detection toolkit built around MMDetection/OpenMMLab workflows for training, inference, evaluation, and experiment packaging.

## Installation

Base package only:

```bash
python -m pip install simpledet
```

Supported CPU runtime for the public detection APIs:

```bash
python -m pip install "simpledet[cpu]"
```

Backward-compatible alias:

```bash
python -m pip install "simpledet[openmmlab]"
```

Optional workflow extras:

```bash
python -m pip install "simpledet[geo,plots]"
```

From source while iterating locally:

```bash
git clone https://github.com/sirbastiano/MDet.git
cd MDet
python -m pip install -e ".[cpu]"
```

Sanity checks:

```bash
python -m simpledet --version
python -m simpledet --check-openmmlab
```

## Supported Publish Matrix

The package is published as a pure-Python wheel. The supported runtime contract is:

- Linux, macOS, and Windows
- Python 3.10, 3.11, and 3.12
- CPU-only dependency stack for `simpledet[cpu]`
- wheel-only dependency resolution for release verification

If a dependency cannot be installed from wheels on a claimed platform/Python pair, that matrix entry is not considered supported.

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
- `simpledet/src/`: packaged configs and custom model components
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
