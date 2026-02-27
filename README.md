# SimpleDet

[![PyPI](https://img.shields.io/pypi/v/simpledet.svg)](https://pypi.org/project/simpledet/)
[![Python](https://img.shields.io/pypi/pyversions/simpledet.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Install](https://img.shields.io/badge/Install-Quick%20Start-2ea44f?logo=python&logoColor=white)](#installation)
[![Use](https://img.shields.io/badge/Usage-Training%20%26%20Inference-0a66c2?logo=python)](#quick-start)
[![Publish](https://img.shields.io/badge/Release-0.1.0-brightgreen?logo=pypi&logoColor=white)](https://pypi.org/project/simpledet/0.1.0/)

## Overview

SimpleDet is a custom object-detection toolkit built on top of OpenMMLab components. It is designed for satellite and related computer-vision workloads where you need reproducible workflows for:

- training and fine-tuning detection models
- inference and evaluation
- experiment-friendly project-level packaging

## Installation

From source:

```bash
git clone https://github.com/sirbastiano/MDet.git
cd MDet
python -m pip install .
```

For OpenMMLab runtime extras:

```bash
python -m pip install ".[openmmlab]"
```

Install directly from PyPI:

```bash
python -m pip install simpledet
```

## Quick Start

Install in editable mode while iterating:

```bash
python -m pip install -e .
```

Sanity check:

```bash
python -m simpledet --version
```

### Core entry points

- `simpledet.train`: training entry points
- `simpledet.detect`: inference entry points
- `simpledet.evaluate`: evaluation utilities
- `simpledet.cli`: command-line checks and diagnostics

### Repository structure

- `simpledet/`: installable Python package (package code and model utilities)
- `MyConfigs/`: dataset/model configuration sources used in experiments
- `studies/`, `notebooks/`: exploratory analyses and research workflows
- `runscripts/`: executable training/inference launch scripts
- `build/`: generated build output (not intended for direct edits)

## Contributing

1. Create a branch.
2. Make your changes.
3. Open a PR with a clear summary.

## License

This project is licensed under the [MIT License](LICENSE).
