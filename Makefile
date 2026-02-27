SHELL := /usr/bin/env bash
PYTHON ?= python3
PACKAGE ?= simpledet

.PHONY: help install install-editable build sdist wheel check publish publish-test clean

help:
	@echo "Targets:"
	@echo "  make install        Install project from source"
	@echo "  make install-editable Install with optional OpenMMLab extras"
	@echo "  make bootstrap      Ensure build/publishing tools are up to date"
	@echo "  make build          Build source distribution and wheel"
	@echo "  make sdist           Build source distribution only"
	@echo "  make wheel           Build wheel only"
	@echo "  make check           Validate sdist/wheel metadata"
	@echo "  make publish         Run bootstrap, build, check, then upload to PyPI"
	@echo "  make publish-test    Run bootstrap, build, check, then upload to TestPyPI"
	@echo "  make clean           Remove build artifacts"

install:
	$(PYTHON) -m pip install .

install-editable:
	$(PYTHON) -m pip install -e ".[openmmlab]"

bootstrap:
	$(PYTHON) -m pip install -U pip build twine

build:
	$(PYTHON) -m build

sdist:
	$(PYTHON) -m build --sdist

wheel:
	$(PYTHON) -m build --wheel

check:
	$(PYTHON) -m twine check dist/*

publish: bootstrap build check
	$(PYTHON) -m twine upload dist/*

publish-test: bootstrap build check
	$(PYTHON) -m twine upload --repository testpypi dist/*

clean:
	rm -rf dist build .venv
