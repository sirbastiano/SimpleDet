SHELL := /usr/bin/env bash
PYTHON ?= python3
PACKAGE ?= simpledet

.PHONY: help install install-editable build sdist wheel check publish publish-test test clean docs-check docs-verify

help:
	@echo "Targets:"
	@echo "  make install        Install project from source"
	@echo "  make install-editable Install with optional OpenMMLab extras"
	@echo "  make bootstrap      Ensure build/publishing tools are up to date"
	@echo "  make build          Build source distribution and wheel"
	@echo "  make sdist           Build source distribution only"
	@echo "  make wheel           Build wheel only"
	@echo "  make docs-check      Validate docs files and local links"
	@echo "  make check           Run tests, docs checks, and twine metadata checks"
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

check: test docs-check
	$(PYTHON) -m twine check dist/*

docs-check: docs-verify
	@echo "Docs check complete."

docs-verify:
	$(PYTHON) scripts/verify_docs.py

publish: bootstrap build check
	$(PYTHON) -m twine upload dist/*

publish-test: bootstrap build check
	$(PYTHON) -m twine upload --repository testpypi dist/*

test:
	PYTHONPATH=simpledet $(PYTHON) -m unittest discover -s tests -p 'test*.py'

clean:
	rm -rf dist build .venv
