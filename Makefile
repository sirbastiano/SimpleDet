SHELL := /usr/bin/env bash
PYTHON ?= python3
PACKAGE ?= simpledet
UV ?= uv
UVX ?= uvx
UV_SYNC_FLAGS ?=

.PHONY: help venv sync sync-cpu install install-runtime install-editable build sdist wheel verify-dist check publish publish-test test clean docs-check docs-audit docs-verify

help:
	@echo "Targets:"
	@echo "  make venv           Create the project virtual environment with uv"
	@echo "  make sync           Create/update .venv and install core dependencies"
	@echo "  make sync-cpu       Create/update .venv and install the supported CPU runtime"
	@echo "  make install        Install project from source"
	@echo "  make install-runtime Install the supported CPU runtime from source"
	@echo "  make install-editable Install editable package with the CPU runtime"
	@echo "  make bootstrap      Ensure build/publishing tools are up to date"
	@echo "  make build          Build source distribution and wheel"
	@echo "  make sdist           Build source distribution only"
	@echo "  make wheel           Build wheel only"
	@echo "  make docs-audit      Run comprehensive docs audit (links + HTML health checks)"
	@echo "  make docs-check      Validate docs files and local links"
	@echo "  make verify-dist     Audit built wheel and sdist contents"
	@echo "  make check           Run tests, docs checks, artifact audits, and twine metadata checks"
	@echo "  make publish         Run bootstrap, build, check, then upload to PyPI"
	@echo "  make publish-test    Run bootstrap, build, check, then upload to TestPyPI"
	@echo "  make clean           Remove build artifacts"

venv:
	$(UV) venv

sync:
	$(UV) sync $(UV_SYNC_FLAGS)

sync-cpu:
	$(UV) sync --extra cpu $(UV_SYNC_FLAGS)

install:
	$(PYTHON) -m pip install .

install-runtime:
	$(PYTHON) -m pip install ".[cpu]"

install-editable:
	$(PYTHON) -m pip install -e ".[cpu]"

bootstrap:
	$(PYTHON) -m pip install -U pip build

build:
	$(PYTHON) -m build

sdist:
	$(PYTHON) -m build --sdist

wheel:
	$(PYTHON) -m build --wheel

verify-dist:
	$(UVX) check-wheel-contents --ignore W002,W004 dist/*.whl
	PYTHONPATH=simpledet $(PYTHON) -m unittest tests.test_packaging

check: test docs-check verify-dist
	$(UVX) twine check dist/*

docs-check: docs-audit
	@echo "Docs check complete."

docs-audit: docs-verify
	$(PYTHON) scripts/docs_audit.py
	@echo "Docs audit passed."

docs-verify:
	$(PYTHON) scripts/verify_docs.py

publish: bootstrap build check
	$(UVX) twine upload dist/*

publish-test: bootstrap build check
	$(UVX) twine upload --repository testpypi dist/*

test:
	PYTHONPATH=simpledet $(PYTHON) -m unittest discover -s tests -p 'test*.py'

clean:
	rm -rf dist build .venv
