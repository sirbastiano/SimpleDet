"""CLI adapter for detector workflows."""

from __future__ import annotations

from collections.abc import Iterable

from simpledet.cli import main as _package_main


def main(argv: Iterable[str] | None = None) -> int:
    """Run the package CLI through the public detectors namespace."""
    return _package_main(argv)
