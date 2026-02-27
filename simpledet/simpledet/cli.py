"""Command-line interface for the simpledet package."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simpledet",
        description="SimpleDet package bootstrap and diagnostics utility.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    parser.add_argument(
        "--check-openmmlab",
        action="store_true",
        help=(
            "Validate optional OpenMMLab runtime dependency resolution used by "
            "simpledet.api."
        ),
    )
    return parser


def _check_openmmlab() -> int:
    """Return non-zero when runtime dependencies are not installed."""
    try:
        from . import api
        api._ensure_openmmlab()
    except ModuleNotFoundError as exc:
        dependency = getattr(exc, "name", "")
        if dependency in {"torch", "numpy", "torchvision", "mmengine", "mmdet"}:
            print(
                "simpledet.api requires optional runtime dependencies. "
                "Install required dependencies before running this check."
            )
            if dependency:
                print(f"Missing dependency: {dependency}")
        else:
            print(str(exc))
        return 1

    print("Optional OpenMMLab runtime stack is available.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """Run CLI and return process exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(__version__)
        return 0

    if args.check_openmmlab:
        return _check_openmmlab()

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
