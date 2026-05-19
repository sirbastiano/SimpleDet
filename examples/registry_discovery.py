"""Inspect detector, backbone, and head registry entries."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from simpledet import list_backbones, list_detectors, list_heads


def discovery_summary(pattern: str | None = None) -> dict:
    return {
        "detectors": list_detectors(pattern=pattern),
        "backbones": list_backbones(pattern=pattern),
        "heads": list_heads(pattern=pattern),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print registry-backed SimpleDet component names.")
    parser.add_argument("--pattern", help="Case-insensitive substring filter.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(discovery_summary(pattern=args.pattern), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
