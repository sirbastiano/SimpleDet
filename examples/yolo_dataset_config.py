"""Create and optionally inspect a YOLO-format dataset config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from simpledet.detectors.data import load_dataset


DEFAULT_SAMPLE_ROOT = Path(__file__).resolve().parent / "sample_data" / "yolo"


def build_yolo_dataset_config(dataset_root: Path, *, classes_file: Path | None = None) -> dict:
    payload = {
        "format": "yolo",
        "root": str(dataset_root),
        "images": str(dataset_root / "images"),
        "labels": str(dataset_root / "labels"),
    }
    if classes_file is not None:
        payload["classes_file"] = str(classes_file)
    return payload


def require_dataset_root(path: Path) -> Path:
    root = path.expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Sample YOLO dataset not found: {root}. "
            "Pass --dataset-root pointing to a YOLO dataset with images/ and labels/."
        )
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a YOLO-format dataset config. Use --inspect to load labels with "
            "the SimpleDet dataset adapter."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--classes-file", type=Path)
    parser.add_argument("--inspect", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = require_dataset_root(args.dataset_root)
    config = build_yolo_dataset_config(root, classes_file=args.classes_file)
    if args.inspect:
        dataset = load_dataset(
            str(root),
            format="yolo",
            classes_file=str(args.classes_file) if args.classes_file else None,
        )
        config["sample_count"] = len(dataset.get("samples", ()))
        config["categories"] = dataset.get("categories", ())
    print(json.dumps(config, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
