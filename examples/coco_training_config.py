"""Create a COCO training project config for native SimpleDet training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from simpledet import ProjectConfig, run_project, validate_project_config


DEFAULT_SAMPLE_ROOT = Path(__file__).resolve().parent / "sample_data" / "coco"


def build_coco_project_config(
    dataset_root: Path,
    *,
    workdir: Path,
    classes: Sequence[str] = ("object",),
) -> ProjectConfig:
    return ProjectConfig.from_mapping(
        {
            "stages": ["build", "train"],
            "workdir": str(workdir),
            "seed": 71,
            "detector": {
                "name": "retinanet",
                "num_classes": len(tuple(classes)),
                "backbone": "resnet18",
                "pretrained": False,
            },
            "dataset": {
                "format": "coco",
                "root": str(dataset_root),
                "train": "annotations/train.json",
                "val": "annotations/val.json",
                "test": "annotations/test.json",
                "images_dir": "images",
                "classes": list(classes),
                "in_channels": 3,
            },
            "runtime": {
                "batch_size": 2,
                "max_epochs": 1,
                "num_workers": 0,
                "accelerator": "cpu",
                "devices": 1,
            },
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.001,
            },
            "checkpoint": {"resume": False},
            "export": {"formats": ["json"]},
        }
    )


def require_dataset_root(path: Path) -> Path:
    root = path.expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Sample COCO dataset not found: {root}. "
            "Pass --dataset-root pointing to a COCO dataset with images/ and annotations/."
        )
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate a native COCO RetinaNet training config. "
            "By default this expects examples/sample_data/coco; pass --dataset-root "
            "for your own data."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd() / "simpledet-runs" / "coco-retinanet",
    )
    parser.add_argument("--class-name", action="append", dest="classes")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run build/train instead of printing the config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    classes = tuple(args.classes or ("object",))
    project = build_coco_project_config(
        require_dataset_root(args.dataset_root),
        workdir=args.workdir,
        classes=classes,
    )
    validation = validate_project_config(project, strict=True)
    if args.run:
        result = run_project(project, stages=("build", "train"))
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            json.dumps(
                {"project": asdict(project), "validation": validation},
                indent=2,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
