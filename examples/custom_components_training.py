"""Train a custom-backbone/custom-neck/custom-head detector on COCO-format data."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "simpledet"
for _path in (REPO_ROOT, SOURCE_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import examples.custom_components  # noqa: F401

from simpledet import ProjectConfig, run_project, validate_project_config
from simpledet.suite import (
    build_custom_encoder,
    build_custom_head,
    build_custom_neck,
    build_detector,
)


CUSTOM_COMPONENT_IMPORT = "examples.custom_components"


def build_custom_detector_spec(*, num_classes: int = 1):
    """Return a RetinaNet spec wired to custom registered components."""

    backbone = build_custom_encoder(
        "ExampleTinyBackbone",
        feature_channels=(16, 32, 64, 128),
        imports=(CUSTOM_COMPONENT_IMPORT,),
        in_channels=3,
        widths=(16, 32, 64, 128),
    )
    neck = build_custom_neck(
        "ExampleTinyNeck",
        imports=(CUSTOM_COMPONENT_IMPORT,),
        out_channels=32,
        num_outs=4,
    )
    head = build_custom_head(
        "ExampleTinyDenseHead",
        imports=(CUSTOM_COMPONENT_IMPORT,),
        num_classes=num_classes,
        num_anchors=9,
        hidden_channels=32,
    )
    return build_detector(
        "retinanet",
        encoder=backbone,
        neck=neck,
        head=head,
        num_classes=num_classes,
        pretrained=False,
    )


def build_custom_coco_project_config(
    dataset_root: Path,
    *,
    workdir: Path,
    classes: Sequence[str] = ("object",),
) -> ProjectConfig:
    detector_spec = build_custom_detector_spec(num_classes=len(tuple(classes)))
    return ProjectConfig.from_mapping(
        {
            "stages": ["build", "train", "test", "infer"],
            "workdir": str(workdir),
            "seed": 71,
            "detector_spec": asdict(detector_spec),
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
                "batch_size": 1,
                "max_epochs": 1,
                "num_workers": 0,
                "accelerator": "cpu",
                "devices": 1,
            },
            "optimizer": {
                "name": "SGD",
                "learning_rate": 0.001,
            },
            "checkpoint": {"resume": False},
            "export": {"formats": ["json"]},
        }
    )


def create_tiny_coco_dataset(root: Path) -> Path:
    """Create a tiny COCO-format dataset suitable for a one-epoch smoke run."""

    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for image_id, split in enumerate(("train", "val", "test"), start=1):
        _write_png(root / "images" / f"{split}.png")
        _write_coco_split(root / "annotations" / f"{split}.json", split=split, image_id=image_id)
    return root


def run_custom_coco_training(
    dataset_root: Path,
    *,
    workdir: Path,
    classes: Sequence[str] = ("object",),
) -> dict:
    project = build_custom_coco_project_config(dataset_root, workdir=workdir, classes=classes)
    validate_project_config(project, strict=True)
    return run_project(project, stages=("build", "train", "test", "infer"))


def summarize_run_result(result: dict) -> dict:
    """Return the compact evidence users need after a custom-component run."""

    return {
        "stages": result.get("stages"),
        "architecture": result.get("architecture"),
        "checkpoint_path": result.get("train", {}).get("checkpoint_path"),
        "manifest_path": result.get("manifest_path"),
        "custom_components": {
            "backbone": result.get("build", {})
            .get("detector_plan", {})
            .get("encoder", {})
            .get("type"),
            "neck": result.get("build", {})
            .get("detector_plan", {})
            .get("neck", {})
            .get("type"),
            "head": result.get("build", {})
            .get("detector_plan", {})
            .get("head", {})
            .get("type"),
        },
        "test_predictions": len(result.get("test", {}).get("predictions", [])),
        "infer_predictions": len(result.get("infer", {}).get("predictions", [])),
    }


def _write_png(path: Path, *, width: int = 32, height: int = 32) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            if 8 <= x < 24 and 8 <= y < 24:
                row.extend((220, 40, 40))
            else:
                row.extend((24, 32, 44))
        rows.append(b"\x00" + bytes(row))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _write_coco_split(path: Path, *, split: str, image_id: int) -> None:
    payload = {
        "images": [
            {"id": image_id, "file_name": f"{split}.png", "width": 32, "height": 32}
        ],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one CPU epoch of a RetinaNet detector using a custom registered "
            "backbone, neck, and dense head on COCO-format data."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="COCO-format dataset root. If omitted, a tiny COCO-format dataset is created in /tmp.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "simpledet-custom-components-run",
    )
    parser.add_argument("--class-name", action="append", dest="classes")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run build/train/test/infer instead of printing the validated config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = Path(tempfile.gettempdir()) / "simpledet-custom-components-coco"
        create_tiny_coco_dataset(dataset_root)
    classes = tuple(args.classes or ("object",))
    project = build_custom_coco_project_config(
        dataset_root,
        workdir=args.workdir,
        classes=classes,
    )
    validation = validate_project_config(project, strict=True)
    if args.run:
        result = run_project(project, stages=("build", "train", "test", "infer"))
        print(json.dumps(summarize_run_result(result), indent=2, default=str))
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
    raise SystemExit(main())
