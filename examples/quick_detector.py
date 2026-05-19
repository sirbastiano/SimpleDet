"""Build and inspect a small RetinaNet detector spec."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from simpledet.suite import build_detector, build_neck, compile_native_detector_plan


def build_quick_detector(num_classes: int = 3, backbone: str = "resnet18"):
    """Return a registry-backed RetinaNet spec without constructing torch modules."""

    return build_detector(
        "retinanet",
        backbone=backbone,
        neck=build_neck("FPN", out_channels=256, num_outs=4),
        num_classes=num_classes,
        pretrained=False,
    )


def detector_summary(num_classes: int = 3, backbone: str = "resnet18") -> dict:
    spec = build_quick_detector(num_classes=num_classes, backbone=backbone)
    plan = compile_native_detector_plan(spec)
    return {
        "architecture": spec.architecture,
        "family": spec.family,
        "num_classes": spec.num_classes,
        "encoder": {
            "source": spec.encoder.source,
            "name": spec.encoder.name,
            "pretrained": spec.encoder.pretrained,
        },
        "neck": plan.neck.to_dict(),
        "head": plan.head.to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a minimal RetinaNet detector spec and print its native plan."
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--backbone", default="resnet18")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(detector_summary(args.num_classes, args.backbone), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
