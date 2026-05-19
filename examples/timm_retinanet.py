"""Build a RetinaNet spec with a TIMM ResNet-18 backbone."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from simpledet.suite import build_detector, compile_native_detector_plan


TIMM_BACKBONE = "timm:resnet18"


def build_timm_retinanet_spec(num_classes: int = 3):
    """Return the TIMM-backed RetinaNet spec required by the example gallery."""

    return build_detector(
        "retinanet",
        encoder=TIMM_BACKBONE,
        num_classes=num_classes,
        pretrained=False,
    )


def timm_retinanet_summary(num_classes: int = 3) -> dict:
    spec = build_timm_retinanet_spec(num_classes=num_classes)
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
        "plan_encoder": plan.encoder.to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a RetinaNet detector spec that uses encoder='timm:resnet18'. "
            "This compiles metadata only; install simpledet[timm] before building "
            "the actual torch module."
        )
    )
    parser.add_argument("--num-classes", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(timm_retinanet_summary(args.num_classes), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
