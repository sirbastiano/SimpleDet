"""Load a lightweight checkpoint for image-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from simpledet import load_checkpoint_for_inference


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "sample_data" / "checkpoints" / "model.pt"


def require_checkpoint(path: Path) -> Path:
    checkpoint = path.expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Sample checkpoint not found: {checkpoint}. "
            "Pass --checkpoint pointing to a trusted .pt, .pth, or .ckpt file."
        )
    return checkpoint


def load_predictor_from_checkpoint(
    checkpoint: Path,
    *,
    model_name: str,
    num_classes: int,
    device: str = "cpu",
):
    return load_checkpoint_for_inference(
        require_checkpoint(checkpoint),
        model_name=model_name,
        num_classes=num_classes,
        device=device,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trusted lightweight checkpoint through the public inference API. "
            "Only pass checkpoint files you control."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = require_checkpoint(args.checkpoint)
    predictor = load_predictor_from_checkpoint(
        checkpoint,
        model_name=args.model_name,
        num_classes=args.num_classes,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "model_name": args.model_name,
                "num_classes": args.num_classes,
                "device": args.device,
                "predictor_type": type(predictor).__name__,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
