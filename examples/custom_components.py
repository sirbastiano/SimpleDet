"""Custom native components used by the custom COCO training example."""

from __future__ import annotations

import argparse
import json
import types
from typing import Any

from simpledet.extensions import ENCODERS, HEADS, NECKS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # Keep example-gallery imports usable without simpledet[cpu].
    torch = None
    F = None

    class _MissingTorchModule:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "Example custom components require the CPU runtime. "
                "Install with: python -m pip install -e '.[cpu]'"
            )

    nn = types.SimpleNamespace(Module=_MissingTorchModule)


def _register_once(registry, name: str, component, **metadata):
    if name in registry.names():
        return registry.get(name)
    return registry.register(name, **metadata)(component)


class _ExampleResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class _ExampleTinyBackbone(nn.Module):
    """Four-stage CNN backbone that returns feature maps for an FPN-style neck."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        widths: tuple[int, ...] | list[int] = (16, 32, 64, 128),
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_channels = tuple(int(width) for width in widths)
        if len(self.feature_channels) != 4:
            raise ValueError("ExampleTinyBackbone expects exactly four stage widths.")
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feature_channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        previous = self.feature_channels[0]
        for width in self.feature_channels:
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    _ExampleResidualBlock(width),
                )
            )
            previous = width

    def forward(self, x):
        features = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)


ExampleTinyBackbone = _register_once(
    ENCODERS,
    "ExampleTinyBackbone",
    _ExampleTinyBackbone,
    aliases=("example_tiny_backbone", "custom_tiny_backbone"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_sequence", "feature_channels"),
    validation_status="runtime_validated",
    family="custom",
    summary="Small custom CNN backbone for native detector composition examples.",
)


class _ExampleTinyNeck(nn.Module):
    """Project every backbone feature level to a shared channel width."""

    def __init__(
        self,
        *,
        in_channels: list[int] | tuple[int, ...],
        out_channels: int = 32,
        num_outs: int | None = None,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs or len(self.in_channels))
        self.projections = nn.ModuleList(
            nn.Conv2d(channel, self.out_channels, kernel_size=1)
            for channel in self.in_channels
        )

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError(
                f"ExampleTinyNeck expected {len(self.projections)} feature maps, "
                f"got {len(features)}."
            )
        outputs = [projection(feature) for projection, feature in zip(self.projections, features)]
        while len(outputs) < self.num_outs:
            outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))
        return tuple(outputs[: self.num_outs])


ExampleTinyNeck = _register_once(
    NECKS,
    "ExampleTinyNeck",
    _ExampleTinyNeck,
    aliases=("example_tiny_neck", "custom_tiny_neck"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_sequence", "feature_pyramid"),
    validation_status="runtime_validated",
    family="neck",
    summary="Small custom projection neck for native detector composition examples.",
)


class _ExampleTinyDenseHead(nn.Module):
    """Dense head returning cls_logits and bbox_regression for Retina-style losses."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        hidden_channels: int | None = None,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        hidden = int(hidden_channels or in_channels)
        self.tower = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(hidden, self.num_anchors * self.num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(hidden, self.num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, features):
        cls_logits = []
        bbox_regression = []
        for feature in features:
            hidden = self.tower(feature)
            logits = self.cls_logits(hidden)
            boxes = self.bbox_pred(hidden)
            batch, _channels, height, width = logits.shape
            cls_logits.append(
                logits.view(batch, self.num_anchors, self.num_classes, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(batch, -1, self.num_classes)
            )
            bbox_regression.append(
                boxes.view(batch, self.num_anchors, 4, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(batch, -1, 4)
            )
        return {
            "cls_logits": torch.cat(cls_logits, dim=1),
            "bbox_regression": torch.cat(bbox_regression, dim=1),
        }


ExampleTinyDenseHead = _register_once(
    HEADS,
    "ExampleTinyDenseHead",
    _ExampleTinyDenseHead,
    aliases=("example_tiny_dense_head", "custom_tiny_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "retinanet_head_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="Small custom dense head that satisfies the Retina-style output contract.",
)


__all__ = [
    "ExampleTinyBackbone",
    "ExampleTinyDenseHead",
    "ExampleTinyNeck",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List the custom native components used by custom_components_training.py."
    )
    parser.parse_args(argv)
    print(
        json.dumps(
            {
                "backbone": "ExampleTinyBackbone",
                "neck": "ExampleTinyNeck",
                "head": "ExampleTinyDenseHead",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
