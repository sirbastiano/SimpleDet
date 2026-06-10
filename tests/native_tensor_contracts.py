"""Reusable CPU tensor-contract helpers for native detector smoke tests.

The helpers in this module keep detector-family tests small and consistent:
they build deterministic CPU tensors for images, multiscale feature maps,
targets, and metadata, then assert the dense-head output contract that later
families must satisfy before claiming runtime support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import sys
import unittest


@dataclass(frozen=True)
class DetectorSmokeBatch:
    """Bundle of tensor fixtures used by CPU-only native detector tests."""

    images: list[Any]
    feature_maps: list[Any]
    boxes: list[Any]
    labels: list[Any]
    metadata: list[dict[str, Any]]
    targets: list[dict[str, Any]]


def require_torch():
    """Return torch or skip the current unittest when the CPU extra is absent."""

    try:
        import torch
    except ImportError as exc:
        raise unittest.SkipTest("PyTorch CPU runtime is not installed.") from exc
    return torch


def make_dummy_images(
    *,
    batch_size: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
):
    """Create a list of deterministic CHW image tensors for native models."""

    torch = require_torch()
    total_values = max(int(batch_size) * int(channels) * int(height) * int(width), 1)
    batched = torch.arange(total_values, dtype=torch.float32, device=device).reshape(
        int(batch_size),
        int(channels),
        int(height),
        int(width),
    )
    batched = batched / float(total_values)
    return [batched[index].clone() for index in range(int(batch_size))]


def make_dummy_feature_maps(
    *,
    batch_size: int = 2,
    channels: int = 8,
    spatial_shapes: Sequence[tuple[int, int]] = ((16, 16), (8, 8), (4, 4)),
    device: str = "cpu",
):
    """Create multiscale NCHW feature tensors for dense-head smoke tests."""

    torch = require_torch()
    feature_maps = []
    for level, (height, width) in enumerate(spatial_shapes):
        feature = torch.full(
            (int(batch_size), int(channels), int(height), int(width)),
            fill_value=float(level + 1) / 10.0,
            dtype=torch.float32,
            device=device,
        )
        feature_maps.append(feature)
    return feature_maps


def make_dummy_boxes(
    *,
    num_images: int = 2,
    boxes_per_image: int = 2,
    image_size: tuple[int, int] = (64, 64),
    device: str = "cpu",
):
    """Create per-image xyxy boxes that stay inside the image extent."""

    torch = require_torch()
    height, width = int(image_size[0]), int(image_size[1])
    base_boxes = []
    for index in range(int(boxes_per_image)):
        x1 = min(width - 2, 1 + index * 2)
        y1 = min(height - 2, 1 + index * 2)
        x2 = min(width - 1, x1 + max(1, width // 4))
        y2 = min(height - 1, y1 + max(1, height // 4))
        base_boxes.append((float(x1), float(y1), float(x2), float(y2)))
    boxes = torch.tensor(base_boxes, dtype=torch.float32, device=device).reshape(int(boxes_per_image), 4)
    return [boxes.clone() for _ in range(int(num_images))]


def make_dummy_labels(
    *,
    num_images: int = 2,
    boxes_per_image: int = 2,
    num_classes: int = 3,
    device: str = "cpu",
):
    """Create one-based class labels compatible with detection targets."""

    torch = require_torch()
    labels = (torch.arange(int(boxes_per_image), dtype=torch.long, device=device) % int(num_classes)) + 1
    return [labels.clone() for _ in range(int(num_images))]


def make_dummy_metadata(
    *,
    num_images: int = 2,
    image_size: tuple[int, int] = (64, 64),
) -> list[dict[str, Any]]:
    """Create lightweight image metadata for detector-family smoke tests."""

    height, width = int(image_size[0]), int(image_size[1])
    return [
        {
            "image_id": index,
            "image_shape": (height, width),
            "original_shape": (height, width),
            "scale_factor": 1.0,
        }
        for index in range(int(num_images))
    ]


def make_dummy_targets(
    *,
    boxes: Sequence[Any],
    labels: Sequence[Any],
    metadata: Sequence[Mapping[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Combine boxes, labels, and metadata into native detection targets."""

    torch = require_torch()
    targets = []
    for image_boxes, image_labels, image_metadata in zip(boxes, labels, metadata):
        area = (image_boxes[:, 2] - image_boxes[:, 0]) * (image_boxes[:, 3] - image_boxes[:, 1])
        targets.append(
            {
                "boxes": image_boxes,
                "labels": image_labels,
                "image_id": torch.tensor([int(image_metadata["image_id"])], dtype=torch.long, device=device),
                "area": area,
                "iscrowd": torch.zeros((int(image_boxes.shape[0]),), dtype=torch.long, device=device),
            }
        )
    return targets


def make_cpu_detector_smoke_batch(
    *,
    batch_size: int = 2,
    image_channels: int = 3,
    image_size: tuple[int, int] = (64, 64),
    feature_channels: int = 8,
    feature_shapes: Sequence[tuple[int, int]] = ((16, 16), (8, 8), (4, 4)),
    boxes_per_image: int = 2,
    num_classes: int = 3,
    device: str = "cpu",
) -> DetectorSmokeBatch:
    """Build all tensor fixtures needed for a CPU-only detector smoke test."""

    images = make_dummy_images(
        batch_size=batch_size,
        channels=image_channels,
        height=image_size[0],
        width=image_size[1],
        device=device,
    )
    feature_maps = make_dummy_feature_maps(
        batch_size=batch_size,
        channels=feature_channels,
        spatial_shapes=feature_shapes,
        device=device,
    )
    boxes = make_dummy_boxes(
        num_images=batch_size,
        boxes_per_image=boxes_per_image,
        image_size=image_size,
        device=device,
    )
    labels = make_dummy_labels(
        num_images=batch_size,
        boxes_per_image=boxes_per_image,
        num_classes=num_classes,
        device=device,
    )
    metadata = make_dummy_metadata(num_images=batch_size, image_size=image_size)
    targets = make_dummy_targets(boxes=boxes, labels=labels, metadata=metadata, device=device)
    return DetectorSmokeBatch(
        images=images,
        feature_maps=feature_maps,
        boxes=boxes,
        labels=labels,
        metadata=metadata,
        targets=targets,
    )


def assert_dense_head_output_contract(
    outputs: Mapping[str, Sequence[Any]],
    feature_maps: Sequence[Any],
    *,
    batch_size: int,
    required_keys: Sequence[str] = ("cls_logits", "bbox_regression"),
) -> None:
    """Assert that a dense head returns one batched tensor per feature level."""

    if not isinstance(outputs, Mapping):
        raise AssertionError(
            f"dense head output contract expected a mapping, got {type(outputs).__name__}."
        )

    expected_levels = len(feature_maps)
    for key in required_keys:
        if key not in outputs:
            raise AssertionError(f"dense head output contract missing required key '{key}'.")
        levels = outputs[key]
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            raise AssertionError(
                f"dense head output '{key}' expected a sequence of tensors, got {type(levels).__name__}."
            )
        if len(levels) != expected_levels:
            raise AssertionError(
                f"dense head output '{key}' level count mismatch: expected {expected_levels} "
                f"levels from feature_maps, got {len(levels)}."
            )

        for level_index, (feature, tensor) in enumerate(zip(feature_maps, levels)):
            tensor_shape = _shape_tuple(tensor, key=key, level_index=level_index)
            feature_shape = _shape_tuple(feature, key="feature_maps", level_index=level_index)
            if len(tensor_shape) != 4:
                raise AssertionError(
                    f"dense head output '{key}' level {level_index} expected a 4D NCHW tensor, "
                    f"got shape {tensor_shape}."
                )
            if int(tensor_shape[0]) != int(batch_size):
                raise AssertionError(
                    f"dense head output '{key}' level {level_index} batch mismatch: expected "
                    f"batch size {int(batch_size)}, got {int(tensor_shape[0])}."
                )
            if len(feature_shape) == 4 and tuple(tensor_shape[-2:]) != tuple(feature_shape[-2:]):
                raise AssertionError(
                    f"dense head output '{key}' level {level_index} spatial mismatch: expected "
                    f"{tuple(feature_shape[-2:])} from feature_maps, got {tuple(tensor_shape[-2:])}."
                )


def _shape_tuple(value: Any, *, key: str, level_index: int) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise AssertionError(
            f"dense head output '{key}' level {level_index} expected tensor-like value with shape."
        )
    return tuple(int(dimension) for dimension in shape)


def clear_native_runtime_state() -> None:
    """Remove native module and registry state that may have been loaded with fake torch."""

    from simpledet.extensions import (
        ASSIGNERS,
        DECODERS,
        DETECTORS,
        ENCODERS,
        HEADS,
        LOSSES,
        NECKS,
        POSTPROCESSORS,
    )

    registries = (ASSIGNERS, DECODERS, DETECTORS, ENCODERS, HEADS, LOSSES, NECKS, POSTPROCESSORS)
    for registry in registries:
        for name, factory in tuple(registry._items.items()):
            if str(getattr(factory, "__module__", "")).startswith("simpledet.native"):
                registry._items.pop(name, None)
                registry._metadata.pop(name, None)
    for module_name in tuple(sys.modules):
        if module_name == "simpledet.native" or module_name.startswith("simpledet.native."):
            sys.modules.pop(module_name, None)
    simpledet_package = sys.modules.get("simpledet")
    if simpledet_package is not None:
        vars(simpledet_package).pop("native", None)


__all__ = [
    "DetectorSmokeBatch",
    "assert_dense_head_output_contract",
    "clear_native_runtime_state",
    "make_cpu_detector_smoke_batch",
    "make_dummy_boxes",
    "make_dummy_feature_maps",
    "make_dummy_images",
    "make_dummy_labels",
    "make_dummy_metadata",
    "make_dummy_targets",
    "require_torch",
]
