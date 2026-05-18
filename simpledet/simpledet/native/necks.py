"""Native neck implementations for the Lightning backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import NECKS

require_dependency("torch", "native necks")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


@dataclass(slots=True, frozen=True)
class NeckSpec:
    name: str
    out_channels: int
    num_outs: int


class TensorContractError(ValueError):
    """Raised when a neck receives tensors that violate its declared contract."""


_TORCH_CPU = (("torch", "cpu"),)
_TORCHVISION_CPU = (("torch", "cpu"), ("torchvision", "cpu"))
_FEATURE_PYRAMID_CONTRACTS = ("feature_sequence", "feature_pyramid")
_PROJECTED_CONTRACTS = ("feature_sequence", "projected_feature_sequence")


def _normalize_in_channels(in_channels: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(in_channels, int):
        channels = [int(in_channels)]
    else:
        channels = [int(channel) for channel in in_channels]
    if not channels:
        raise TensorContractError("neck tensor contract requires at least one in_channels entry.")
    if any(channel <= 0 for channel in channels):
        raise TensorContractError("neck tensor contract requires positive in_channels values.")
    return channels


def _positive_int(value: int | None, *, default: int, field_name: str) -> int:
    resolved = int(default if value is None else value)
    if resolved <= 0:
        raise TensorContractError(f"neck tensor contract requires {field_name} > 0.")
    return resolved


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dimension) for dimension in shape)


def _validate_feature_sequence(
    features: Sequence[Any],
    *,
    in_channels: Sequence[int],
    neck_name: str,
) -> None:
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise TensorContractError(
            f"{neck_name} tensor contract expected a sequence of feature tensors, "
            f"got {type(features).__name__}."
        )
    if len(features) != len(in_channels):
        raise TensorContractError(
            f"{neck_name} tensor contract feature level mismatch: expected "
            f"{len(in_channels)} feature levels from in_channels, got {len(features)}."
        )

    for level, (feature, expected_channels) in enumerate(zip(features, in_channels)):
        shape = _shape_tuple(feature)
        if shape is None:
            continue
        if len(shape) != 4:
            raise TensorContractError(
                f"{neck_name} tensor contract expected level {level} to be a 4D NCHW "
                f"tensor, got shape {shape}."
            )
        actual_channels = int(shape[1])
        if actual_channels != int(expected_channels):
            raise TensorContractError(
                f"{neck_name} tensor contract channel mismatch at level {level}: "
                f"expected {int(expected_channels)} channels, got {actual_channels}."
            )


def _has_spatial_shape(value: Any) -> bool:
    shape = _shape_tuple(value)
    return shape is not None and len(shape) == 4


def _resize_like(source: Any, target: Any) -> Any:
    source_shape = _shape_tuple(source)
    target_shape = _shape_tuple(target)
    if source_shape is None or target_shape is None:
        return source
    if len(source_shape) != 4 or len(target_shape) != 4:
        return source
    if tuple(source_shape[-2:]) == tuple(target_shape[-2:]):
        return source
    return F.interpolate(source, size=tuple(target_shape[-2:]), mode="nearest")


def _extend_feature_levels(
    outputs: list[Any],
    *,
    extra_convs: Sequence[Any],
    num_outs: int,
) -> list[Any]:
    extra_index = 0
    while len(outputs) < num_outs:
        previous = outputs[-1]
        if extra_index < len(extra_convs):
            outputs.append(extra_convs[extra_index](previous))
        elif _has_spatial_shape(previous):
            outputs.append(F.max_pool2d(previous, kernel_size=1, stride=2))
        else:
            outputs.append(previous)
        extra_index += 1
    return outputs


def _average_features(features: list[Any]) -> Any:
    fused = features[0]
    for feature in features[1:]:
        fused = fused + feature
    return fused / float(len(features))


@NECKS.register(
    "FPN",
    aliases=("fpn",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Feature pyramid network neck.",
)
class FeaturePyramidNeck(nn.Module):
    """Thin wrapper around torchvision FeaturePyramidNetwork."""

    def __init__(
        self,
        *,
        in_channels: list[int] | tuple[int, ...],
        out_channels: int = 256,
        num_outs: int | None = None,
    ) -> None:
        super().__init__()
        require_dependency("torchvision", "native necks")
        from torchvision.ops import FeaturePyramidNetwork

        self.in_channels = _normalize_in_channels(in_channels)
        self.out_channels = int(out_channels)
        self.num_outs = _positive_int(num_outs, default=len(self.in_channels), field_name="num_outs")
        self.fpn = FeaturePyramidNetwork(self.in_channels, self.out_channels)
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(max(self.num_outs - len(self.in_channels), 0))
            ]
        )

    def __call__(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.forward(features)

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        _validate_feature_sequence(features, in_channels=self.in_channels, neck_name=self.__class__.__name__)
        feature_map = {str(index): feature for index, feature in enumerate(features)}
        outputs = self.fpn(feature_map)
        ordered = [outputs[key] for key in sorted(outputs.keys(), key=int)]
        ordered = _extend_feature_levels(ordered, extra_convs=self.extra_convs, num_outs=self.num_outs)
        return tuple(ordered[: self.num_outs])


@NECKS.register(
    "ChannelMapper",
    aliases=("channel_mapper",),
    required_dependencies=_TORCH_CPU,
    tensor_contracts=_PROJECTED_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Channel projection neck.",
)
class ChannelMapperNeck(nn.Module):
    """Minimal channel projection neck for transformer-style multiscale features."""

    def __init__(
        self,
        *,
        in_channels: list[int] | tuple[int, ...],
        out_channels: int = 256,
        num_outs: int | None = None,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = _normalize_in_channels(in_channels)
        self.out_channels = int(out_channels)
        self.num_outs = _positive_int(num_outs, default=len(self.in_channels), field_name="num_outs")
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(int(channel), self.out_channels, kernel_size=kernel_size)
                for channel in self.in_channels
            ]
        )
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(max(self.num_outs - len(self.in_channels), 0))
            ]
        )

    def __call__(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.forward(features)

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        _validate_feature_sequence(features, in_channels=self.in_channels, neck_name=self.__class__.__name__)
        outputs = []
        for layer, feature in zip(self.layers, features):
            outputs.append(layer(feature))
        outputs = _extend_feature_levels(outputs, extra_convs=self.extra_convs, num_outs=self.num_outs)
        return tuple(outputs[: self.num_outs])


@NECKS.register(
    "PAN",
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Path aggregation pyramid neck.",
)
class PathAggregationNeck(FeaturePyramidNeck):
    """Feature pyramid with a bottom-up path aggregation pass."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.downsample_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(max(self.num_outs - 1, 0))
            ]
        )
        self.pan_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
                for _ in range(max(self.num_outs - 1, 0))
            ]
        )

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        outputs = list(super().forward(features))
        if not outputs or not all(_has_spatial_shape(output) for output in outputs):
            return tuple(outputs)
        for level in range(1, len(outputs)):
            bottom_up = self.downsample_convs[level - 1](outputs[level - 1])
            bottom_up = _resize_like(bottom_up, outputs[level])
            outputs[level] = self.pan_convs[level - 1](outputs[level] + bottom_up)
        return tuple(outputs)


@NECKS.register(
    "PANET",
    aliases=("panet_neck",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="PANet-style bottom-up feature aggregation neck.",
)
class PANETNeck(PathAggregationNeck):
    """PANet naming variant for path aggregation."""


@NECKS.register(
    "BiFPN",
    aliases=("bi_fpn",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Bidirectional feature pyramid neck.",
)
class BiFPNNeck(FeaturePyramidNeck):
    """Bidirectional feature pyramid with adjacent-level fusion."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fusion_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
                for _ in range(self.num_outs)
            ]
        )

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        outputs = list(super().forward(features))
        if not outputs or not all(_has_spatial_shape(output) for output in outputs):
            return tuple(outputs)
        fused_outputs = []
        for level, output in enumerate(outputs):
            candidates = [output]
            if level > 0:
                candidates.append(_resize_like(outputs[level - 1], output))
            if level + 1 < len(outputs):
                candidates.append(_resize_like(outputs[level + 1], output))
            fused_outputs.append(self.fusion_convs[level](_average_features(candidates)))
        return tuple(fused_outputs)


@NECKS.register(
    "BiFPNV2",
    aliases=("bi_fpn_v2",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="BiFPNv2 naming variant.",
)
class BiFPNV2Neck(BiFPNNeck):
    """BiFPNv2 naming variant."""


@NECKS.register(
    "PAFPN",
    aliases=("pa_fpn",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Path aggregation FPN neck.",
)
class PAFPNNeck(PathAggregationNeck):
    """FPN with path aggregation naming used by dense detectors."""


@NECKS.register(
    "NASFPN",
    aliases=("nas_fpn",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="NAS-FPN-style repeated feature fusion neck.",
)
class NASFPNNeck(FeaturePyramidNeck):
    """Repeated adjacent-level feature fusion inspired by NAS-FPN cells."""

    def __init__(self, *, stack_times: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stack_times = max(int(stack_times), 1)
        self.fusion_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
                for _ in range(self.stack_times * self.num_outs)
            ]
        )

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        outputs = list(super().forward(features))
        if not outputs or not all(_has_spatial_shape(output) for output in outputs):
            return tuple(outputs)
        for stack_index in range(self.stack_times):
            next_outputs = []
            for level, output in enumerate(outputs):
                candidates = [output]
                if level > 0:
                    candidates.append(_resize_like(outputs[level - 1], output))
                if level + 1 < len(outputs):
                    candidates.append(_resize_like(outputs[level + 1], output))
                conv = self.fusion_convs[stack_index * self.num_outs + level]
                next_outputs.append(conv(_average_features(candidates)))
            outputs = next_outputs
        return tuple(outputs)


@NECKS.register(
    "FPNLite",
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Lightweight FPN naming variant.",
)
class FPNLiteNeck(FeaturePyramidNeck):
    """Alias retained for lightweight FPN naming variations."""


@NECKS.register(
    "FPNLiteNeck",
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="Camel-cased FPNLite alias.",
)
class FPNLiteAliasNeck(FeaturePyramidNeck):
    """Explicit alias retained for camel-cased callers."""


@NECKS.register(
    "DilatedEncoder",
    aliases=("dilated_encoder",),
    required_dependencies=_TORCH_CPU,
    tensor_contracts=("single_feature_sequence", "dilated_feature_sequence"),
    validation_status="runtime_validated",
    family="neck",
    summary="YOLOF-style dilated encoder neck for one feature level.",
)
class DilatedEncoderNeck(nn.Module):
    """Dilated encoder used by YOLOF-style single-level detectors."""

    def __init__(
        self,
        *,
        in_channels: int | list[int] | tuple[int, ...],
        out_channels: int = 256,
        num_outs: int | None = None,
        dilations: tuple[int, ...] | list[int] = (2, 4, 6, 8),
    ) -> None:
        super().__init__()
        self.in_channels = _normalize_in_channels(in_channels)
        if len(self.in_channels) != 1:
            raise TensorContractError(
                "DilatedEncoderNeck tensor contract expected exactly one "
                f"in_channels entry, got {len(self.in_channels)}."
            )
        self.out_channels = int(out_channels)
        self.num_outs = _positive_int(num_outs, default=1, field_name="num_outs")
        self.project = nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=int(dilation),
                    dilation=int(dilation),
                )
                for dilation in dilations
            ]
        )
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(max(self.num_outs - 1, 0))
            ]
        )

    def __call__(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.forward(features)

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        _validate_feature_sequence(features, in_channels=self.in_channels, neck_name=self.__class__.__name__)
        output = self.project(features[0])
        for block in self.blocks:
            output = output + block(output)
        outputs = [output]
        outputs = _extend_feature_levels(outputs, extra_convs=self.extra_convs, num_outs=self.num_outs)
        return tuple(outputs[: self.num_outs])


@NECKS.register(
    "HRFPN",
    aliases=("hr_fpn",),
    required_dependencies=_TORCH_CPU,
    tensor_contracts=("multi_resolution_feature_sequence", "feature_pyramid"),
    validation_status="runtime_validated",
    family="neck",
    summary="HRNet-style high-resolution feature pyramid neck.",
)
class HRFPNNeck(nn.Module):
    """Fuse multi-resolution HRNet features into an FPN-style pyramid."""

    def __init__(
        self,
        *,
        in_channels: list[int] | tuple[int, ...],
        out_channels: int = 256,
        num_outs: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _normalize_in_channels(in_channels)
        self.out_channels = int(out_channels)
        self.num_outs = _positive_int(num_outs, default=len(self.in_channels), field_name="num_outs")
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(channel, self.out_channels, kernel_size=1) for channel in self.in_channels]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
                for _ in range(self.num_outs)
            ]
        )

    def __call__(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.forward(features)

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        _validate_feature_sequence(features, in_channels=self.in_channels, neck_name=self.__class__.__name__)
        projected = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        if not projected or not all(_has_spatial_shape(feature) for feature in projected):
            return tuple(projected[: self.num_outs])
        reference = projected[0]
        aligned = [_resize_like(feature, reference) for feature in projected]
        fused = _average_features(aligned)
        outputs = [self.output_convs[0](fused)]
        while len(outputs) < self.num_outs:
            pooled = F.max_pool2d(outputs[-1], kernel_size=1, stride=2)
            outputs.append(self.output_convs[len(outputs)](pooled))
        return tuple(outputs[: self.num_outs])


@NECKS.register(
    "SSDNeck",
    aliases=("ssd_neck",),
    required_dependencies=_TORCH_CPU,
    tensor_contracts=_PROJECTED_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="SSD-style projection and extra-level neck.",
)
class SSDNeck(ChannelMapperNeck):
    """Project SSD feature levels and extend them with stride-2 pyramid levels."""


@NECKS.register(
    "YOLOXPAFPN",
    aliases=("yolox_pafpn",),
    required_dependencies=_TORCHVISION_CPU,
    tensor_contracts=_FEATURE_PYRAMID_CONTRACTS,
    validation_status="runtime_validated",
    family="neck",
    summary="YOLOX path aggregation feature pyramid neck.",
)
class YOLOXPAFPNNeck(PathAggregationNeck):
    """YOLOX naming variant for path aggregation FPN."""


def _resolve_neck_name(requested: str) -> str:
    normalized = "".join(char for char in str(requested).lower() if char.isalnum())
    if not normalized:
        raise ValueError("Neck type cannot be empty.")

    return NECKS.resolve_name(str(requested))


def build_native_neck(neck_plan, *, feature_channels: tuple[int, ...]) -> tuple[Any, NeckSpec]:
    if neck_plan is None or str(neck_plan.type).strip().lower() in {"", "auto"}:
        neck_plan = type("AutoNeckPlan", (), {"type": "FPN", "params": {}})()

    neck_type = str(neck_plan.type)
    neck_type = _resolve_neck_name(neck_type)
    params = dict(getattr(neck_plan, "params", {}))
    params.setdefault("in_channels", list(feature_channels))
    out_channels = int(params.get("out_channels", 256))
    num_outs = int(params.get("num_outs", len(feature_channels)))
    factory = NECKS.get(neck_type)
    neck = factory(**params)
    spec = NeckSpec(name=neck_type, out_channels=out_channels, num_outs=num_outs)
    return neck, spec
