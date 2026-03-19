"""Native neck implementations for the Lightning backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import NECKS

require_dependency("torch", "native necks")
import torch.nn as nn


@dataclass(slots=True, frozen=True)
class NeckSpec:
    name: str
    out_channels: int
    num_outs: int


@NECKS.register("FPN")
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

        self.in_channels = [int(channel) for channel in in_channels]
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs or len(self.in_channels))
        self.fpn = FeaturePyramidNetwork(self.in_channels, self.out_channels)

    def __call__(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.forward(features)

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        feature_map = {str(index): feature for index, feature in enumerate(features)}
        outputs = self.fpn(feature_map)
        ordered = [outputs[key] for key in sorted(outputs.keys(), key=int)]
        return tuple(ordered[: self.num_outs])


@NECKS.register("ChannelMapper")
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

        self.in_channels = [int(channel) for channel in in_channels]
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs or len(self.in_channels))
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(int(channel), self.out_channels, kernel_size=kernel_size)
                for channel in self.in_channels[: self.num_outs]
            ]
        )

    def __call__(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.forward(features)

    def forward(self, features: tuple[Any, ...]) -> tuple[Any, ...]:
        outputs = []
        for layer, feature in zip(self.layers, features):
            outputs.append(layer(feature))
        return tuple(outputs[: self.num_outs])


@NECKS.register("PAN")
class PathAggregationNeck(FeaturePyramidNeck):
    """Alias retained for callers that request PAN-style names."""


@NECKS.register("PANET")
class PANETNeck(PathAggregationNeck):
    """Alias retained for PANet naming."""


@NECKS.register("BiFPN")
class BiFPNNeck(ChannelMapperNeck):
    """Alias retained for callers that request BiFPN-style naming."""


@NECKS.register("BiFPNV2")
class BiFPNV2Neck(BiFPNNeck):
    """Alias retained for BiFPNv2 naming."""


@NECKS.register("PAFPN")
class PAFPNNeck(FeaturePyramidNeck):
    """Alias retained for callers that request PAFPN-style naming."""


@NECKS.register("NASFPN")
class NASFPNNeck(FeaturePyramidNeck):
    """Alias retained for callers that request NAS-FPN naming."""


@NECKS.register("FPNLite")
class FPNLiteNeck(FeaturePyramidNeck):
    """Alias retained for lightweight FPN naming variations."""


@NECKS.register("FPNLiteNeck")
class FPNLiteAliasNeck(FeaturePyramidNeck):
    """Explicit alias retained for camel-cased callers."""


def _resolve_neck_name(requested: str) -> str:
    normalized = "".join(char for char in str(requested).lower() if char.isalnum())
    if not normalized:
        raise ValueError("Neck type cannot be empty.")

    exact_name = str(requested).strip()
    if exact_name in NECKS.names():
        return exact_name
    compact_name_map = {name.lower().replace("_", "").replace("-", ""): name for name in NECKS.names()}
    resolved = compact_name_map.get(normalized)
    if resolved is not None:
        return resolved

    if normalized.startswith("fpn"):
        return "FPN"
    if normalized.startswith("panet") or normalized == "pan":
        return "PANET"
    if normalized.startswith("bifpn"):
        return "BiFPN"
    if normalized.startswith("pafpn"):
        return "PAFPN"
    if normalized.startswith("naspfn"):
        return "NASFPN"
    if normalized.startswith("channelmapper"):
        return "ChannelMapper"
    if normalized.startswith("fpnlite"):
        return "FPNLite"

    raise KeyError(f"Unknown neck family '{requested}'. Available: {', '.join(NECKS.names())}.")


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
