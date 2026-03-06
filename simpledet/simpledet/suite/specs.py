"""Structured detector suite specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else dict(value)


@dataclass(slots=True)
class EncoderSpec:
    """Backbone/encoder definition for the suite compiler."""

    source: str = "timm"
    name: str | None = None
    pretrained: bool = True
    in_channels: int | None = None
    backbone_cfg: dict[str, Any] | None = None
    feature_channels: tuple[int, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extra = _copy_mapping(self.extra)
        if self.source == "timm" and not self.name:
            raise ValueError("`EncoderSpec.name` is required when source='timm'.")
        if self.source == "config":
            if not isinstance(self.backbone_cfg, dict):
                raise ValueError("`backbone_cfg` is required when source='config'.")
            if not self.feature_channels:
                raise ValueError(
                    "`feature_channels` is required when source='config' so downstream "
                    "components can be adapted safely."
                )
        if self.feature_channels is not None:
            self.feature_channels = tuple(int(channel) for channel in self.feature_channels)


@dataclass(slots=True)
class NeckSpec:
    """Neck definition for the suite compiler."""

    name: str = "auto"
    neck_cfg: dict[str, Any] | None = None
    out_channels: int | None = None
    num_outs: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extra = _copy_mapping(self.extra)
        if self.neck_cfg is not None:
            self.neck_cfg = dict(self.neck_cfg)


@dataclass(slots=True)
class HeadSpec:
    """Head definition for dense and ROI detector families."""

    name: str = "auto"
    head_cfg: dict[str, Any] | None = None
    num_classes: int | None = None
    with_mask: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extra = _copy_mapping(self.extra)
        if self.head_cfg is not None:
            self.head_cfg = dict(self.head_cfg)


@dataclass(slots=True)
class DecoderSpec:
    """Decoder definition for transformer detector families."""

    name: str = "auto"
    decoder_cfg: dict[str, Any] | None = None
    num_queries: int | None = None
    embed_dims: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extra = _copy_mapping(self.extra)
        if self.decoder_cfg is not None:
            self.decoder_cfg = dict(self.decoder_cfg)


@dataclass(slots=True)
class DetectorSpec:
    """Top-level detector suite specification."""

    architecture: str
    family: str
    num_classes: int = 1
    encoder: EncoderSpec | None = None
    neck: NeckSpec | None = None
    head: HeadSpec | None = None
    decoder: DecoderSpec | None = None
    strict_auto_adapt: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.family = str(self.family).strip().lower()
        self.architecture = str(self.architecture).strip().lower()
        self.num_classes = int(self.num_classes)
        if self.num_classes <= 0:
            raise ValueError("`num_classes` must be a positive integer.")
        self.overrides = _copy_mapping(self.overrides)

