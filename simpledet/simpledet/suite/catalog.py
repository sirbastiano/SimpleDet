"""Canonical suite builders and architecture catalog."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .specs import DecoderSpec, DetectorSpec, EncoderSpec, HeadSpec, NeckSpec


ARCHITECTURE_FAMILIES: dict[str, str] = {
    "retinanet": "dense",
    "fcos": "dense",
    "vfnet": "dense",
    "fovea": "dense",
    "faster_rcnn": "roi",
    "mask_rcnn": "roi",
    "cascade_rcnn": "roi",
    "cascade_mask_rcnn": "roi",
    "detr": "transformer",
    "deformable_detr": "transformer",
}


def build_encoder(
    name: str,
    *,
    source: str = "timm",
    pretrained: bool = True,
    in_channels: int | None = None,
    backbone_cfg: dict[str, Any] | None = None,
    feature_channels: tuple[int, ...] | list[int] | None = None,
    **extra: Any,
) -> EncoderSpec:
    return EncoderSpec(
        source=source,
        name=name,
        pretrained=pretrained,
        in_channels=in_channels,
        backbone_cfg=backbone_cfg,
        feature_channels=tuple(feature_channels) if feature_channels is not None else None,
        extra=extra,
    )


def build_neck(
    name: str = "auto",
    *,
    neck_cfg: dict[str, Any] | None = None,
    out_channels: int | None = None,
    num_outs: int | None = None,
    **extra: Any,
) -> NeckSpec:
    return NeckSpec(
        name=name,
        neck_cfg=neck_cfg,
        out_channels=out_channels,
        num_outs=num_outs,
        extra=extra,
    )


def build_head(
    name: str = "auto",
    *,
    head_cfg: dict[str, Any] | None = None,
    num_classes: int | None = None,
    with_mask: bool = False,
    **extra: Any,
) -> HeadSpec:
    return HeadSpec(
        name=name,
        head_cfg=head_cfg,
        num_classes=num_classes,
        with_mask=with_mask,
        extra=extra,
    )


def build_decoder(
    name: str = "auto",
    *,
    decoder_cfg: dict[str, Any] | None = None,
    num_queries: int | None = None,
    embed_dims: int | None = None,
    **extra: Any,
) -> DecoderSpec:
    return DecoderSpec(
        name=name,
        decoder_cfg=decoder_cfg,
        num_queries=num_queries,
        embed_dims=embed_dims,
        extra=extra,
    )


def build_detector(
    architecture: str,
    *,
    num_classes: int = 1,
    encoder: str | EncoderSpec | None = None,
    neck: NeckSpec | None = None,
    head: HeadSpec | None = None,
    decoder: DecoderSpec | None = None,
    in_channels: int = 3,
    pretrained: bool = True,
    strict_auto_adapt: bool = True,
    **overrides: Any,
) -> DetectorSpec:
    normalized_architecture = str(architecture).strip().lower()
    family = ARCHITECTURE_FAMILIES.get(normalized_architecture)
    if family is None:
        known = ", ".join(sorted(ARCHITECTURE_FAMILIES))
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {known}.")

    if isinstance(encoder, str):
        encoder = build_encoder(
            encoder,
            source="timm",
            pretrained=pretrained,
            in_channels=in_channels,
        )
    elif encoder is None:
        encoder = build_encoder(
            "resnet18.a1_in1k",
            source="timm",
            pretrained=pretrained,
            in_channels=in_channels,
        )

    if head is None:
        head = build_head(
            num_classes=num_classes,
            with_mask=normalized_architecture in {"mask_rcnn", "cascade_mask_rcnn"},
        )
    else:
        head = replace(head, num_classes=num_classes if head.num_classes is None else head.num_classes)

    if family == "transformer" and decoder is None:
        decoder = build_decoder()

    return DetectorSpec(
        architecture=normalized_architecture,
        family=family,
        num_classes=num_classes,
        encoder=encoder,
        neck=neck,
        head=head,
        decoder=decoder,
        strict_auto_adapt=strict_auto_adapt,
        overrides=dict(overrides),
    )

