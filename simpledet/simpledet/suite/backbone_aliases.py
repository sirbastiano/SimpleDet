"""Backbone alias catalog for native suite builders."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any


@dataclass(frozen=True, slots=True)
class BackboneAlias:
    name: str
    family: str
    model_name: str
    stage_channels: tuple[int, ...]
    aliases: tuple[str, ...] = ()
    summary: str = ""

    @property
    def out_indices(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.stage_channels) + 1))


class UnknownBackboneError(ValueError):
    """Raised when a requested backbone alias is not registered."""


BACKBONE_ALIASES: tuple[BackboneAlias, ...] = (
    BackboneAlias(
        name="resnet18",
        family="ResNet",
        model_name="resnet18",
        stage_channels=(64, 128, 256, 512),
        aliases=("ResNet-18", "r18"),
        summary="ResNet-18 feature backbone.",
    ),
    BackboneAlias(
        name="resnet50",
        family="ResNet",
        model_name="resnet50",
        stage_channels=(256, 512, 1024, 2048),
        aliases=("ResNet", "ResNet-50", "r50"),
        summary="ResNet-50 feature backbone.",
    ),
    BackboneAlias(
        name="resnet101",
        family="ResNet",
        model_name="resnet101",
        stage_channels=(256, 512, 1024, 2048),
        aliases=("ResNet-101", "r101"),
        summary="ResNet-101 feature backbone.",
    ),
    BackboneAlias(
        name="resnext50_32x4d",
        family="ResNeXt",
        model_name="resnext50_32x4d",
        stage_channels=(256, 512, 1024, 2048),
        aliases=("ResNeXt", "ResNeXt-50"),
        summary="ResNeXt-50 32x4d feature backbone.",
    ),
    BackboneAlias(
        name="res2net50_26w_4s",
        family="Res2Net",
        model_name="res2net50_26w_4s",
        stage_channels=(256, 512, 1024, 2048),
        aliases=("Res2Net", "Res2Net-50"),
        summary="Res2Net-50 26w 4s feature backbone.",
    ),
    BackboneAlias(
        name="hrnet_w18",
        family="HRNet",
        model_name="hrnet_w18",
        stage_channels=(18, 36, 72, 144),
        aliases=("HRNet", "HRNet-W18", "hrnet18"),
        summary="HRNet-W18 feature backbone.",
    ),
    BackboneAlias(
        name="cspdarknet53",
        family="CSPDarkNet",
        model_name="cspdarknet53",
        stage_channels=(128, 256, 512, 1024),
        aliases=("CSPDarkNet", "CSPDarkNet-53", "darknet53"),
        summary="CSPDarkNet-53 feature backbone.",
    ),
    BackboneAlias(
        name="cspnext_tiny",
        family="CSPNeXt",
        model_name="cspnext_tiny",
        stage_channels=(96, 192, 384, 768),
        aliases=("CSPNeXt", "CSPNeXt-Tiny"),
        summary="CSPNeXt-Tiny feature backbone.",
    ),
    BackboneAlias(
        name="mobilenetv2_100",
        family="MobileNetV2",
        model_name="mobilenetv2_100",
        stage_channels=(24, 32, 96, 320),
        aliases=("MobileNetV2",),
        summary="MobileNetV2 feature backbone.",
    ),
    BackboneAlias(
        name="mobilenetv3_large_100",
        family="MobileNetV3",
        model_name="mobilenetv3_large_100",
        stage_channels=(24, 40, 112, 960),
        aliases=("MobileNetV3", "MobileNetV3-Large"),
        summary="MobileNetV3-Large feature backbone.",
    ),
    BackboneAlias(
        name="efficientnet_b0",
        family="EfficientNet",
        model_name="efficientnet_b0",
        stage_channels=(24, 40, 112, 320),
        aliases=("EfficientNet", "EfficientNet-B0"),
        summary="EfficientNet-B0 feature backbone.",
    ),
    BackboneAlias(
        name="convnext_tiny",
        family="ConvNeXt",
        model_name="convnext_tiny",
        stage_channels=(96, 192, 384, 768),
        aliases=("ConvNeXt", "ConvNeXt-Tiny"),
        summary="ConvNeXt-Tiny feature backbone.",
    ),
    BackboneAlias(
        name="swin_tiny_patch4_window7_224",
        family="Swin Transformer",
        model_name="swin_tiny_patch4_window7_224",
        stage_channels=(96, 192, 384, 768),
        aliases=("Swin", "Swin Transformer", "Swin-T", "swin_tiny"),
        summary="Swin-Tiny feature backbone.",
    ),
    BackboneAlias(
        name="vit_base_patch16_224",
        family="Vision Transformer",
        model_name="vit_base_patch16_224",
        stage_channels=(768, 768, 768, 768),
        aliases=("ViT", "Vision Transformer", "ViT-B/16", "vit_base"),
        summary="ViT-Base feature backbone.",
    ),
)


def list_backbone_aliases(pattern: str | None = None) -> list[str]:
    names = [spec.name for spec in BACKBONE_ALIASES]
    if pattern is None:
        return sorted(names)
    token = str(pattern).strip().lower()
    if not token:
        return sorted(names)
    return sorted(
        spec.name
        for spec in BACKBONE_ALIASES
        if token in spec.name.lower()
        or token in spec.family.lower()
        or any(token in alias.lower() for alias in spec.aliases)
    )


def inspect_backbone_alias(name: str) -> dict[str, Any]:
    spec = resolve_backbone_alias(name)
    return {
        "name": spec.name,
        "family": spec.family,
        "model_name": spec.model_name,
        "aliases": list(spec.aliases),
        "out_indices": list(spec.out_indices),
        "feature_channels": list(spec.stage_channels),
        "summary": spec.summary,
    }


def resolve_backbone_alias(name: str) -> BackboneAlias:
    requested = str(name).strip()
    if not requested:
        raise UnknownBackboneError("Backbone name must be a non-empty string.")
    registry = _alias_registry()
    matched = registry.get(_registry_key(requested))
    if matched is not None:
        return matched
    raise UnknownBackboneError(_unknown_backbone_message(requested))


def select_feature_channels(
    spec: BackboneAlias,
    out_indices: tuple[int, ...] | list[int] | str | None,
) -> tuple[int, ...]:
    normalized = normalize_out_indices(spec, out_indices)
    zero_based = _out_indices_are_zero_based(spec, normalized)
    channels: list[int] = []
    for index in normalized:
        position = index if zero_based else index - 1
        channels.append(spec.stage_channels[position])
    return tuple(channels)


def normalize_out_indices(
    spec: BackboneAlias,
    out_indices: tuple[int, ...] | list[int] | str | None,
) -> tuple[int, ...]:
    if out_indices is None:
        return spec.out_indices
    if isinstance(out_indices, str):
        try:
            values = tuple(
                int(item.strip()) for item in out_indices.split(",") if item.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "Backbone out_indices must be an iterable of stage integers or "
                "a comma-separated string."
            ) from exc
    else:
        try:
            values = tuple(int(item) for item in out_indices)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Backbone out_indices must be an iterable of stage integers or "
                "a comma-separated string."
            ) from exc
    if not values:
        raise ValueError("Backbone out_indices must contain at least one stage index.")
    zero_based = _out_indices_are_zero_based(spec, values)
    max_allowed = len(spec.stage_channels) - 1 if zero_based else len(spec.stage_channels)
    min_allowed = 0 if zero_based else 1
    invalid = [index for index in values if index < min_allowed or index > max_allowed]
    if invalid:
        available_indices = (
            range(0, len(spec.stage_channels)) if zero_based else spec.out_indices
        )
        available = ", ".join(str(index) for index in available_indices)
        raise ValueError(
            f"Backbone '{spec.name}' does not expose stage index {invalid[0]}. "
            f"Available out_indices: {available}."
        )
    return values


def _out_indices_are_zero_based(spec: BackboneAlias, values: tuple[int, ...]) -> bool:
    del spec
    return any(index == 0 for index in values)


def _alias_registry() -> dict[str, BackboneAlias]:
    registry: dict[str, BackboneAlias] = {}
    for spec in BACKBONE_ALIASES:
        for value in (spec.name, spec.model_name, *spec.aliases):
            key = _registry_key(value)
            existing = registry.get(key)
            if existing is not None and existing is not spec:
                raise RuntimeError(
                    f"Backbone alias '{value}' is ambiguous for "
                    f"'{existing.name}' and '{spec.name}'."
                )
            registry[key] = spec
    return registry


def _unknown_backbone_message(requested: str) -> str:
    names = list_backbone_aliases()
    aliases = [alias for spec in BACKBONE_ALIASES for alias in spec.aliases]
    candidates = names + aliases
    nearby = get_close_matches(requested, candidates, n=5, cutoff=0.25)
    nearby_text = ", ".join(nearby) if nearby else "<none>"
    supported = ", ".join(names)
    return (
        f"Unknown backbone '{requested}'. Supported backbone aliases: {supported}. "
        f"Nearby backbone aliases/names: {nearby_text}. Use list_backbones() "
        "to discover supported native aliases."
    )


def _registry_key(value: str) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )
