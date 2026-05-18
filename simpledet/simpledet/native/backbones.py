"""Native backbone support for the Lightning backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import ENCODERS
from ..suite.backbone_aliases import BACKBONE_ALIASES

require_dependency("torch", "native backbones")
import torch.nn as nn  # noqa: E402


@dataclass(slots=True, frozen=True)
class BackboneSpec:
    name: str
    source: str
    feature_channels: tuple[int, ...]


@ENCODERS.register(
    "timm",
    aliases=("TimmEncoder",),
    required_dependencies=(("torch", "cpu"), ("timm", "timm")),
    tensor_contracts=("features_only_backbone", "feature_channels"),
    validation_status="runtime_validated",
    family="backbone",
    summary="TIMM feature-map backbone adapter.",
)
class TimmFeatureBackbone(nn.Module):
    """Feature-extracting backbone backed by timm."""

    def __init__(
        self,
        *,
        model_name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        out_indices: tuple[int, ...] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        require_dependency("timm", "native backbones")
        import timm

        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)
        self.in_channels = int(in_channels)
        self.out_indices = tuple(out_indices) if out_indices is not None else None
        kwargs: dict[str, Any] = {
            "in_chans": self.in_channels,
            "pretrained": self.pretrained,
            "features_only": True,
        }
        kwargs.update(dict(timm_kwargs or {}))
        if self.out_indices is not None:
            kwargs["out_indices"] = self.out_indices
        kwargs["features_only"] = bool(kwargs.get("features_only", True))
        self.encoder = timm.create_model(self.model_name, **kwargs)
        self.feature_channels = _extract_feature_channels(self.encoder)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return tuple(self.encoder(x))


for _backbone_alias in BACKBONE_ALIASES:
    ENCODERS.register(
        _backbone_alias.name,
        aliases=_backbone_alias.aliases,
        required_dependencies=(("torch", "cpu"), ("timm", "timm")),
        tensor_contracts=("features_only_backbone", "feature_channels"),
        validation_status="metadata_validated",
        family=_backbone_alias.family,
        summary=_backbone_alias.summary,
    )(TimmFeatureBackbone)


def build_native_backbone(encoder_plan) -> tuple[Any, BackboneSpec]:
    """Build a native backbone instance from a native component plan."""
    if encoder_plan is None:
        raise ValueError("A native backbone requires a non-null encoder plan.")

    normalized_type = str(encoder_plan.type).strip().lower()
    if normalized_type == "timm":
        out_indices = encoder_plan.params.get("out_indices")
        out_indices_value = _coerce_out_indices(out_indices)
        extra = {
            key: value
            for key, value in encoder_plan.params.items()
            if key not in {"model_name", "pretrained", "in_channels", "out_indices"}
        }
        backbone = TimmFeatureBackbone(
            model_name=encoder_plan.params["model_name"],
            pretrained=bool(encoder_plan.params.get("pretrained", True)),
            in_channels=int(encoder_plan.params.get("in_channels") or 3),
            out_indices=out_indices_value,
            timm_kwargs=extra,
        )
        spec = BackboneSpec(
            name=encoder_plan.params["model_name"],
            source="timm",
            feature_channels=tuple(backbone.feature_channels),
        )
        return backbone, spec

    params = dict(encoder_plan.params)
    configured_channels = tuple(
        int(channel) for channel in params.pop("feature_channels", ())
    )
    factory = ENCODERS.get(encoder_plan.type)
    if factory is TimmFeatureBackbone:
        timm_kwargs = dict(params.pop("timm_kwargs", {}) or {})
        for key in list(params):
            if key not in {"model_name", "pretrained", "in_channels", "out_indices"}:
                timm_kwargs[key] = params.pop(key)
        if timm_kwargs:
            params["timm_kwargs"] = timm_kwargs
    backbone = factory(**params)
    runtime_channels = getattr(backbone, "feature_channels", None)
    feature_channels = (
        tuple(int(channel) for channel in runtime_channels)
        if runtime_channels is not None
        else configured_channels
    )
    spec = BackboneSpec(
        name=str(encoder_plan.type),
        source=str(getattr(encoder_plan, "source", "custom") or "custom"),
        feature_channels=feature_channels,
    )
    return backbone, spec


def _coerce_out_indices(out_indices: Any) -> tuple[int, ...] | None:
    if out_indices is None:
        return None
    if isinstance(out_indices, str):
        raw = [item.strip() for item in out_indices.split(",") if item.strip()]
        return tuple(int(item) for item in raw)
    return tuple(int(item) for item in out_indices)


def _extract_feature_channels(model: Any) -> tuple[int, ...]:
    feature_info = getattr(model, "feature_info", None)
    if feature_info is None:
        raise ValueError("Native timm backbones require feature_info metadata.")
    if hasattr(feature_info, "channels"):
        return tuple(int(channel) for channel in feature_info.channels())
    info = getattr(feature_info, "info", None)
    if not isinstance(info, list):
        raise ValueError("Native timm backbone feature_info is not usable.")
    return tuple(int(item["num_chs"]) for item in info)
