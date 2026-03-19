from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from importlib import import_module
from typing import Any

from .detectors._deps import require_dependency
from .extensions import HEADS, NECKS


class ModelPatchError(ValueError):
    """Raised when runtime model patching cannot be completed safely."""


def list_available_encoders(pattern: str | None = None) -> list[str]:
    """Return runtime-available timm encoder names."""
    require_dependency("timm", "list_available_encoders")
    timm = import_module("timm")
    if pattern:
        return sorted(timm.list_models(pattern))
    return sorted(timm.list_models())


def print_available_encoders(pattern: str | None = None) -> None:
    for name in list_available_encoders(pattern=pattern):
        print(name)


def list_available_necks() -> list[str]:
    return _list_registered_model_components(kind="neck")


def print_available_necks() -> None:
    for name in list_available_necks():
        print(name)


def list_available_heads() -> list[str]:
    return _list_registered_model_components(kind="head")


def print_available_heads() -> None:
    for name in list_available_heads():
        print(name)


def apply_runtime_model_overrides(
    model_cfg: MutableMapping[str, Any],
    *,
    encoder_name: str | None = None,
    encoder_pretrained: bool = True,
    encoder_in_chans: int | None = None,
    backbone_cfg: MutableMapping[str, Any] | None = None,
    feature_channels: Sequence[int] | None = None,
    num_classes: int | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Patch a model config in place and return a structured patch summary."""
    if not isinstance(model_cfg, MutableMapping):
        raise TypeError("`model_cfg` must be a mutable mapping.")
    if encoder_name is not None and backbone_cfg is not None:
        raise TypeError("Use either `encoder_name` or `backbone_cfg`, not both.")

    patch_summary: list[dict[str, Any]] = []
    issues: list[str] = []

    resolved_feature_channels = (
        [int(channel) for channel in feature_channels]
        if feature_channels is not None
        else None
    )
    if encoder_name is not None:
        resolved_feature_channels = inspect_encoder_feature_channels(
            encoder_name,
            pretrained=encoder_pretrained,
            in_chans=encoder_in_chans,
        )
        model_cfg["backbone"] = {
            "type": "TimmEncoder",
            "model_name": encoder_name,
            "features_only": True,
            "pretrained": encoder_pretrained,
            "in_chans": encoder_in_chans,
        }
        patch_summary.append(
            {
                "path": "model.backbone",
                "field": "replace",
                "value": {
                    "type": "TimmEncoder",
                    "model_name": encoder_name,
                    "in_chans": encoder_in_chans,
                },
            }
        )
    elif backbone_cfg is not None:
        model_cfg["backbone"] = dict(backbone_cfg)
        patch_summary.append(
            {
                "path": "model.backbone",
                "field": "replace",
                "value": dict(backbone_cfg),
            }
        )

    neck_cfg = model_cfg.get("neck")
    if isinstance(neck_cfg, MutableMapping) and resolved_feature_channels is not None:
        current_in_channels = neck_cfg.get("in_channels")
        replacement = _coerce_channel_value(
            current_in_channels,
            resolved_feature_channels,
            strict=strict,
            issues=issues,
            path="model.neck.in_channels",
        )
        if replacement is not None:
            _set_field(neck_cfg, "in_channels", replacement, "model.neck", patch_summary)
        if "num_outs" in neck_cfg and isinstance(neck_cfg.get("num_outs"), int):
            if isinstance(replacement, list):
                min_outs = len(replacement)
                if int(neck_cfg["num_outs"]) < min_outs:
                    _set_field(neck_cfg, "num_outs", min_outs, "model.neck", patch_summary)

    neck_out_channels = _resolve_neck_output_channels(model_cfg.get("neck"))
    _patch_head_consumers(
        model_cfg,
        num_classes=num_classes,
        neck_out_channels=neck_out_channels,
        feature_channels=resolved_feature_channels,
        strict=strict,
        patch_summary=patch_summary,
        issues=issues,
    )
    _patch_roi_extractors(
        model_cfg,
        neck_out_channels=neck_out_channels,
        patch_summary=patch_summary,
    )
    _patch_transformer_components(
        model_cfg,
        neck_out_channels=neck_out_channels,
        feature_channels=resolved_feature_channels,
        patch_summary=patch_summary,
    )

    if strict and issues:
        raise ModelPatchError("; ".join(issues))
    return patch_summary


def inspect_encoder_feature_channels(
    encoder_name: str,
    *,
    pretrained: bool = True,
    in_chans: int | None = None,
) -> list[int]:
    """Return the feature channels produced by a timm encoder."""
    require_dependency("timm", "inspect_encoder_feature_channels")

    timm = import_module("timm")
    kwargs: dict[str, Any] = {
        "features_only": True,
        "pretrained": pretrained,
    }
    if in_chans is not None:
        kwargs["in_chans"] = in_chans

    model = timm.create_model(encoder_name, **kwargs)
    feature_info = getattr(model, "feature_info", None)
    if feature_info is None:
        raise ModelPatchError(
            f"Encoder '{encoder_name}' does not expose timm feature metadata required for auto patching."
        )

    if hasattr(feature_info, "channels"):
        channels = list(feature_info.channels())
    else:
        info = getattr(feature_info, "info", None)
        if not isinstance(info, Sequence):
            raise ModelPatchError(
                f"Encoder '{encoder_name}' does not expose usable feature metadata for auto patching."
            )
        channels = [int(item["num_chs"]) for item in info]

    if not channels:
        raise ModelPatchError(
            f"Encoder '{encoder_name}' returned no feature channels for auto patching."
        )
    return [int(channel) for channel in channels]


def patch_backbone_input_channels(
    model_cfg: MutableMapping[str, Any],
    in_channels: int,
) -> list[dict[str, Any]]:
    patch_summary: list[dict[str, Any]] = []
    backbone_cfg = model_cfg.get("backbone")
    if isinstance(backbone_cfg, MutableMapping) and "in_channels" in backbone_cfg:
        _set_field(backbone_cfg, "in_channels", int(in_channels), "model.backbone", patch_summary)
    return patch_summary


def patch_model_num_classes(
    model_cfg: MutableMapping[str, Any],
    num_classes: int,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    patch_summary: list[dict[str, Any]] = []
    issues: list[str] = []
    _patch_head_consumers(
        model_cfg,
        num_classes=num_classes,
        neck_out_channels=_resolve_neck_output_channels(model_cfg.get("neck")),
        feature_channels=None,
        strict=strict,
        patch_summary=patch_summary,
        issues=issues,
    )
    if strict and issues:
        raise ModelPatchError("; ".join(issues))
    return patch_summary


_FALLBACK_NATIVE_HEADS = ("FCOSHead", "RetinaHead")
_FALLBACK_NATIVE_NECKS = ("ChannelMapper", "FPN")


def _list_registered_model_components(kind: str) -> list[str]:
    registry = {"head": HEADS, "neck": NECKS}.get(kind)
    if registry is None:
        return []
    names = sorted(registry.names())
    if names:
        return names
    if kind == "head":
        return sorted(_FALLBACK_NATIVE_HEADS)
    if kind == "neck":
        return sorted(_FALLBACK_NATIVE_NECKS)
    return []


def _patch_head_consumers(
    node: Any,
    *,
    num_classes: int | None,
    neck_out_channels: int | None,
    feature_channels: list[int] | None,
    strict: bool,
    patch_summary: list[dict[str, Any]],
    issues: list[str],
    path: str = "model",
) -> None:
    if isinstance(node, MutableMapping):
        for key, value in node.items():
            child_path = f"{path}.{key}"
            if _is_head_key(key):
                _patch_single_head(
                    value,
                    path=child_path,
                    num_classes=num_classes,
                    neck_out_channels=neck_out_channels,
                    feature_channels=feature_channels,
                    strict=strict,
                    patch_summary=patch_summary,
                    issues=issues,
                )
            else:
                _patch_head_consumers(
                    value,
                    num_classes=num_classes,
                    neck_out_channels=neck_out_channels,
                    feature_channels=feature_channels,
                    strict=strict,
                    patch_summary=patch_summary,
                    issues=issues,
                    path=child_path,
                )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _patch_head_consumers(
                item,
                num_classes=num_classes,
                neck_out_channels=neck_out_channels,
                feature_channels=feature_channels,
                strict=strict,
                patch_summary=patch_summary,
                issues=issues,
                path=f"{path}[{index}]",
            )


def _patch_single_head(
    head_cfg: Any,
    *,
    path: str,
    num_classes: int | None,
    neck_out_channels: int | None,
    feature_channels: list[int] | None,
    strict: bool,
    patch_summary: list[dict[str, Any]],
    issues: list[str],
) -> None:
    if isinstance(head_cfg, list):
        for index, item in enumerate(head_cfg):
            _patch_single_head(
                item,
                path=f"{path}[{index}]",
                num_classes=num_classes,
                neck_out_channels=neck_out_channels,
                feature_channels=feature_channels,
                strict=strict,
                patch_summary=patch_summary,
                issues=issues,
            )
        return

    if not isinstance(head_cfg, MutableMapping):
        if strict:
            issues.append(f"Unsupported head config at {path}: expected mapping, found {type(head_cfg).__name__}.")
        return

    if num_classes is not None and "num_classes" in head_cfg:
        _set_field(head_cfg, "num_classes", int(num_classes), path, patch_summary)

    source_channels = _resolve_head_source_channels(
        path=path,
        head_cfg=head_cfg,
        neck_out_channels=neck_out_channels,
        feature_channels=feature_channels,
        strict=strict,
        issues=issues,
    )
    if source_channels is None:
        return

    if "in_channels" in head_cfg:
        replacement = _coerce_channel_value(
            head_cfg["in_channels"],
            source_channels,
            strict=strict,
            issues=issues,
            path=f"{path}.in_channels",
        )
        if replacement is not None:
            _set_field(head_cfg, "in_channels", replacement, path, patch_summary)

    if "feat_channels" in head_cfg and isinstance(head_cfg.get("feat_channels"), int) and neck_out_channels is not None:
        _set_field(head_cfg, "feat_channels", int(neck_out_channels), path, patch_summary)

    for key, value in head_cfg.items():
        if _is_head_key(key):
            _patch_single_head(
                value,
                path=f"{path}.{key}",
                num_classes=num_classes,
                neck_out_channels=neck_out_channels,
                feature_channels=feature_channels,
                strict=strict,
                patch_summary=patch_summary,
                issues=issues,
            )
        elif isinstance(value, (MutableMapping, list)):
            _patch_head_consumers(
                value,
                num_classes=num_classes,
                neck_out_channels=neck_out_channels,
                feature_channels=feature_channels,
                strict=strict,
                patch_summary=patch_summary,
                issues=issues,
                path=f"{path}.{key}",
            )


def _resolve_head_source_channels(
    *,
    path: str,
    head_cfg: MutableMapping[str, Any],
    neck_out_channels: int | None,
    feature_channels: list[int] | None,
    strict: bool,
    issues: list[str],
) -> int | list[int] | None:
    if neck_out_channels is not None:
        return int(neck_out_channels)
    if not feature_channels:
        return None
    current = head_cfg.get("in_channels")
    if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        if len(current) == len(feature_channels):
            return list(feature_channels)
        if strict:
            issues.append(
                f"Cannot safely patch {path}.in_channels without a neck: expected {len(current)} feature channels, got {len(feature_channels)}."
            )
        return None
    return int(feature_channels[-1])


def _coerce_channel_value(
    current_value: Any,
    source_channels: int | list[int],
    *,
    strict: bool,
    issues: list[str],
    path: str,
) -> Any | None:
    if isinstance(current_value, int):
        if isinstance(source_channels, list):
            return int(source_channels[-1])
        return int(source_channels)

    if isinstance(current_value, Sequence) and not isinstance(current_value, (str, bytes)):
        if isinstance(source_channels, list):
            if len(source_channels) < len(current_value):
                if strict:
                    issues.append(
                        f"Cannot safely patch {path}: expected {len(current_value)} channels, got {len(source_channels)}."
                    )
                return None
            if len(source_channels) == len(current_value):
                return list(source_channels)
            # When the source exposes more features than the consumer expects,
            # keep the deepest levels so FPN/ChannelMapper style consumers stay valid.
            return list(source_channels[-len(current_value):])
        return [int(source_channels) for _ in current_value]

    if strict:
        issues.append(f"Unsupported channel field at {path}: {type(current_value).__name__}.")
    return None


def _resolve_neck_output_channels(neck_cfg: Any) -> int | None:
    if not isinstance(neck_cfg, MutableMapping):
        return None
    out_channels = neck_cfg.get("out_channels")
    if isinstance(out_channels, int):
        return int(out_channels)
    return None


def _set_field(
    node: MutableMapping[str, Any],
    field: str,
    value: Any,
    path: str,
    patch_summary: list[dict[str, Any]],
) -> None:
    previous = node.get(field)
    if previous == value:
        return
    node[field] = value
    patch_summary.append(
        {
            "path": path,
            "field": field,
            "before": previous,
            "after": value,
        }
    )


def _is_head_key(key: str) -> bool:
    return key == "head" or key.endswith("_head")


def _patch_roi_extractors(
    node: Any,
    *,
    neck_out_channels: int | None,
    patch_summary: list[dict[str, Any]],
    path: str = "model",
) -> None:
    if neck_out_channels is None:
        return
    if isinstance(node, MutableMapping):
        for key, value in node.items():
            child_path = f"{path}.{key}"
            if key.endswith("_roi_extractor") and isinstance(value, MutableMapping):
                if isinstance(value.get("out_channels"), int):
                    _set_field(value, "out_channels", int(neck_out_channels), child_path, patch_summary)
            if isinstance(value, (MutableMapping, list)):
                _patch_roi_extractors(
                    value,
                    neck_out_channels=neck_out_channels,
                    patch_summary=patch_summary,
                    path=child_path,
                )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _patch_roi_extractors(
                item,
                neck_out_channels=neck_out_channels,
                patch_summary=patch_summary,
                path=f"{path}[{index}]",
            )


def _patch_transformer_components(
    model_cfg: MutableMapping[str, Any],
    *,
    neck_out_channels: int | None,
    feature_channels: list[int] | None,
    patch_summary: list[dict[str, Any]],
) -> None:
    if neck_out_channels is not None:
        for key in ("encoder", "decoder", "bbox_head"):
            section = model_cfg.get(key)
            if isinstance(section, MutableMapping):
                _patch_embed_dim_fields(
                    section,
                    embed_dims=int(neck_out_channels),
                    patch_summary=patch_summary,
                    path=f"model.{key}",
                )
        positional_encoding = model_cfg.get("positional_encoding")
        if isinstance(positional_encoding, MutableMapping) and isinstance(positional_encoding.get("num_feats"), int):
            _set_field(
                positional_encoding,
                "num_feats",
                max(int(neck_out_channels) // 2, 1),
                "model.positional_encoding",
                patch_summary,
            )

    if isinstance(model_cfg.get("num_feature_levels"), int):
        inferred_levels = _resolve_num_feature_levels(model_cfg.get("neck"), feature_channels)
        if inferred_levels is not None:
            _set_field(model_cfg, "num_feature_levels", inferred_levels, "model", patch_summary)


def _patch_embed_dim_fields(
    node: MutableMapping[str, Any],
    *,
    embed_dims: int,
    patch_summary: list[dict[str, Any]],
    path: str,
) -> None:
    for key, value in node.items():
        child_path = f"{path}.{key}"
        if key == "embed_dims" and isinstance(value, int):
            _set_field(node, key, int(embed_dims), path, patch_summary)
        elif isinstance(value, MutableMapping):
            _patch_embed_dim_fields(
                value,
                embed_dims=embed_dims,
                patch_summary=patch_summary,
                path=child_path,
            )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, MutableMapping):
                    _patch_embed_dim_fields(
                        item,
                        embed_dims=embed_dims,
                        patch_summary=patch_summary,
                        path=f"{child_path}[{index}]",
                    )


def _resolve_num_feature_levels(neck_cfg: Any, feature_channels: list[int] | None) -> int | None:
    if isinstance(neck_cfg, MutableMapping):
        if isinstance(neck_cfg.get("num_outs"), int):
            return int(neck_cfg["num_outs"])
        in_channels = neck_cfg.get("in_channels")
        if isinstance(in_channels, Sequence) and not isinstance(in_channels, (str, bytes)):
            return len(in_channels)
    if feature_channels:
        return len(feature_channels)
    return None
