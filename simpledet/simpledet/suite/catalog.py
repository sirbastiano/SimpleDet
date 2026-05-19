"""Canonical suite builders and architecture catalog."""

from __future__ import annotations

from dataclasses import replace
from difflib import get_close_matches
from importlib import import_module
from typing import Any

from .backbone_aliases import (
    inspect_backbone_alias,
    list_backbone_aliases,
    normalize_out_indices,
    resolve_backbone_alias,
    select_feature_channels,
)
from .specs import DecoderSpec, DetectorSpec, EncoderSpec, HeadSpec, NeckSpec


_DEFAULT_ALIAS_OUT_INDICES = (1, 2, 3, 4)
_OUT_INDICES_UNSET = object()
_TIMM_BACKBONE_PREFIX = "timm:"
_TRANSFORMER_HEAD_KEYS = {
    "detr",
    "detrhead",
    "conditionaldetr",
    "conditionaldetrhead",
    "dabdetr",
    "dabdetrhead",
    "deformabledetr",
    "deformabledetrhead",
    "dino",
    "dinohead",
}

_PLANNED_TRANSFORMER_VARIANTS = {
    "detrnext": "DETR-next",
    "detrv2": "DETRv2",
    "detr3d": "DETR3D",
    "deformabledetrnext": "Deformable DETR-next",
    "deformabledetrv2": "Deformable DETR v2",
    "deformabledetr2": "Deformable DETR v2",
    "conditionaldetrnext": "Conditional DETR-next",
    "conditionaldetrv2": "Conditional DETR v2",
    "conditionaldetr2": "Conditional DETR v2",
    "dabdetrnext": "DAB-DETR-next",
    "dabdetrv2": "DAB-DETR v2",
    "dabdetr2": "DAB-DETR v2",
    "dinov2": "DINOv2",
    "dino2": "DINOv2",
}


ARCHITECTURE_FAMILIES: dict[str, str] = {
    "retinanet": "dense",
    "retina": "dense",
    "cornernet": "dense",
    "detr": "transformer",
    "deformable_detr": "transformer",
    "deformabledetr": "transformer",
    "conditional_detr": "transformer",
    "dab_detr": "transformer",
    "dabdetr": "transformer",
    "dino": "transformer",
    "fcos": "dense",
    "atss": "dense",
    "fsaf": "dense",
    "free_anchor": "dense",
    "gfl": "dense",
    "gfocalv2": "dense",
    "vfnet": "dense",
    "fovea": "dense",
    "paa": "dense",
    "reppoints": "dense",
    "yolof": "dense",
    "ddod": "dense",
    "auto_assign": "dense",
    "nas_fcos": "dense",
    "centernet": "dense",
    "yolo": "dense",
    "yolo3": "dense",
    "yolo_v3": "dense",
    "yolov3": "dense",
    "yolov5": "dense",
    "yolov6": "dense",
    "yolov7": "dense",
    "yolov8": "dense",
    "yolox": "dense",
    "rtmdet": "dense",
    "tood": "dense",
    "ssd": "dense",
    "efficientdet": "dense",
    "sabl": "dense",
    "solov2": "dense",
    "faster_rcnn": "roi",
    "faster-rcnn": "roi",
    "fast_rcnn": "roi",
    "fast-rcnn": "roi",
    "rpn": "proposal",
    "mask_rcnn": "roi",
    "mask-rcnn": "roi",
    "grid_rcnn": "roi",
    "cascade_rcnn": "roi",
    "cascade_mask_rcnn": "roi",
    "libra_rcnn": "roi",
    "double_head_rcnn": "roi",
    "dynamic_rcnn": "roi",
    "sparse_rcnn": "roi",
}


def _normalize_architecture_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _compact_architecture_name(normalized: str) -> str:
    return "".join(char for char in normalized if char.isalnum())


def _compact_component_name(name: str | None) -> str:
    return "".join(char for char in str(name or "").strip().lower() if char.isalnum())


def _is_transformer_head_request(name: str | None, head_cfg: dict[str, Any] | None) -> bool:
    requested = None
    if head_cfg:
        requested = head_cfg.get("type")
    token = _compact_component_name(requested or name)
    return token in _TRANSFORMER_HEAD_KEYS


def resolve_architecture_name(name: str) -> str:
    normalized = _normalize_architecture_name(name)
    compact = _compact_architecture_name(normalized)
    if compact.startswith("yolof"):
        return "yolof"
    if compact.startswith("yolox"):
        return "yolox"
    if compact in _PLANNED_TRANSFORMER_VARIANTS:
        return normalized
    if compact == "detr":
        return "detr"
    if compact == "deformabledetr":
        return "deformable_detr"
    if compact == "conditionaldetr":
        return "conditional_detr"
    if compact == "dabdetr":
        return "dab_detr"
    if compact == "dino":
        return "dino"
    if compact.startswith("deformabledetr"):
        return "deformable_detr"
    if compact.startswith("conditionaldetr"):
        return "conditional_detr"
    if compact.startswith("dabdetr"):
        return "dab_detr"
    if compact.startswith("dino"):
        return "dino"
    if compact.startswith("detr"):
        return "detr"
    if compact in {
        "fasterrcnn",
        "fasterrcnncv",
        "fasterrcnnnext",
    }:
        return "faster_rcnn"
    if compact in {"fastrcnn", "fastrcnncv", "fastrcnnnext"}:
        return "fast_rcnn"
    if compact in {"rpn", "rpndetector", "regionproposalnetwork", "rpnhead"}:
        return "rpn"
    if compact in {
        "maskrcnn",
        "maskrcnncv",
        "maskrcnnnext",
    }:
        return "mask_rcnn"
    if compact in {"gridrcnn", "gridrcnncv", "gridrcnnnext"}:
        return "grid_rcnn"
    if compact in {"cascadercnn", "cascadercnncv", "cascadercnnnext"}:
        return "cascade_rcnn"
    if compact in {"cascademaskrcnn", "cascademaskrcnncv", "cascademaskrcnnnext"}:
        return "cascade_mask_rcnn"
    if compact in {"librarcnn", "librarcnncv", "librarcnnnext"}:
        return "libra_rcnn"
    if compact in {"doubleheadrcnn", "doubleheadrcnncv", "doubleheadrcnnnext"}:
        return "double_head_rcnn"
    if compact in {"dynamicrcnn", "dynamicrcnncv", "dynamicrcnnnext"}:
        return "dynamic_rcnn"
    if compact in {"sparsercnn", "sparsercnncv", "sparsercnnnext"}:
        return "sparse_rcnn"
    if compact.startswith("cornernet"):
        return "cornernet"
    if compact.startswith("retinanet"):
        return "retinanet"
    if compact.startswith("fcos"):
        return "fcos"
    if compact.startswith("atss"):
        return "atss"
    if compact.startswith("fsaf"):
        return "fsaf"
    if compact.startswith("freeanchor"):
        return "free_anchor"
    if compact.startswith("gflv2") or compact.startswith("gfocalv2"):
        return "gfocalv2"
    if compact.startswith("gfl"):
        return "gfl"
    if compact.startswith("paa"):
        return "paa"
    if compact.startswith("ddod"):
        return "ddod"
    if compact.startswith("autoassign"):
        return "auto_assign"
    if compact.startswith("nasfcos"):
        return "nas_fcos"
    if compact.startswith("vfnet"):
        return "vfnet"
    if compact.startswith("fovea"):
        return "fovea"
    if compact.startswith("reppoints"):
        return "reppoints"
    if compact.startswith("centernet"):
        return "centernet"
    if compact.startswith("yolo"):
        return "yolo"
    if compact.startswith("rtmdet"):
        return "rtmdet"
    if compact.startswith("efficientdet"):
        return "efficientdet"
    if compact.startswith("solov2"):
        return "solov2"
    if compact.startswith("tood"):
        return "tood"
    if compact.startswith("ssd"):
        return "ssd"
    if compact.startswith("sabl"):
        return "sabl"
    if compact.startswith("fasterrcnn"):
        return "faster_rcnn"
    if compact.startswith("fastrcnn"):
        return "fast_rcnn"
    if compact.startswith("rpn"):
        return "rpn"
    if compact.startswith("maskrcnn"):
        return "mask_rcnn"
    if compact.startswith("gridrcnn"):
        return "grid_rcnn"
    if compact.startswith("cascadercnn"):
        return "cascade_rcnn"
    if compact.startswith("cascademaskrcnn"):
        return "cascade_mask_rcnn"
    if compact.startswith("librarcnn"):
        return "libra_rcnn"
    if compact.startswith("doubleheadrcnn"):
        return "double_head_rcnn"
    if compact.startswith("dynamicrcnn"):
        return "dynamic_rcnn"
    if compact.startswith("sparsercnn"):
        return "sparse_rcnn"
    return normalized


_DENSE_DEFAULT_HEAD_BY_ARCHITECTURE = {
    "cornernet": "CornerNetHead",
    "retinanet": "RetinaHead",
    "fcos": "FCOSHead",
    "atss": "ATSSHead",
    "fsaf": "FSAFHead",
    "free_anchor": "FreeAnchorRetinaHead",
    "gfl": "GFLHead",
    "gfocalv2": "GFLV2Head",
    "gflv2": "GFLV2Head",
    "paa": "PAAHead",
    "ddod": "DDODHead",
    "auto_assign": "AutoAssignHead",
    "nas_fcos": "NASFCOSHead",
    "yolo": "YOLOXHead",
    "yolo3": "YOLOXHead",
    "yolo_v3": "YOLOXHead",
    "yolov3": "YOLOXHead",
    "yolov4": "YOLOXHead",
    "yolov5": "YOLOXHead",
    "yolov6": "YOLOXHead",
    "yolov7": "YOLOXHead",
    "yolov8": "YOLOXHead",
    "yolov9": "YOLOXHead",
    "yolov10": "YOLOXHead",
    "yolox": "YOLOXHead",
    "rtmdet": "RTMDetHead",
    "tood": "TOODHead",
    "ssd": "SSDHead",
    "efficientdet": "EfficientDetHead",
    "sabl": "ATSSHead",
    "solov2": "FCOSHead",
    "vfnet": "VFNetHead",
    "fovea": "FoveaHead",
    "foveabox": "FoveaHead",
    "reppoints": "RepPointsHead",
    "yolof": "YOLOFHead",
    "centernet": "CenterNetHead",
}


_ARCHITECTURE_SUGGESTION_ALIASES = (
    "RetinaNet",
    "FCOS",
    "ATSS",
    "FSAF",
    "FoveaBox",
    "FOVEA",
    "FreeAnchor",
    "GFL",
    "GFocalV2",
    "VFNet",
    "PAA",
    "RepPoints",
    "YOLOF",
    "TOOD",
    "DDOD",
    "AutoAssign",
    "NAS-FCOS",
    "CenterNet",
    "CornerNet",
    "DETR",
    "Conditional DETR",
    "DAB-DETR",
    "Deformable DETR",
    "DINO",
    "YOLOX",
    "RTMDet",
    "SSD",
    "SSD300",
    "EfficientDet",
    "EfficientDet-D0",
    "Grid R-CNN",
    "Cascade R-CNN",
    "Sparse R-CNN",
)

_LIGHTWEIGHT_DEFAULT_BACKBONE_BY_ARCHITECTURE = {
    "cornernet": "resnet18",
    "yolo": "cspdarknet53",
    "yolox": "cspdarknet53",
    "rtmdet": "cspnext_tiny",
    "ssd": "mobilenetv2_100",
    "efficientdet": "efficientnet_b0",
    "centernet": "resnet18",
}

_LIGHTWEIGHT_DEFAULT_NECK_BY_ARCHITECTURE: dict[str, dict[str, Any]] = {
    "yolo": {"name": "YOLOXPAFPN", "out_channels": 256, "num_outs": 4},
    "yolox": {"name": "YOLOXPAFPN", "out_channels": 256, "num_outs": 4},
    "rtmdet": {"name": "YOLOXPAFPN", "out_channels": 256, "num_outs": 4},
    "ssd": {"name": "SSDNeck", "out_channels": 256, "num_outs": 6},
    "efficientdet": {"name": "BiFPN", "out_channels": 64, "num_outs": 5},
    "centernet": {"name": "FPN", "out_channels": 256, "num_outs": 4},
    "cornernet": {"name": "FPN", "out_channels": 256, "num_outs": 4},
}

_YOLO_COMPATIBLE_NECKS = {"yoloxpafpn"}


def _resolve_detector_name(architecture: str | None, name: str | None) -> str:
    if architecture is None:
        if name is None:
            raise ValueError("build_detector() missing required architecture name.")
        return str(name)
    if name is not None:
        resolved_architecture = resolve_architecture_name(str(architecture))
        resolved_name = resolve_architecture_name(str(name))
        if resolved_architecture != resolved_name:
            raise ValueError(
                "build_detector() received conflicting architecture names: "
                f"{architecture!r} and name={name!r}."
            )
    return str(architecture)


def _neck_is_auto(neck: NeckSpec | None) -> bool:
    if neck is None:
        return True
    return _compact_component_name(neck.name) in {"", "auto"}


def _default_encoder_for_architecture(
    architecture: str,
    *,
    pretrained: bool,
    in_channels: int,
) -> EncoderSpec:
    backbone_name = _LIGHTWEIGHT_DEFAULT_BACKBONE_BY_ARCHITECTURE.get(architecture)
    if backbone_name is not None:
        return build_backbone(
            backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
        )
    return build_encoder(
        "resnet18.a1_in1k",
        source="timm",
        pretrained=pretrained,
        in_channels=in_channels,
    )


def _default_neck_for_architecture(architecture: str) -> NeckSpec | None:
    defaults = _LIGHTWEIGHT_DEFAULT_NECK_BY_ARCHITECTURE.get(architecture)
    if defaults is None:
        return None
    params = dict(defaults)
    name = str(params.pop("name"))
    return build_neck(name, **params)


def _validate_detector_neck_choice(architecture: str, neck: NeckSpec | None) -> None:
    if architecture not in {"yolo", "yolox"} or neck is None:
        return
    requested = _compact_component_name(neck.name)
    configured_type = None
    if neck.neck_cfg:
        configured_type = neck.neck_cfg.get("type")
    if configured_type is not None:
        requested = _compact_component_name(str(configured_type))
    if requested not in _YOLO_COMPATIBLE_NECKS:
        raise ValueError(
            f"YOLO-family detector '{architecture}' requires a YOLOXPAFPN neck; "
            f"got '{neck.name}'."
        )


def _default_dense_head_extra(architecture: str, neck: NeckSpec | None) -> dict[str, Any]:
    if architecture != "reppoints":
        return {}
    num_levels = 4
    if neck is not None and neck.num_outs is not None:
        num_levels = int(neck.num_outs)
    return {"point_strides": tuple(2 ** (level + 3) for level in range(num_levels))}


def _unknown_architecture_message(requested: str, normalized: str) -> str:
    planned_variant = _planned_transformer_variant_name(requested)
    if planned_variant is not None:
        supported = ", ".join(
            ("detr", "conditional_detr", "dab_detr", "deformable_detr", "dino")
        )
        return (
            f"Unsupported transformer variant '{requested}' ({planned_variant}) is planned "
            f"but not available in the native 2D query detector registry yet. "
            f"Use one of: {supported}."
        )
    candidates = sorted(set(ARCHITECTURE_FAMILIES) | set(_ARCHITECTURE_SUGGESTION_ALIASES), key=str.lower)
    nearby = get_close_matches(str(requested), candidates, n=5, cutoff=0.25)
    if not nearby:
        nearby = get_close_matches(str(normalized), candidates, n=5, cutoff=0.25)
    supported = ", ".join(sorted(ARCHITECTURE_FAMILIES))
    suggestions = ", ".join(nearby) if nearby else "<none>"
    return f"Unknown architecture '{requested}'. Supported: {supported}. Suggestions: {suggestions}."


def _planned_transformer_variant_name(name: str) -> str | None:
    compact = _compact_architecture_name(_normalize_architecture_name(name))
    return _PLANNED_TRANSFORMER_VARIANTS.get(compact)

_ROI_DEFAULT_BBOX_HEAD_BY_ARCHITECTURE = {
    "cascade_rcnn": "CascadeBBoxHead",
    "cascade_mask_rcnn": "CascadeBBoxHead",
    "double_head_rcnn": "DoubleConvFCBBoxHead",
    "dynamic_rcnn": "DynamicBBoxHead",
    "fast_rcnn": "Shared2FCBBoxHead",
    "faster_rcnn": "Shared2FCBBoxHead",
    "grid_rcnn": "Shared2FCBBoxHead",
    "libra_rcnn": "Shared2FCBBoxHead",
    "mask_rcnn": "Shared2FCBBoxHead",
    "sparse_rcnn": "SparseRoIHead",
}


def list_native_encoder_families(pattern: str | None = None) -> list[str]:
    return _list_native_families(kind="encoder", pattern=pattern)


def list_native_head_families(pattern: str | None = None) -> list[str]:
    return _list_native_families(kind="head", pattern=pattern)


def list_heads(kind: str | None = None, pattern: str | None = None) -> list[str]:
    """Return registered head names and aliases, optionally filtered by family."""

    family = None
    if kind is not None:
        family = str(kind).strip().lower()
        if family in {"", "all", "*"}:
            family = None
        elif family not in {"dense", "roi", "transformer"}:
            raise ValueError("Head kind must be one of: dense, roi, transformer.")
    token = None
    if pattern is not None:
        token = str(pattern).strip().lower()
        if not token:
            token = None
    if not _native_registries_available():
        return []

    results: list[str] = []
    seen: set[str] = set()
    for metadata in _native_registry("head").entries():
        metadata_family = None if metadata.family is None else str(metadata.family).lower()
        if family is not None and metadata_family != family:
            continue
        for value in (metadata.name, *metadata.aliases):
            if token is not None and token not in value.lower():
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(value)
    return sorted(results, key=str.lower)


def list_native_neck_families(pattern: str | None = None) -> list[str]:
    return _list_native_families(kind="neck", pattern=pattern)


def list_native_detector_families(pattern: str | None = None) -> list[str]:
    return _list_native_families(kind="detector", pattern=pattern)


def list_detectors(family: str | None = None, pattern: str | None = None) -> list[str]:
    """Return supported detector names and aliases."""

    family_filter = _normalize_family_filter(family)
    token = _normalize_pattern_filter(pattern)
    if _native_registries_available():
        return _list_registry_names_and_aliases(
            "detector",
            family=family_filter,
            pattern=token,
        )
    return _list_static_detector_names(family=family_filter, pattern=token)


def inspect_native_encoder_family(name: str) -> dict[str, Any]:
    return _inspect_native_family("encoder", name)


def inspect_native_head_family(name: str) -> dict[str, Any]:
    return _inspect_native_family("head", name)


def inspect_native_neck_family(name: str) -> dict[str, Any]:
    return _inspect_native_family("neck", name)


def inspect_native_detector_family(name: str) -> dict[str, Any]:
    return _inspect_native_family("detector", name)


def list_backbones(pattern: str | None = None) -> list[str]:
    """Return registered native backbone aliases."""
    return list_backbone_aliases(pattern)


def inspect_backbone(name: str) -> dict[str, Any]:
    """Return metadata for a registered native backbone alias."""
    return inspect_backbone_alias(name)


def resolve_native_encoder_family(name: str | None = None) -> Any:
    return _resolve_native_family("encoder", name, default="timm")


def resolve_native_head_family(name: str | None = None) -> Any:
    return _resolve_native_family("head", name, default="RetinaHead")


def resolve_native_neck_family(name: str | None = None) -> Any:
    return _resolve_native_family("neck", name, default="FPN")


def resolve_native_detector_family(name: str | None = None) -> Any:
    return _resolve_native_family("detector", name, default="retinanet")


def _list_native_families(*, kind: str, pattern: str | None = None) -> list[str]:
    if not _native_registries_available():
        return []
    registry = _native_registry(kind)
    names = registry.names()
    if not pattern:
        return names
    token = str(pattern).strip().lower()
    if not token:
        return names
    return [name for name in names if token in name.lower()]


def _normalize_family_filter(family: str | None) -> str | None:
    if family is None:
        return None
    normalized = str(family).strip().lower()
    if normalized in {"", "all", "*"}:
        return None
    valid = {"dense", "proposal", "roi", "transformer"}
    if normalized not in valid:
        expected = ", ".join(sorted(valid))
        raise ValueError(f"Detector family must be one of: {expected}.")
    return normalized


def _normalize_pattern_filter(pattern: str | None) -> str | None:
    if pattern is None:
        return None
    token = str(pattern).strip().lower()
    return token or None


def _list_registry_names_and_aliases(
    kind: str,
    *,
    family: str | None = None,
    pattern: str | None = None,
) -> list[str]:
    registry = _native_registry(kind)
    results: list[str] = []
    seen: set[str] = set()
    for metadata in registry.entries():
        metadata_family = None if metadata.family is None else str(metadata.family).lower()
        if family is not None and metadata_family != family:
            continue
        for value in (metadata.name, *metadata.aliases):
            if pattern is not None and pattern not in value.lower():
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(value)
    return sorted(results, key=str.lower)


def _list_static_detector_names(
    *,
    family: str | None = None,
    pattern: str | None = None,
) -> list[str]:
    values: list[str] = []
    for name, detector_family in ARCHITECTURE_FAMILIES.items():
        if family is not None and detector_family != family:
            continue
        values.append(name)
    for alias in _ARCHITECTURE_SUGGESTION_ALIASES:
        resolved = resolve_architecture_name(alias)
        detector_family = ARCHITECTURE_FAMILIES.get(resolved)
        if family is not None and detector_family != family:
            continue
        values.append(alias)

    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pattern is not None and pattern not in value.lower():
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(value)
    return sorted(results, key=str.lower)


def _inspect_native_family(kind: str, name: str) -> dict[str, Any]:
    if not _native_registries_available():
        _raise_native_registries_error()
    registry = _native_registry(kind)
    return registry.lookup(name).as_dict()


def _resolve_native_family(kind: str, name: str | None, *, default: str) -> Any:
    if not _native_registries_available():
        _raise_native_registries_error()
    registry = _native_registry(kind)
    requested = name
    if requested is None:
        requested = default
    if str(requested).strip().lower() == "auto":
        requested = default
    component_name = _resolve_registered_name(registry, str(requested))
    return registry.get(component_name)


def _resolve_registered_name(registry: Any, name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("Native component name must be a non-empty string.")
    return registry.resolve_name(normalized)


_NATIVE_REGISTRY_IMPORTED: bool = False
_NATIVE_REGISTRY_IMPORT_ERROR: Exception | None = None


def _native_registries_available() -> bool:
    _load_native_registries()
    return _NATIVE_REGISTRY_IMPORT_ERROR is None


def _raise_native_registries_error() -> None:
    _load_native_registries()
    if _NATIVE_REGISTRY_IMPORT_ERROR is None:
        return
    raise ImportError(
        "Native registry components are unavailable. Ensure optional Lightning-native dependencies are installed."
    ) from _NATIVE_REGISTRY_IMPORT_ERROR


def _load_native_registries() -> None:
    global _NATIVE_REGISTRY_IMPORTED, _NATIVE_REGISTRY_IMPORT_ERROR
    if _NATIVE_REGISTRY_IMPORTED and _NATIVE_REGISTRY_IMPORT_ERROR is None:
        return
    try:
        import_module("simpledet.native")
    except Exception as exc:  # pragma: no cover - import-time optional deps
        _NATIVE_REGISTRY_IMPORTED = False
        _NATIVE_REGISTRY_IMPORT_ERROR = exc
    else:
        _NATIVE_REGISTRY_IMPORTED = True
        _NATIVE_REGISTRY_IMPORT_ERROR = None


def _native_registry(kind: str):
    from ..extensions import DETECTORS, ENCODERS, HEADS, NECKS
    if not _native_registries_available():
        _raise_native_registries_error()
    if kind == "encoder":
        return ENCODERS
    if kind == "neck":
        return NECKS
    if kind == "head":
        return HEADS
    if kind == "detector":
        return DETECTORS
    raise ValueError(f"Unknown native component kind '{kind}'. Expected one of: encoder, head, neck, detector.")


def build_encoder(
    name: str,
    *,
    source: str = "timm",
    pretrained: bool = True,
    in_channels: int | None = None,
    backbone_cfg: dict[str, Any] | None = None,
    feature_channels: tuple[int, ...] | list[int] | None = None,
    imports: tuple[str, ...] | list[str] | None = None,
    **extra: Any,
) -> EncoderSpec:
    return EncoderSpec(
        source=source,
        name=name,
        pretrained=pretrained,
        in_channels=in_channels,
        backbone_cfg=backbone_cfg,
        feature_channels=tuple(feature_channels) if feature_channels is not None else None,
        imports=tuple(imports or ()),
        extra=extra,
    )


def build_backbone(
    name: str,
    *,
    pretrained: bool = True,
    in_channels: int | None = None,
    out_indices: tuple[int, ...] | list[int] | str | None | object = _OUT_INDICES_UNSET,
    imports: tuple[str, ...] | list[str] | None = None,
    **extra: Any,
) -> EncoderSpec:
    timm_model_name = _timm_backbone_model_name(name)
    if timm_model_name is not None:
        if out_indices is _OUT_INDICES_UNSET or out_indices is None:
            raise ValueError(
                "TIMM backbone names require explicit out_indices, for example "
                "build_backbone('timm:resnet18', out_indices=(1, 2, 3, 4))."
            )
        return build_encoder(
            timm_model_name,
            source="timm",
            pretrained=pretrained,
            in_channels=in_channels,
            imports=imports,
            out_indices=_normalize_timm_out_indices(out_indices),
            **extra,
        )

    alias = resolve_backbone_alias(name)
    alias_out_indices = (
        _DEFAULT_ALIAS_OUT_INDICES if out_indices is _OUT_INDICES_UNSET else out_indices
    )
    resolved_out_indices = normalize_out_indices(alias, alias_out_indices)
    feature_channels = select_feature_channels(alias, resolved_out_indices)
    backbone_cfg: dict[str, Any] = {
        "type": alias.name,
        "model_name": alias.model_name,
        "pretrained": pretrained,
        "out_indices": resolved_out_indices,
    }
    if in_channels is not None:
        backbone_cfg["in_channels"] = int(in_channels)
    backbone_cfg.update(extra)
    return build_encoder(
        alias.name,
        source="native",
        pretrained=pretrained,
        in_channels=in_channels,
        backbone_cfg=backbone_cfg,
        feature_channels=feature_channels,
        imports=imports,
    )


def _timm_backbone_model_name(name: str) -> str | None:
    requested = str(name).strip()
    if not requested.lower().startswith(_TIMM_BACKBONE_PREFIX):
        return None
    model_name = requested[len(_TIMM_BACKBONE_PREFIX):].strip()
    if not model_name:
        raise ValueError("TIMM backbone names must include a model after 'timm:'.")
    return model_name


def _normalize_timm_out_indices(
    out_indices: tuple[int, ...] | list[int] | str | object,
) -> tuple[int, ...]:
    if isinstance(out_indices, str):
        try:
            values = tuple(
                int(item.strip()) for item in out_indices.split(",") if item.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "TIMM backbone out_indices must be an iterable of non-negative "
                "integers or a comma-separated string."
            ) from exc
    else:
        try:
            values = tuple(int(item) for item in out_indices)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "TIMM backbone out_indices must be an iterable of non-negative "
                "integers or a comma-separated string."
            ) from exc
    if not values:
        raise ValueError("TIMM backbone out_indices must contain at least one stage index.")
    if any(index < 0 for index in values):
        raise ValueError("TIMM backbone out_indices must be non-negative stage indices.")
    return values


def build_neck(
    name: str = "auto",
    *,
    neck_cfg: dict[str, Any] | None = None,
    out_channels: int | None = None,
    num_outs: int | None = None,
    imports: tuple[str, ...] | list[str] | None = None,
    **extra: Any,
) -> NeckSpec:
    return NeckSpec(
        name=name,
        neck_cfg=neck_cfg,
        out_channels=out_channels,
        num_outs=num_outs,
        imports=tuple(imports or ()),
        extra=extra,
    )


def build_head(
    name: str = "auto",
    *,
    head_cfg: dict[str, Any] | None = None,
    num_classes: int | None = None,
    with_mask: bool = False,
    imports: tuple[str, ...] | list[str] | None = None,
    **extra: Any,
) -> HeadSpec:
    native_in_channels = extra.pop("in_channels", None)
    native_out_channels = extra.pop("out_channels", None)
    direct_transformer_head = _is_transformer_head_request(name, head_cfg)
    if native_in_channels is not None or native_out_channels is not None or direct_transformer_head:
        if num_classes is None:
            raise ValueError("`num_classes` is required when building a native head.")
        from .native_plan import ComponentPlan
        from simpledet.native.heads import build_native_head

        params = dict(head_cfg or {})
        params.update(extra)
        head_type = str(params.pop("type", name or "auto"))
        out_channels = native_in_channels if native_in_channels is not None else native_out_channels
        if out_channels is None:
            out_channels = params.get("in_channels") or params.get("hidden_dim") or params.get("embed_dims") or 256
        head, native_spec = build_native_head(
            ComponentPlan(
                kind="head",
                type=head_type,
                params=params,
                imports=tuple(imports or ()),
            ),
            out_channels=int(out_channels),
            num_classes=int(num_classes),
        )
        head.native_head_spec = native_spec
        return head

    return HeadSpec(
        name=name,
        head_cfg=head_cfg,
        num_classes=num_classes,
        with_mask=with_mask,
        imports=tuple(imports or ()),
        extra=extra,
    )


def build_decoder(
    name: str = "auto",
    *,
    decoder_cfg: dict[str, Any] | None = None,
    num_queries: int | None = None,
    embed_dims: int | None = None,
    imports: tuple[str, ...] | list[str] | None = None,
    **extra: Any,
) -> DecoderSpec:
    return DecoderSpec(
        name=name,
        decoder_cfg=decoder_cfg,
        num_queries=num_queries,
        embed_dims=embed_dims,
        imports=tuple(imports or ()),
        extra=extra,
    )


def build_custom_encoder(
    component_type: str,
    *,
    feature_channels: tuple[int, ...] | list[int],
    imports: tuple[str, ...] | list[str],
    backbone_cfg: dict[str, Any] | None = None,
    **extra: Any,
) -> EncoderSpec:
    resolved_cfg = dict(backbone_cfg or {})
    resolved_cfg.setdefault("type", component_type)
    resolved_cfg.update(extra)
    return build_encoder(
        component_type,
        source="config",
        backbone_cfg=resolved_cfg,
        feature_channels=feature_channels,
        imports=imports,
    )


def build_custom_neck(
    component_type: str,
    *,
    imports: tuple[str, ...] | list[str],
    neck_cfg: dict[str, Any] | None = None,
    **extra: Any,
) -> NeckSpec:
    resolved_cfg = None
    if neck_cfg is not None:
        resolved_cfg = dict(neck_cfg)
        resolved_cfg.setdefault("type", component_type)
        resolved_cfg.update(extra)
        extra = {}
    return build_neck(component_type, neck_cfg=resolved_cfg, imports=imports, **extra)


def build_custom_head(
    component_type: str,
    *,
    imports: tuple[str, ...] | list[str],
    head_cfg: dict[str, Any] | None = None,
    num_classes: int | None = None,
    with_mask: bool = False,
    **extra: Any,
) -> HeadSpec:
    resolved_cfg = None
    if head_cfg is not None:
        resolved_cfg = dict(head_cfg)
        resolved_cfg.setdefault("type", component_type)
        resolved_cfg.update(extra)
        extra = {}
    return build_head(
        component_type,
        head_cfg=resolved_cfg,
        num_classes=num_classes,
        with_mask=with_mask,
        imports=imports,
        **extra,
    )


def build_custom_decoder(
    component_type: str,
    *,
    imports: tuple[str, ...] | list[str],
    decoder_cfg: dict[str, Any] | None = None,
    num_queries: int | None = None,
    embed_dims: int | None = None,
    **extra: Any,
) -> DecoderSpec:
    resolved_cfg = None
    if decoder_cfg is not None:
        resolved_cfg = dict(decoder_cfg)
        resolved_cfg.setdefault("type", component_type)
        resolved_cfg.update(extra)
        extra = {}
    return build_decoder(
        component_type,
        decoder_cfg=resolved_cfg,
        num_queries=num_queries,
        embed_dims=embed_dims,
        imports=imports,
        **extra,
    )


def build_detector(
    architecture: str | None = None,
    *,
    name: str | None = None,
    num_classes: int = 1,
    encoder: str | EncoderSpec | None = None,
    backbone: str | EncoderSpec | None = None,
    neck: NeckSpec | None = None,
    head: HeadSpec | None = None,
    decoder: DecoderSpec | None = None,
    in_channels: int = 3,
    pretrained: bool = True,
    build: bool = False,
    strict_auto_adapt: bool = True,
    imports: tuple[str, ...] | list[str] | None = None,
    **overrides: Any,
) -> DetectorSpec | Any:
    if not isinstance(build, bool):
        raise ValueError("`build` must be a boolean.")
    if backbone is not None:
        if encoder is not None:
            raise ValueError("Pass either `encoder` or `backbone`, not both.")
        encoder = _coerce_backbone_argument(
            backbone,
            pretrained=pretrained,
            in_channels=in_channels,
        )
    elif encoder is not None:
        encoder = _coerce_encoder_argument(
            encoder,
            pretrained=pretrained,
            in_channels=in_channels,
        )
    _validate_component_argument("neck", neck, NeckSpec)
    _validate_component_argument("head", head, HeadSpec)
    _validate_component_argument("decoder", decoder, DecoderSpec)

    requested_architecture = _resolve_detector_name(architecture, name)
    normalized_architecture = resolve_architecture_name(requested_architecture)
    family = ARCHITECTURE_FAMILIES.get(normalized_architecture)
    if family is None:
        raise ValueError(_unknown_architecture_message(requested_architecture, normalized_architecture))

    if encoder is None:
        encoder = _default_encoder_for_architecture(
            normalized_architecture,
            pretrained=pretrained,
            in_channels=in_channels,
        )

    if _neck_is_auto(neck):
        neck = _default_neck_for_architecture(normalized_architecture)
    else:
        _validate_detector_neck_choice(normalized_architecture, neck)

    if head is None:
        if family == "dense":
            default_head = _DENSE_DEFAULT_HEAD_BY_ARCHITECTURE.get(
                normalized_architecture,
                "RetinaHead",
            )
            head = build_head(
                default_head,
                num_classes=num_classes,
                with_mask=normalized_architecture == "mask_rcnn",
                **_default_dense_head_extra(normalized_architecture, neck),
            )
        elif family == "roi":
            head = build_head(
                _ROI_DEFAULT_BBOX_HEAD_BY_ARCHITECTURE.get(
                    normalized_architecture,
                    "Shared2FCBBoxHead",
                ),
                num_classes=num_classes,
                with_mask=normalized_architecture in {"mask_rcnn", "cascade_mask_rcnn"},
            )
        elif family == "proposal":
            head = build_head(
                "RPNHead",
                num_classes=1,
                num_anchors=1,
            )
        else:
            head = None
    else:
        head = replace(head, num_classes=num_classes if head.num_classes is None else head.num_classes)

    detector_spec = DetectorSpec(
        architecture=normalized_architecture,
        family=family,
        num_classes=num_classes,
        encoder=encoder,
        neck=neck,
        head=head,
        decoder=decoder,
        strict_auto_adapt=strict_auto_adapt,
        imports=tuple(imports or ()),
        overrides=dict(overrides),
    )
    if build:
        return _build_native_detector_module(detector_spec, in_channels=in_channels)
    return detector_spec


def _coerce_encoder_argument(
    encoder: str | EncoderSpec,
    *,
    pretrained: bool,
    in_channels: int,
) -> EncoderSpec:
    if isinstance(encoder, EncoderSpec):
        return encoder
    if isinstance(encoder, str):
        return build_encoder(
            encoder,
            source="timm",
            pretrained=pretrained,
            in_channels=in_channels,
        )
    raise ValueError("`encoder` must be a string model name, EncoderSpec, or None.")


def _coerce_backbone_argument(
    backbone: str | EncoderSpec,
    *,
    pretrained: bool,
    in_channels: int,
) -> EncoderSpec:
    if isinstance(backbone, EncoderSpec):
        return backbone
    if isinstance(backbone, str):
        return build_backbone(
            backbone,
            pretrained=pretrained,
            in_channels=in_channels,
        )
    raise ValueError("`backbone` must be a string alias, EncoderSpec, or None.")


def _validate_component_argument(
    parameter: str,
    value: Any,
    expected_type: type,
) -> None:
    if value is None or isinstance(value, expected_type):
        return
    raise ValueError(
        f"`{parameter}` must be a {expected_type.__name__} instance or None."
    )


def _build_native_detector_module(spec: DetectorSpec, *, in_channels: int) -> Any:
    from ..native.modeling import build_native_model

    return build_native_model(
        spec.architecture,
        num_classes=spec.num_classes,
        in_channels=int(in_channels),
        detector_spec=spec,
    )


def build_custom_detector(
    architecture: str,
    *,
    family: str,
    num_classes: int = 1,
    encoder: str | EncoderSpec | None = None,
    neck: NeckSpec | None = None,
    head: HeadSpec | None = None,
    decoder: DecoderSpec | None = None,
    in_channels: int = 3,
    pretrained: bool = True,
    strict_auto_adapt: bool = True,
    imports: tuple[str, ...] | list[str] | None = None,
    **overrides: Any,
) -> DetectorSpec:
    normalized_architecture = resolve_architecture_name(architecture)
    normalized_family = str(family).strip().lower()
    if not normalized_architecture:
        raise ValueError("Custom detector architectures require a non-empty name.")
    if normalized_family not in {"dense", "proposal", "roi", "transformer"}:
        raise ValueError("Custom detector families must be one of: dense, proposal, roi, transformer.")

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
        if normalized_family == "dense":
            default_head = _DENSE_DEFAULT_HEAD_BY_ARCHITECTURE.get(
                normalized_architecture,
                "RetinaHead",
            )
            with_mask = normalized_architecture == "mask_rcnn"
            head = build_head(
                default_head,
                num_classes=num_classes,
                with_mask=with_mask,
                **_default_dense_head_extra(normalized_architecture, neck),
            )
        elif normalized_family == "roi":
            head = build_head(
                _ROI_DEFAULT_BBOX_HEAD_BY_ARCHITECTURE.get(
                    normalized_architecture,
                    "Shared2FCBBoxHead",
                ),
                num_classes=num_classes,
                with_mask=normalized_architecture in {"mask_rcnn", "cascade_mask_rcnn"},
            )
        elif normalized_family == "proposal":
            head = build_head(
                "RPNHead",
                num_classes=1,
                num_anchors=1,
            )
        else:
            head = None
    else:
        head = replace(head, num_classes=num_classes if head.num_classes is None else head.num_classes)

    return DetectorSpec(
        architecture=normalized_architecture,
        family=normalized_family,
        num_classes=num_classes,
        encoder=encoder,
        neck=neck,
        head=head,
        decoder=decoder,
        strict_auto_adapt=strict_auto_adapt,
        imports=tuple(imports or ()),
        overrides=dict(overrides),
    )
