"""Backend-neutral detector build plans for the native runtime migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .specs import DecoderSpec, DetectorSpec, EncoderSpec, HeadSpec, NeckSpec


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else dict(value)


def _copy_imports(value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(slots=True, frozen=True)
class ComponentPlan:
    """Native backend component plan independent of MMDet config shape."""

    kind: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    imports: tuple[str, ...] = ()
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DetectorBuildPlan:
    """Structured native build plan compiled from a detector spec."""

    architecture: str
    family: str
    num_classes: int
    imports: tuple[str, ...] = ()
    encoder: ComponentPlan | None = None
    neck: ComponentPlan | None = None
    head: ComponentPlan | None = None
    rpn_head: ComponentPlan | None = None
    bbox_head: ComponentPlan | None = None
    mask_head: ComponentPlan | None = None
    grid_head: ComponentPlan | None = None
    decoder: ComponentPlan | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def compile_native_detector_plan(spec: DetectorSpec) -> DetectorBuildPlan:
    """Compile a :class:`DetectorSpec` into a backend-neutral native build plan."""
    if not isinstance(spec, DetectorSpec):
        raise ValueError("`spec` must be an instance of DetectorSpec.")

    head = _compile_head_plan(spec.head)
    decoder = _compile_decoder_plan(spec.decoder)
    rpn_head = None
    bbox_head = None
    mask_head = None
    grid_head = None
    if spec.family == "roi":
        _validate_roi_detector_spec(spec, head)
        if _roi_detector_requires_rpn(spec.architecture):
            rpn_head = _default_head_plan("RPNHead", num_classes=1, num_anchors=1)
        bbox_head = head or _default_head_plan(_default_roi_bbox_head_type(spec.architecture), num_classes=spec.num_classes)
        head = bbox_head
        if spec.architecture in {"mask_rcnn", "cascade_mask_rcnn"}:
            mask_head = _default_head_plan(_default_roi_mask_head_type(spec.architecture), num_classes=spec.num_classes)
        if spec.architecture == "grid_rcnn":
            grid_head = _default_head_plan("GridHead", num_classes=spec.num_classes, grid_size=7)
    elif spec.family == "proposal":
        rpn_head = head or _default_head_plan("RPNHead", num_classes=1, num_anchors=1)
        head = rpn_head
    elif spec.family == "transformer":
        head = _compile_transformer_head_plan(spec, head, decoder)

    return DetectorBuildPlan(
        architecture=spec.architecture,
        family=spec.family,
        num_classes=spec.num_classes,
        imports=_collect_imports(spec),
        encoder=_compile_encoder_plan(spec.encoder),
        neck=_compile_neck_plan(spec.neck),
        head=head,
        rpn_head=rpn_head,
        bbox_head=bbox_head,
        mask_head=mask_head,
        grid_head=grid_head,
        decoder=decoder,
        overrides=_copy_mapping(spec.overrides),
    )


def _default_head_plan(head_type: str, **params: Any) -> ComponentPlan:
    return ComponentPlan(kind="head", type=head_type, params=dict(params))


def _default_roi_bbox_head_type(architecture: str) -> str:
    if architecture in {"cascade_rcnn", "cascade_mask_rcnn"}:
        return "CascadeBBoxHead"
    if architecture == "double_head_rcnn":
        return "DoubleConvFCBBoxHead"
    if architecture == "dynamic_rcnn":
        return "DynamicBBoxHead"
    if architecture == "sparse_rcnn":
        return "SparseRoIHead"
    return "Shared2FCBBoxHead"


def _default_roi_mask_head_type(architecture: str) -> str:
    if architecture == "cascade_mask_rcnn":
        return "CascadeMaskHead"
    return "FCNMaskHead"


def _roi_detector_requires_rpn(architecture: str) -> bool:
    return architecture not in {"fast_rcnn", "sparse_rcnn"}


_TRANSFORMER_DEFAULT_HEAD_BY_ARCHITECTURE = {
    "detr": "DETRHead",
    "conditional_detr": "ConditionalDETRHead",
    "dab_detr": "DABDETRHead",
    "deformable_detr": "DeformableDETRHead",
    "dino": "DINOHead",
}

_TRANSFORMER_QUERY_DEFAULTS = {
    "detr": 100,
    "conditional_detr": 300,
    "dab_detr": 300,
    "deformable_detr": 300,
    "dino": 300,
}

_TRANSFORMER_HEAD_PARAM_KEYS = {
    "num_queries",
    "hidden_dim",
    "num_heads",
    "num_encoder_layers",
    "num_decoder_layers",
    "dim_feedforward",
    "dropout",
    "activation",
    "num_feature_levels",
}


def _compile_transformer_head_plan(
    spec: DetectorSpec,
    head: ComponentPlan | None,
    decoder: ComponentPlan | None,
) -> ComponentPlan:
    if head is not None:
        return head

    head_type = _TRANSFORMER_DEFAULT_HEAD_BY_ARCHITECTURE.get(spec.architecture)
    if head_type is None:
        raise ValueError(
            f"Transformer detector '{spec.architecture}' requires an explicit query head."
        )

    params: dict[str, Any] = {}
    if decoder is not None:
        params.update(decoder.params)
        if "embed_dims" in params:
            params.setdefault("hidden_dim", params.pop("embed_dims"))
    for key in _TRANSFORMER_HEAD_PARAM_KEYS:
        if key in spec.overrides:
            params[key] = spec.overrides[key]
    params.setdefault("num_classes", spec.num_classes)
    params.setdefault("num_queries", _TRANSFORMER_QUERY_DEFAULTS.get(spec.architecture, 100))
    return ComponentPlan(kind="head", type=head_type, params=params)


def _validate_roi_detector_spec(spec: DetectorSpec, head: ComponentPlan | None) -> None:
    if spec.architecture in {"mask_rcnn", "cascade_mask_rcnn"}:
        if head is None or not bool(head.params.get("with_mask", False)):
            raise ValueError(
                f"{spec.architecture} build-plan validation requires a mask-capable ROI head; use "
                f"build_detector('{spec.architecture}', ...) defaults or pass a head with with_mask=True."
            )


def _compile_encoder_plan(encoder: EncoderSpec | None) -> ComponentPlan | None:
    if encoder is None:
        return None
    params = _copy_mapping(encoder.extra)
    if encoder.source == "timm":
        params.update(
            {
                "model_name": encoder.name,
                "pretrained": encoder.pretrained,
                "in_channels": encoder.in_channels,
            }
        )
        return ComponentPlan(
            kind="encoder",
            type="timm",
            source="timm",
            params=params,
            imports=_copy_imports(encoder.imports),
        )

    params.update(_copy_mapping(encoder.backbone_cfg))
    if encoder.feature_channels is not None:
        params["feature_channels"] = list(encoder.feature_channels)
    return ComponentPlan(
        kind="encoder",
        type=str((encoder.backbone_cfg or {}).get("type") or encoder.name or "custom"),
        source=encoder.source,
        params=params,
        imports=_copy_imports(encoder.imports),
    )


def _compile_neck_plan(neck: NeckSpec | None) -> ComponentPlan | None:
    if neck is None:
        return None
    params = _copy_mapping(neck.neck_cfg)
    params.update(_copy_mapping(neck.extra))
    if neck.out_channels is not None:
        params["out_channels"] = neck.out_channels
    if neck.num_outs is not None:
        params["num_outs"] = neck.num_outs
    return ComponentPlan(
        kind="neck",
        type=str(params.pop("type", neck.name or "auto")),
        params=params,
        imports=_copy_imports(neck.imports),
    )


def _compile_head_plan(head: HeadSpec | None) -> ComponentPlan | None:
    if head is None:
        return None
    params = _copy_mapping(head.head_cfg)
    params.update(_copy_mapping(head.extra))
    if head.num_classes is not None:
        params["num_classes"] = head.num_classes
    if head.with_mask:
        params["with_mask"] = True
    return ComponentPlan(
        kind="head",
        type=str(params.pop("type", head.name or "auto")),
        params=params,
        imports=_copy_imports(head.imports),
    )


def _compile_decoder_plan(decoder: DecoderSpec | None) -> ComponentPlan | None:
    if decoder is None:
        return None
    params = _copy_mapping(decoder.decoder_cfg)
    params.update(_copy_mapping(decoder.extra))
    if decoder.num_queries is not None:
        params["num_queries"] = decoder.num_queries
    if decoder.embed_dims is not None:
        params["embed_dims"] = decoder.embed_dims
    return ComponentPlan(
        kind="decoder",
        type=str(params.pop("type", decoder.name or "auto")),
        params=params,
        imports=_copy_imports(decoder.imports),
    )


def _collect_imports(spec: DetectorSpec) -> tuple[str, ...]:
    imports: list[str] = []
    for group in (
        spec.imports,
        spec.encoder.imports if spec.encoder is not None else (),
        spec.neck.imports if spec.neck is not None else (),
        spec.head.imports if spec.head is not None else (),
        spec.decoder.imports if spec.decoder is not None else (),
    ):
        for item in group:
            text = str(item).strip()
            if text and text not in imports:
                imports.append(text)
    return tuple(imports)
