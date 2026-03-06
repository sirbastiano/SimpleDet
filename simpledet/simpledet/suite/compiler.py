"""Compilation of suite detector specs into MMDetection-style configs."""

from __future__ import annotations

import copy
from typing import Any

from .._model_resolution import apply_runtime_model_overrides
from ..src import models as model_library
from .specs import DecoderSpec, DetectorSpec, EncoderSpec, HeadSpec, NeckSpec


ARCHITECTURE_TEMPLATES: dict[str, str] = {
    "retinanet": "retinanet_r50_fpn",
    "fcos": "fcos_r18_fpn",
    "vfnet": "VFNet_r50_fpn",
    "fovea": "fovea_r50",
    "faster_rcnn": "faster_rcnn_r50_fpn",
    "mask_rcnn": "mask_rcnn_r50_fpn",
    "cascade_rcnn": "cascade_rcnn_r50_fpn",
    "cascade_mask_rcnn": "cascade_mask_rcnn_r50_fpn",
    "detr": "detr_r50",
    "deformable_detr": "deformable_detr_r50",
}


def compile_detector_spec(spec: DetectorSpec) -> dict[str, Any]:
    """Compile a :class:`DetectorSpec` into a model config dictionary."""
    if not isinstance(spec, DetectorSpec):
        raise TypeError("`spec` must be an instance of DetectorSpec.")

    template_name = ARCHITECTURE_TEMPLATES[spec.architecture]
    model_cfg = copy.deepcopy(getattr(model_library, template_name))

    if spec.neck is not None:
        _apply_neck_spec(model_cfg, spec.neck)
    if spec.head is not None:
        _apply_head_spec(model_cfg, spec.head, family=spec.family)
    if spec.decoder is not None:
        _apply_decoder_spec(model_cfg, spec.decoder, family=spec.family)

    if spec.encoder is not None:
        _apply_encoder_spec(
            model_cfg,
            encoder=spec.encoder,
            num_classes=spec.num_classes,
            strict=spec.strict_auto_adapt,
        )
    else:
        _set_num_classes(model_cfg, spec.family, spec.num_classes)

    if spec.overrides:
        _deep_merge(model_cfg, spec.overrides)

    return model_cfg


def _apply_encoder_spec(
    model_cfg: dict[str, Any],
    *,
    encoder: EncoderSpec,
    num_classes: int,
    strict: bool,
) -> None:
    if encoder.source == "timm":
        apply_runtime_model_overrides(
            model_cfg,
            encoder_name=encoder.name,
            encoder_pretrained=encoder.pretrained,
            encoder_in_chans=encoder.in_channels,
            num_classes=num_classes,
            strict=strict,
        )
        return

    if encoder.source == "config":
        backbone_cfg = copy.deepcopy(encoder.backbone_cfg or {})
        if encoder.extra:
            _deep_merge(backbone_cfg, encoder.extra)
        apply_runtime_model_overrides(
            model_cfg,
            backbone_cfg=backbone_cfg,
            feature_channels=list(encoder.feature_channels or ()),
            num_classes=num_classes,
            strict=strict,
        )
        return

    raise ValueError(f"Unsupported encoder source '{encoder.source}'.")


def _apply_neck_spec(model_cfg: dict[str, Any], neck: NeckSpec) -> None:
    neck_cfg = model_cfg.get("neck")
    if neck.neck_cfg is not None:
        model_cfg["neck"] = copy.deepcopy(neck.neck_cfg)
        neck_cfg = model_cfg["neck"]
    elif not isinstance(neck_cfg, dict):
        neck_cfg = {}
        model_cfg["neck"] = neck_cfg

    if not isinstance(neck_cfg, dict):
        return

    if neck.name not in {"", "auto"}:
        neck_cfg["type"] = neck.name
    if neck.out_channels is not None:
        neck_cfg["out_channels"] = int(neck.out_channels)
    if neck.num_outs is not None:
        neck_cfg["num_outs"] = int(neck.num_outs)
    if neck.extra:
        _deep_merge(neck_cfg, neck.extra)


def _apply_head_spec(model_cfg: dict[str, Any], head: HeadSpec, *, family: str) -> None:
    if family == "dense":
        target = model_cfg.setdefault("bbox_head", {})
        if head.head_cfg is not None:
            model_cfg["bbox_head"] = copy.deepcopy(head.head_cfg)
            target = model_cfg["bbox_head"]
        if head.name not in {"", "auto"}:
            target["type"] = head.name
        if head.num_classes is not None:
            target["num_classes"] = int(head.num_classes)
        if head.extra:
            _deep_merge(target, head.extra)
        return

    if family == "roi":
        roi_head = model_cfg.setdefault("roi_head", {})
        bbox_head = roi_head.get("bbox_head")
        if isinstance(bbox_head, list):
            for item in bbox_head:
                if head.name not in {"", "auto"}:
                    item["type"] = head.name
                if head.num_classes is not None:
                    item["num_classes"] = int(head.num_classes)
                if head.extra:
                    _deep_merge(item, head.extra)
        elif isinstance(bbox_head, dict):
            if head.head_cfg is not None:
                roi_head["bbox_head"] = copy.deepcopy(head.head_cfg)
                bbox_head = roi_head["bbox_head"]
            if head.name not in {"", "auto"}:
                bbox_head["type"] = head.name
            if head.num_classes is not None:
                bbox_head["num_classes"] = int(head.num_classes)
            if head.extra:
                _deep_merge(bbox_head, head.extra)

        if head.with_mask and isinstance(roi_head.get("mask_head"), dict):
            roi_head["mask_head"]["num_classes"] = int(head.num_classes or 1)
        return

    if family == "transformer":
        target = model_cfg.setdefault("bbox_head", {})
        if head.head_cfg is not None:
            model_cfg["bbox_head"] = copy.deepcopy(head.head_cfg)
            target = model_cfg["bbox_head"]
        if head.name not in {"", "auto"}:
            target["type"] = head.name
        if head.num_classes is not None:
            target["num_classes"] = int(head.num_classes)
        if head.extra:
            _deep_merge(target, head.extra)
        return


def _apply_decoder_spec(model_cfg: dict[str, Any], decoder: DecoderSpec, *, family: str) -> None:
    if family != "transformer":
        return
    if decoder.num_queries is not None:
        model_cfg["num_queries"] = int(decoder.num_queries)
    if decoder.decoder_cfg is not None:
        model_cfg["decoder"] = copy.deepcopy(decoder.decoder_cfg)
    if decoder.embed_dims is not None:
        _set_transformer_embed_dims(model_cfg, int(decoder.embed_dims))
    if decoder.extra:
        if isinstance(model_cfg.get("decoder"), dict):
            _deep_merge(model_cfg["decoder"], decoder.extra)
        else:
            model_cfg["decoder"] = dict(decoder.extra)


def _set_num_classes(model_cfg: dict[str, Any], family: str, num_classes: int) -> None:
    if family in {"dense", "transformer"}:
        bbox_head = model_cfg.get("bbox_head")
        if isinstance(bbox_head, dict):
            bbox_head["num_classes"] = int(num_classes)
        return

    if family == "roi":
        roi_head = model_cfg.get("roi_head", {})
        bbox_head = roi_head.get("bbox_head")
        if isinstance(bbox_head, list):
            for item in bbox_head:
                item["num_classes"] = int(num_classes)
        elif isinstance(bbox_head, dict):
            bbox_head["num_classes"] = int(num_classes)
        if isinstance(roi_head.get("mask_head"), dict):
            roi_head["mask_head"]["num_classes"] = int(num_classes)


def _set_transformer_embed_dims(model_cfg: dict[str, Any], embed_dims: int) -> None:
    for key in ("encoder", "decoder", "bbox_head"):
        section = model_cfg.get(key)
        if isinstance(section, dict):
            _patch_embed_dims(section, embed_dims)
    positional_encoding = model_cfg.get("positional_encoding")
    if isinstance(positional_encoding, dict) and "num_feats" in positional_encoding:
        positional_encoding["num_feats"] = max(embed_dims // 2, 1)


def _patch_embed_dims(node: Any, embed_dims: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "embed_dims" and isinstance(value, int):
                node[key] = embed_dims
            elif isinstance(value, dict):
                _patch_embed_dims(value, embed_dims)
            elif isinstance(value, list):
                for item in value:
                    _patch_embed_dims(item, embed_dims)


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)

