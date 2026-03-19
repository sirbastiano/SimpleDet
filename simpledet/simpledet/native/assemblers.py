"""Detector assembler layer for component-first native model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extensions import DETECTORS
from ..suite import compile_native_detector_plan
from .backbones import build_native_backbone
from .dense_ops import (
    DenseATSSDecoder,
    DenseATSSLoss,
    DenseFCOSDecoder,
    DenseFCOSLoss,
    DenseGFLDecoder,
    DenseGFLLoss,
    DenseRetinaNetDecoder,
    DenseRetinaNetLoss,
)
from .heads import build_native_head
from .modeling import NativeDetrModel, NativeRetinaNetModel
from .necks import build_native_neck
from .roi import build_native_roi_detector
from .transformer_ops import NativeDetrDecoder, NativeDetrLoss, NativeDetrPostProcessor


@dataclass(slots=True)
class NativeModelComponents:
    plan: object
    backbone: object
    backbone_spec: object
    neck: object
    neck_spec: object
    head: object | None = None
    head_spec: object | None = None


_TRANSFORMER_QUERY_DEFAULTS = {
    "detr": 100,
    "deformable_detr": 300,
    "conditional_detr": 300,
    "dino": 300,
}


def build_native_components(detector_spec) -> NativeModelComponents:
    plan = compile_native_detector_plan(detector_spec)
    DETECTORS.import_modules(*plan.imports)
    backbone, backbone_spec = build_native_backbone(plan.encoder)
    neck, neck_spec = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
    head = None
    head_spec = None
    if plan.family == "dense" and plan.head is not None:
        head, head_spec = build_native_head(
            plan.head,
            out_channels=neck_spec.out_channels,
            num_classes=int(plan.num_classes),
        )
    return NativeModelComponents(
        plan=plan,
        backbone=backbone,
        backbone_spec=backbone_spec,
        neck=neck,
        neck_spec=neck_spec,
        head=head,
        head_spec=head_spec,
    )


def _build_transformer_components(
    plan,
    neck_spec,
    *,
    num_classes: int,
):
    params: dict[str, Any] = {}
    if plan.decoder is not None:
        params.update(plan.decoder.params)
        if "embed_dims" in params:
            params.setdefault("in_channels", params.pop("embed_dims"))
    params.setdefault("in_channels", int(neck_spec.out_channels))
    params.setdefault("num_classes", int(num_classes))
    params.setdefault(
        "num_queries",
        _TRANSFORMER_QUERY_DEFAULTS.get(plan.architecture, 100),
    )
    return NativeDetrDecoder(**params), NativeDetrLoss(), NativeDetrPostProcessor()


def _assemble_dense_detector(
    *,
    components: NativeModelComponents,
    loss_fn,
    decoder: object,
):
    if components.head is None or components.head_spec is None:
        raise ValueError(f"{components.plan.architecture} assembly requires a native head.")
    return NativeRetinaNetModel(
        backbone=components.backbone,
        neck=components.neck,
        head=components.head,
        backbone_spec=components.backbone_spec,
        neck_spec=components.neck_spec,
        head_spec=components.head_spec,
        loss_fn=loss_fn(),
        decoder=decoder(),
    )


@DETECTORS.register("detr")
@DETECTORS.register("deformable_detr")
@DETECTORS.register("conditional_detr")
@DETECTORS.register("dino")
def assemble_transformer_detector(components: NativeModelComponents, *, num_classes: int):
    decoder, loss_fn, postprocessor = _build_transformer_components(
        components.plan,
        components.neck_spec,
        num_classes=num_classes,
    )
    return NativeDetrModel(
        backbone=components.backbone,
        neck=components.neck,
        decoder=decoder,
        loss_fn=loss_fn,
        postprocessor=postprocessor,
    )


@DETECTORS.register("retinanet")
@DETECTORS.register("retina")
def assemble_retinanet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseRetinaNetLoss,
        decoder=DenseRetinaNetDecoder,
    )


@DETECTORS.register("fcos")
def assemble_fcos_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("atss")
def assemble_atss_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseATSSLoss,
        decoder=DenseATSSDecoder,
    )


@DETECTORS.register("gfl")
def assemble_gfl_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseGFLLoss,
        decoder=DenseGFLDecoder,
    )


@DETECTORS.register("vfnet")
def assemble_vfnet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseATSSLoss,
        decoder=DenseATSSDecoder,
    )


@DETECTORS.register("yolox")
@DETECTORS.register("yolov3")
@DETECTORS.register("yolov5")
@DETECTORS.register("yolov6")
@DETECTORS.register("yolov7")
@DETECTORS.register("yolov8")
@DETECTORS.register("yolov4")
@DETECTORS.register("yolov9")
@DETECTORS.register("yolov10")
@DETECTORS.register("yolo4")
@DETECTORS.register("yolo")
@DETECTORS.register("yolo3")
@DETECTORS.register("yolo_v3")
def assemble_yolo_like_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("rtmdet")
@DETECTORS.register("rtmdet_tiny")
@DETECTORS.register("rtmdet_l")
@DETECTORS.register("rtmdet_m")
@DETECTORS.register("rtmdet_s")
def assemble_rtmdet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("fovea")
@DETECTORS.register("foveabox")
def assemble_fovea_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("reppoints")
def assemble_reppoints_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseATSSLoss,
        decoder=DenseATSSDecoder,
    )


@DETECTORS.register("ssd")
@DETECTORS.register("sabl")
@DETECTORS.register("ssd300")
@DETECTORS.register("ssd512")
@DETECTORS.register("ssdlite")
def assemble_ssd_like_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseATSSLoss,
        decoder=DenseATSSDecoder,
    )


@DETECTORS.register("tood")
@DETECTORS.register("solov2")
@DETECTORS.register("solov2_light")
def assemble_tood_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("yolof")
def assemble_yolof_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("centernet")
def assemble_centernet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register("faster_rcnn")
@DETECTORS.register("faster-rcnn")
def assemble_faster_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector("faster_rcnn", components, num_classes=int(num_classes))


@DETECTORS.register("mask_rcnn")
@DETECTORS.register("mask-rcnn")
def assemble_mask_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector("mask_rcnn", components, num_classes=int(num_classes))


@DETECTORS.register("grid_rcnn")
@DETECTORS.register("gridrcnn")
@DETECTORS.register("cascade_rcnn")
@DETECTORS.register("cascadercnn")
def assemble_grid_or_cascade_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector(components.plan.architecture, components, num_classes=int(num_classes))
