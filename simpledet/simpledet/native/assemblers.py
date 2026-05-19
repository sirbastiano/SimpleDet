"""Detector assembler layer for component-first native model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extensions import DETECTORS, HEADS
from ..suite import compile_native_detector_plan
from .backbones import build_native_backbone
from .dense_ops import (
    DenseATSSDecoder,
    DenseATSSLoss,
    DenseEfficientDetDecoder,
    DenseEfficientDetLoss,
    DenseFCOSDecoder,
    DenseFCOSLoss,
    DenseGFLDecoder,
    DenseGFLLoss,
    DenseRTMDetDecoder,
    DenseRTMDetLoss,
    DenseRetinaNetDecoder,
    DenseRetinaNetLoss,
    DenseSSDDecoder,
    DenseSSDLoss,
    DenseVFNetLoss,
    DenseYOLOXDecoder,
    DenseYOLOXLoss,
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

_DETECTOR_DEPENDENCIES = (("torch", "cpu"), ("torchvision", "cpu"))
_DENSE_CONTRACTS = ("feature_pyramid", "dense_predictions", "postprocessed_boxes")
_ROI_CONTRACTS = ("feature_pyramid", "roi_proposals", "postprocessed_boxes")
_TRANSFORMER_CONTRACTS = ("feature_sequence", "set_predictions", "postprocessed_boxes")


def build_native_components(detector_spec) -> NativeModelComponents:
    plan = compile_native_detector_plan(detector_spec)
    DETECTORS.import_modules(*plan.imports)
    backbone, backbone_spec = build_native_backbone(plan.encoder)
    neck, neck_spec = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
    head = None
    head_spec = None
    if plan.family == "dense" and plan.head is not None:
        _validate_dense_head_plan(plan)
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
    if components.plan.family != "dense":
        raise ValueError(
            f"Single-stage assembly requires a dense detector plan, got "
            f"{components.plan.family!r} for '{components.plan.architecture}'."
        )
    if components.plan.head is not None:
        _validate_dense_head_plan(components.plan)
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
        postprocessor=decoder(),
    )


def _validate_dense_head_plan(plan) -> None:
    metadata = HEADS.lookup(plan.head.type)
    head_family = None if metadata.family is None else str(metadata.family).strip().lower()
    if head_family is not None and head_family != "dense":
        raise ValueError(
            f"Single-stage detector '{plan.architecture}' requires a dense head, "
            f"but head '{metadata.name}' has family '{metadata.family}'."
        )


def _build_default_yolox_loss():
    return DenseYOLOXLoss(objectness_target_config={"positive": 1.0, "negative": 0.0})


@DETECTORS.register(
    "detr",
    aliases=("DETR",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_TRANSFORMER_CONTRACTS,
    validation_status="runtime_validated",
    family="transformer",
)
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


@DETECTORS.register(
    "retinanet",
    aliases=("RetinaNet",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="runtime_validated",
    family="dense",
)
@DETECTORS.register("retina")
def assemble_retinanet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseRetinaNetLoss,
        decoder=DenseRetinaNetDecoder,
    )


@DETECTORS.register(
    "fcos",
    aliases=("FCOS",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="runtime_validated",
    family="dense",
)
def assemble_fcos_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register(
    "atss",
    aliases=("ATSS",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="runtime_validated",
    family="dense",
)
def assemble_atss_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseATSSLoss,
        decoder=DenseATSSDecoder,
    )


@DETECTORS.register(
    "gfl",
    aliases=("GFL",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="runtime_validated",
    family="dense",
)
def assemble_gfl_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseGFLLoss,
        decoder=DenseGFLDecoder,
    )


@DETECTORS.register(
    "vfnet",
    aliases=("VFNet",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
    summary="VFNet detector family routed through native ATSS-compatible components.",
)
def assemble_vfnet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseVFNetLoss,
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
        loss_fn=_build_default_yolox_loss,
        decoder=DenseYOLOXDecoder,
    )


@DETECTORS.register("rtmdet")
@DETECTORS.register("rtmdet_tiny")
@DETECTORS.register("rtmdet_l")
@DETECTORS.register("rtmdet_m")
@DETECTORS.register("rtmdet_s")
def assemble_rtmdet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseRTMDetLoss,
        decoder=DenseRTMDetDecoder,
    )


@DETECTORS.register(
    "fovea",
    aliases=("FOVEA",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
@DETECTORS.register("foveabox", aliases=("FoveaBox",))
def assemble_fovea_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register(
    "reppoints",
    aliases=("RepPoints",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
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
        loss_fn=DenseSSDLoss,
        decoder=DenseSSDDecoder,
    )


@DETECTORS.register("efficientdet")
@DETECTORS.register("efficientdet_d0")
@DETECTORS.register("efficientdet_d1")
@DETECTORS.register("efficientdet_d2")
def assemble_efficientdet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseEfficientDetLoss,
        decoder=DenseEfficientDetDecoder,
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


@DETECTORS.register(
    "yolof",
    aliases=("YOLOF",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_yolof_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register(
    "centernet",
    aliases=("CenterNet",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_centernet_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFCOSLoss,
        decoder=DenseFCOSDecoder,
    )


@DETECTORS.register(
    "faster_rcnn",
    aliases=("Faster R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_ROI_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("faster-rcnn")
def assemble_faster_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector("faster_rcnn", components, num_classes=int(num_classes))


@DETECTORS.register(
    "mask_rcnn",
    aliases=("Mask R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_ROI_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("mask-rcnn")
def assemble_mask_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector("mask_rcnn", components, num_classes=int(num_classes))


@DETECTORS.register(
    "grid_rcnn",
    aliases=("Grid R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_ROI_CONTRACTS,
    validation_status="compatibility_alias",
    family="roi",
)
@DETECTORS.register("gridrcnn")
@DETECTORS.register(
    "cascade_rcnn",
    aliases=("Cascade R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_ROI_CONTRACTS,
    validation_status="compatibility_alias",
    family="roi",
)
@DETECTORS.register("cascadercnn")
def assemble_grid_or_cascade_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector(components.plan.architecture, components, num_classes=int(num_classes))
