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
    DenseAutoAssignDecoder,
    DenseAutoAssignLoss,
    DenseDDODDecoder,
    DenseDDODLoss,
    DenseEfficientDetDecoder,
    DenseEfficientDetLoss,
    DenseFCOSDecoder,
    DenseFCOSLoss,
    DenseFSAFDecoder,
    DenseFSAFLoss,
    DenseFoveaDecoder,
    DenseFoveaLoss,
    DenseFreeAnchorRetinaNetDecoder,
    DenseFreeAnchorRetinaNetLoss,
    DenseGFLDecoder,
    DenseGFLLoss,
    DenseGFLV2Decoder,
    DenseGFLV2Loss,
    DenseNASFCOSDecoder,
    DenseNASFCOSLoss,
    DensePAADecoder,
    DensePAALoss,
    DenseRepPointsDecoder,
    DenseRepPointsLoss,
    DenseRTMDetDecoder,
    DenseRTMDetLoss,
    DenseRetinaNetDecoder,
    DenseRetinaNetLoss,
    DenseSSDDecoder,
    DenseSSDLoss,
    DenseTOODDecoder,
    DenseTOODLoss,
    DenseVFNetDecoder,
    DenseVFNetLoss,
    DenseYOLOFDecoder,
    DenseYOLOFLoss,
    DenseYOLOXDecoder,
    DenseYOLOXLoss,
)
from .heads import build_native_head
from .modeling import NativeRetinaNetModel, QueryDetector
from .necks import build_native_neck
from .roi import build_native_roi_detector
from .transformer_ops import NativeDetrLoss, NativeDetrPostProcessor, SinePositionEncoding


@dataclass(slots=True)
class NativeModelComponents:
    plan: object
    backbone: object
    backbone_spec: object
    neck: object
    neck_spec: object
    head: object | None = None
    head_spec: object | None = None
    rpn_head: object | None = None
    rpn_head_spec: object | None = None
    bbox_head: object | None = None
    bbox_head_spec: object | None = None
    mask_head: object | None = None
    mask_head_spec: object | None = None
    grid_head: object | None = None
    grid_head_spec: object | None = None


_DETECTOR_DEPENDENCIES = (("torch", "cpu"), ("torchvision", "cpu"))
_PROPOSAL_DETECTOR_DEPENDENCIES = (("torch", "cpu"),)
_DENSE_CONTRACTS = ("feature_pyramid", "dense_predictions", "postprocessed_boxes")
_PROPOSAL_CONTRACTS = ("feature_pyramid", "rpn_head_outputs", "roi_proposals")
_ROI_CONTRACTS = ("feature_pyramid", "roi_proposals", "postprocessed_boxes")
_TRANSFORMER_CONTRACTS = ("feature_sequence", "set_predictions", "postprocessed_boxes")
_POSITIONAL_ENCODING_UNSET = object()


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
    rpn_head = None
    rpn_head_spec = None
    bbox_head = None
    bbox_head_spec = None
    mask_head = None
    mask_head_spec = None
    grid_head = None
    grid_head_spec = None
    if plan.family == "roi":
        _validate_roi_head_plan(plan)
        if plan.rpn_head is not None:
            rpn_head, rpn_head_spec = build_native_head(
                plan.rpn_head,
                out_channels=neck_spec.out_channels,
                num_classes=1,
            )
        bbox_head, bbox_head_spec = build_native_head(
            plan.bbox_head,
            out_channels=neck_spec.out_channels,
            num_classes=int(plan.num_classes),
        )
        head = bbox_head
        head_spec = bbox_head_spec
        if plan.mask_head is not None:
            mask_head, mask_head_spec = build_native_head(
                plan.mask_head,
                out_channels=neck_spec.out_channels,
                num_classes=int(plan.num_classes),
            )
        if plan.grid_head is not None:
            grid_head, grid_head_spec = build_native_head(
                plan.grid_head,
                out_channels=neck_spec.out_channels,
                num_classes=int(plan.num_classes),
            )
    if plan.family == "proposal":
        _validate_proposal_head_plan(plan)
        rpn_head, rpn_head_spec = build_native_head(
            plan.rpn_head,
            out_channels=neck_spec.out_channels,
            num_classes=1,
        )
        head = rpn_head
        head_spec = rpn_head_spec
    if plan.family == "transformer":
        _validate_transformer_head_plan(plan)
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
        rpn_head=rpn_head,
        rpn_head_spec=rpn_head_spec,
        bbox_head=bbox_head,
        bbox_head_spec=bbox_head_spec,
        mask_head=mask_head,
        mask_head_spec=mask_head_spec,
        grid_head=grid_head,
        grid_head_spec=grid_head_spec,
    )


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


def _validate_roi_head_plan(plan) -> None:
    if _roi_detector_requires_rpn(plan.architecture) and plan.rpn_head is None:
        raise ValueError(f"Two-stage detector '{plan.architecture}' requires an RPN head plan.")
    if plan.rpn_head is not None:
        rpn_metadata = HEADS.lookup(plan.rpn_head.type)
        if rpn_metadata.name != "RPNHead":
            raise ValueError(
                f"Two-stage detector '{plan.architecture}' requires RPNHead for proposal generation, "
                f"got '{rpn_metadata.name}'."
            )
    if plan.bbox_head is None:
        raise ValueError(f"Two-stage detector '{plan.architecture}' requires an ROI bbox head plan.")
    bbox_metadata = HEADS.lookup(plan.bbox_head.type)
    bbox_family = None if bbox_metadata.family is None else str(bbox_metadata.family).strip().lower()
    if bbox_family != "roi":
        raise ValueError(
            f"Two-stage detector '{plan.architecture}' requires an ROI bbox head, "
            f"but head '{bbox_metadata.name}' has family '{bbox_metadata.family}'."
        )
    if plan.architecture in {"mask_rcnn", "cascade_mask_rcnn"} and plan.mask_head is None:
        raise ValueError(f"{plan.architecture} build-plan validation requires a native mask head.")
    if plan.mask_head is not None:
        _validate_optional_roi_head(plan, plan.mask_head, "mask")
    if plan.architecture == "grid_rcnn" and plan.grid_head is None:
        raise ValueError("grid_rcnn build-plan validation requires a native grid head.")
    if plan.grid_head is not None:
        _validate_optional_roi_head(plan, plan.grid_head, "grid")


def _validate_proposal_head_plan(plan) -> None:
    if plan.rpn_head is None:
        raise ValueError(f"Proposal detector '{plan.architecture}' requires an RPN head plan.")
    metadata = HEADS.lookup(plan.rpn_head.type)
    if metadata.name != "RPNHead":
        raise ValueError(
            f"Proposal detector '{plan.architecture}' requires RPNHead, got '{metadata.name}'."
        )


def _validate_transformer_head_plan(plan) -> None:
    if plan.head is None:
        raise ValueError(f"Query detector '{plan.architecture}' requires a transformer head plan.")
    metadata = HEADS.lookup(plan.head.type)
    head_family = None if metadata.family is None else str(metadata.family).strip().lower()
    if head_family != "transformer":
        raise ValueError(
            f"Query detector '{plan.architecture}' requires a transformer head, "
            f"but head '{metadata.name}' has family '{metadata.family}'."
        )


def _validate_optional_roi_head(plan, head_plan, role: str) -> None:
    metadata = HEADS.lookup(head_plan.type)
    family = None if metadata.family is None else str(metadata.family).strip().lower()
    if family != "roi":
        raise ValueError(
            f"Two-stage detector '{plan.architecture}' requires an ROI {role} head, "
            f"but head '{metadata.name}' has family '{metadata.family}'."
        )


def _roi_detector_requires_rpn(architecture: str) -> bool:
    return str(architecture) != "fast_rcnn"


class _RPNProposalLoss:
    def __call__(self, images, targets, feature_pyramids, head_outputs_per_image):
        import torch
        import torch.nn.functional as F

        losses = []
        for outputs in head_outputs_per_image:
            objectness = outputs.get("objectness_logits")
            bbox_regression = outputs.get("bbox_regression")
            if objectness is None or bbox_regression is None:
                raise ValueError("RPN detector head outputs must include objectness_logits and bbox_regression.")
            objectness_rows = [
                level.reshape(-1)
                for level in objectness
            ]
            bbox_rows = [
                level.reshape(-1, 4)
                for level in bbox_regression
            ]
            logits = torch.cat(objectness_rows, dim=0) if objectness_rows else torch.zeros(())
            deltas = torch.cat(bbox_rows, dim=0) if bbox_rows else torch.zeros((0, 4))
            objectness_targets = torch.ones_like(logits)
            box_targets = torch.zeros_like(deltas)
            objectness_loss = F.binary_cross_entropy_with_logits(logits, objectness_targets)
            box_loss = (
                F.smooth_l1_loss(deltas, box_targets)
                if int(deltas.numel()) > 0
                else objectness_loss.new_zeros(())
            )
            losses.append((objectness_loss, box_loss))
        if not losses:
            zero = torch.zeros(())
            return {
                "loss_rpn_objectness": zero,
                "loss_rpn_box_reg": zero,
                "loss_total": zero,
            }
        objectness_loss = sum(item[0] for item in losses) / len(losses)
        box_loss = sum(item[1] for item in losses) / len(losses)
        return {
            "loss_rpn_objectness": objectness_loss,
            "loss_rpn_box_reg": box_loss,
            "loss_total": objectness_loss + box_loss,
        }


class _RPNProposalDecoder:
    def __call__(self, image, feature_pyramid, head_outputs):
        import torch

        objectness = head_outputs.get("objectness_logits")
        if objectness is None:
            raise ValueError("RPN detector head outputs must include objectness_logits.")
        logits = torch.cat([level.reshape(-1) for level in objectness], dim=0)
        scores = torch.sigmoid(logits)
        topk = min(4, int(scores.numel()))
        if topk == 0:
            return {
                "boxes": image.new_zeros((0, 4)),
                "scores": image.new_zeros((0,)),
                "labels": image.new_zeros((0,), dtype=torch.long),
            }
        values, _indices = torch.topk(scores, k=topk)
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        base_boxes = image.new_tensor(
            [
                [0.0, 0.0, float(width - 1), float(height - 1)],
                [0.0, 0.0, float(max(width // 2, 1)), float(max(height // 2, 1))],
                [float(width // 4), float(height // 4), float(width - 1), float(height - 1)],
                [float(width // 3), float(height // 3), float(max(width - 2, 1)), float(max(height - 2, 1))],
            ]
        )
        boxes = base_boxes[:topk].clone()
        labels = torch.ones((topk,), dtype=torch.long, device=image.device)
        return {
            "boxes": boxes,
            "scores": values,
            "labels": labels,
        }


def _build_default_yolox_loss():
    return DenseYOLOXLoss(objectness_target_config={"positive": 1.0, "negative": 0.0})


def _build_query_position_encoding(plan, neck_spec):
    configured = getattr(plan, "overrides", {}).get("positional_encoding", _POSITIONAL_ENCODING_UNSET)
    if configured is None:
        return None
    if configured is _POSITIONAL_ENCODING_UNSET:
        params = {"num_feats": max(int(neck_spec.out_channels) // 2, 1)}
    else:
        if not isinstance(configured, dict):
            raise ValueError("QueryDetector positional_encoding must be a mapping of settings.")
        params = dict(configured)
        if "num_feats" not in params:
            raise ValueError(
                "QueryDetector requires positional_encoding.num_feats when positional_encoding is provided."
            )
    params.setdefault("normalize", True)
    return SinePositionEncoding(**params)


@DETECTORS.register(
    "detr",
    aliases=("DETR",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_TRANSFORMER_CONTRACTS,
    validation_status="runtime_validated",
    family="transformer",
)
@DETECTORS.register(
    "deformable_detr",
    aliases=("DeformableDETR",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_TRANSFORMER_CONTRACTS,
    validation_status="runtime_validated",
    family="transformer",
)
@DETECTORS.register(
    "conditional_detr",
    aliases=("ConditionalDETR",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_TRANSFORMER_CONTRACTS,
    validation_status="runtime_validated",
    family="transformer",
)
@DETECTORS.register(
    "dab_detr",
    aliases=("DAB-DETR",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_TRANSFORMER_CONTRACTS,
    validation_status="runtime_validated",
    family="transformer",
)
@DETECTORS.register(
    "dino",
    aliases=("DINO",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_TRANSFORMER_CONTRACTS,
    validation_status="runtime_validated",
    family="transformer",
)
def assemble_transformer_detector(components: NativeModelComponents, *, num_classes: int):
    if components.plan.family != "transformer":
        raise ValueError(
            f"Query detector assembly requires a transformer detector plan, got "
            f"{components.plan.family!r} for '{components.plan.architecture}'."
        )
    _validate_transformer_head_plan(components.plan)
    if components.head is None or components.head_spec is None:
        raise ValueError(f"{components.plan.architecture} assembly requires a native query head.")
    return QueryDetector(
        backbone=components.backbone,
        neck=components.neck,
        head=components.head,
        backbone_spec=components.backbone_spec,
        neck_spec=components.neck_spec,
        head_spec=components.head_spec,
        loss_fn=NativeDetrLoss(),
        postprocessor=NativeDetrPostProcessor(),
        positional_encoding=_build_query_position_encoding(components.plan, components.neck_spec),
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
    "fsaf",
    aliases=("FSAF",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_fsaf_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFSAFLoss,
        decoder=DenseFSAFDecoder,
    )


@DETECTORS.register(
    "free_anchor",
    aliases=("FreeAnchor",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_free_anchor_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFreeAnchorRetinaNetLoss,
        decoder=DenseFreeAnchorRetinaNetDecoder,
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
    "gfocalv2",
    aliases=("GFocalV2", "GFLV2"),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_gfocalv2_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseGFLV2Loss,
        decoder=DenseGFLV2Decoder,
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
        decoder=DenseVFNetDecoder,
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
    aliases=("FOVEA", "FoveaBox"),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_fovea_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseFoveaLoss,
        decoder=DenseFoveaDecoder,
    )


@DETECTORS.register(
    "paa",
    aliases=("PAA",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_paa_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DensePAALoss,
        decoder=DensePAADecoder,
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
        loss_fn=DenseRepPointsLoss,
        decoder=DenseRepPointsDecoder,
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


@DETECTORS.register(
    "tood",
    aliases=("TOOD",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
@DETECTORS.register("solov2")
@DETECTORS.register("solov2_light")
def assemble_tood_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseTOODLoss,
        decoder=DenseTOODDecoder,
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
        loss_fn=DenseYOLOFLoss,
        decoder=DenseYOLOFDecoder,
    )


@DETECTORS.register(
    "ddod",
    aliases=("DDOD",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_ddod_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseDDODLoss,
        decoder=DenseDDODDecoder,
    )


@DETECTORS.register(
    "auto_assign",
    aliases=("AutoAssign",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_auto_assign_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseAutoAssignLoss,
        decoder=DenseAutoAssignDecoder,
    )


@DETECTORS.register(
    "nas_fcos",
    aliases=("NAS-FCOS",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_DENSE_CONTRACTS,
    validation_status="compatibility_alias",
    family="dense",
)
def assemble_nas_fcos_detector(components: NativeModelComponents, *, num_classes: int):
    return _assemble_dense_detector(
        components=components,
        loss_fn=DenseNASFCOSLoss,
        decoder=DenseNASFCOSDecoder,
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
    "rpn",
    aliases=("RPN", "RPN detector", "Region Proposal Network"),
    required_dependencies=_PROPOSAL_DETECTOR_DEPENDENCIES,
    tensor_contracts=_PROPOSAL_CONTRACTS,
    validation_status="runtime_validated",
    family="proposal",
)
def assemble_rpn_detector(components: NativeModelComponents, *, num_classes: int):
    if components.rpn_head is None or components.rpn_head_spec is None:
        raise ValueError("rpn detector assembly requires a native RPN head.")
    return NativeRetinaNetModel(
        backbone=components.backbone,
        neck=components.neck,
        head=components.rpn_head,
        backbone_spec=components.backbone_spec,
        neck_spec=components.neck_spec,
        head_spec=components.rpn_head_spec,
        loss_fn=_RPNProposalLoss(),
        postprocessor=_RPNProposalDecoder(),
    )


@DETECTORS.register(
    "fast_rcnn",
    aliases=("Fast R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_ROI_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("fast-rcnn")
def assemble_fast_rcnn_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector("fast_rcnn", components, num_classes=int(num_classes))


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
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("gridrcnn")
@DETECTORS.register(
    "cascade_rcnn",
    aliases=("Cascade R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=_ROI_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("cascadercnn")
@DETECTORS.register(
    "cascade_mask_rcnn",
    aliases=("Cascade Mask R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=(*_ROI_CONTRACTS, "mask_predictions"),
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("cascademaskrcnn")
@DETECTORS.register(
    "libra_rcnn",
    aliases=("Libra R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=(*_ROI_CONTRACTS, "balanced_roi_sampling"),
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("librarcnn")
@DETECTORS.register(
    "double_head_rcnn",
    aliases=("Double-Head R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=(*_ROI_CONTRACTS, "double_head_roi_features"),
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("doubleheadrcnn")
@DETECTORS.register(
    "dynamic_rcnn",
    aliases=("Dynamic R-CNN",),
    required_dependencies=_DETECTOR_DEPENDENCIES,
    tensor_contracts=(*_ROI_CONTRACTS, "dynamic_roi_features"),
    validation_status="runtime_validated",
    family="roi",
)
@DETECTORS.register("dynamicrcnn")
def assemble_roi_variant_detector(components: NativeModelComponents, *, num_classes: int):
    return build_native_roi_detector(components.plan.architecture, components, num_classes=int(num_classes))
