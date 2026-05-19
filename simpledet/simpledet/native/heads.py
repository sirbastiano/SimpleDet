"""Native dense heads for the Lightning backend."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import HEADS, LOSSES

require_dependency("torch", "native heads")
import torch.nn as nn  # noqa: E402


@dataclass(slots=True, frozen=True)
class HeadSpec:
    name: str
    num_classes: int
    in_channels: int
    num_anchors: int


def _positive_int(value: Any, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} is required.")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.")
    try:
        values = tuple(_positive_int(item, name) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.") from exc
    if not values:
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.")
    return values


@HEADS.register(
    "RetinaHead",
    aliases=("retina", "retina_head"),
    required_dependencies=(("torch", "cpu"), ("torchvision", "cpu")),
    tensor_contracts=("feature_pyramid", "retinanet_head_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="RetinaNet-compatible dense head.",
)
class RetinaDenseHead(nn.Module):
    """Thin wrapper around torchvision RetinaNetHead."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_anchors: int = 9,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_anchors = _positive_int(num_anchors, "num_anchors")
        self.num_classes = _positive_int(num_classes, "num_classes")
        require_dependency("torchvision", "native heads")
        from torchvision.models.detection.retinanet import RetinaNetHead

        self.head = RetinaNetHead(self.in_channels, self.num_anchors, self.num_classes)

    def __call__(self, features: list[Any] | tuple[Any, ...]):
        return self.forward(features)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        return self.head(list(features))


@HEADS.register("RetinaNetHead")
class RetinaNetHead(RetinaDenseHead):
    """Compatibility alias used by external detector configs."""


@HEADS.register(
    "FreeAnchorRetinaHead",
    aliases=("free_anchor", "free_anchor_head", "free_anchor_retina_head"),
    required_dependencies=(("torch", "cpu"), ("torchvision", "cpu")),
    tensor_contracts=("feature_pyramid", "free_anchor_head_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="FreeAnchor-compatible RetinaNet dense head.",
)
class FreeAnchorRetinaHead(RetinaDenseHead):
    """Retina-style dense head used by FreeAnchor detector families."""


@HEADS.register(
    "FCOSHead",
    aliases=("fcos", "fcos_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="FCOS-style dense head.",
)
class FCOSDenseHead(nn.Module):
    """Minimal native FCOS-like head placeholder with shared conv towers."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
    ) -> None:
        super().__init__()

        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_convs = _positive_int(num_convs, "num_convs")
        cls_tower = []
        box_tower = []
        current_channels = self.in_channels
        for _ in range(self.num_convs):
            cls_tower.append(nn.Conv2d(current_channels, self.in_channels, kernel_size=3, padding=1))
            cls_tower.append(nn.ReLU(inplace=True))
            box_tower.append(nn.Conv2d(current_channels, self.in_channels, kernel_size=3, padding=1))
            box_tower.append(nn.ReLU(inplace=True))
            current_channels = self.in_channels
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        self.cls_logits = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(self.in_channels, 1, kernel_size=3, padding=1)

    def __call__(self, features: list[Any] | tuple[Any, ...]):
        return self.forward(features)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        logits = []
        bbox_reg = []
        for feature in features:
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            logits.append(self.cls_logits(cls_feature))
            bbox_reg.append(self.bbox_pred(box_feature))
        centers = [self.centerness(self.box_tower(feature)) for feature in features]
        return {"cls_logits": logits, "bbox_regression": bbox_reg, "centerness": centers}


@HEADS.register("FCOSV2Head")
class FCOSV2Head(FCOSDenseHead):
    """Alias preserved for FCOS derivatives with matching FCOS neck behavior."""


def _default_num_anchors(params: dict[str, Any]) -> dict[str, Any]:
    if "num_anchors" not in params:
        params["num_anchors"] = 9
    return params


def _head_accepts_num_anchors(factory: Any) -> bool:
    try:
        signature = inspect.signature(factory.__init__)
    except (TypeError, ValueError):
        return False
    return "num_anchors" in signature.parameters


@HEADS.register("FCOSHeadV2")
class FCOSHeadV2(FCOSDenseHead):
    """Alternate alias for FCOS head naming."""


@HEADS.register(
    "FSAFHead",
    aliases=("fsaf", "fsaf_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="FSAF-style anchor-free dense head.",
)
class FSAFHead(FCOSDenseHead):
    """Anchor-free dense head used by FSAF detector families."""


@HEADS.register(
    "ATSSHead",
    aliases=("atss", "atss_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="ATSS-style anchor-based dense head.",
)
class ATSSDenseHead(nn.Module):
    """Anchor-based dense head with centerness outputs for ATSS-style models."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_anchors = _positive_int(num_anchors, "num_anchors")
        self.num_convs = _positive_int(num_convs, "num_convs")
        cls_tower = []
        box_tower = []
        current_channels = self.in_channels
        for _ in range(self.num_convs):
            cls_tower.append(nn.Conv2d(current_channels, self.in_channels, kernel_size=3, padding=1))
            cls_tower.append(nn.ReLU(inplace=True))
            box_tower.append(nn.Conv2d(current_channels, self.in_channels, kernel_size=3, padding=1))
            box_tower.append(nn.ReLU(inplace=True))
            current_channels = self.in_channels
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        self.cls_logits = nn.Conv2d(
            self.in_channels,
            self.num_anchors * self.num_classes,
            kernel_size=3,
            padding=1,
        )
        self.bbox_pred = nn.Conv2d(self.in_channels, self.num_anchors * 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(self.in_channels, self.num_anchors, kernel_size=3, padding=1)

    def __call__(self, features: list[Any] | tuple[Any, ...]):
        return self.forward(features)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        logits = []
        bbox_reg = []
        centers = []
        for feature in features:
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            logits.append(self.cls_logits(cls_feature))
            bbox_reg.append(self.bbox_pred(box_feature))
            centers.append(self.centerness(box_feature))
        return {"cls_logits": logits, "bbox_regression": bbox_reg, "centerness": centers}


@HEADS.register("ATSSV2Head")
class ATSSV2Head(ATSSDenseHead):
    """Compatibility alias for ATSS-style variants."""


@HEADS.register(
    "RPNHead",
    aliases=("rpn", "rpn_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "rpn_head_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="Region proposal network dense head.",
)
class RPNHead(nn.Module):
    """Class-agnostic dense proposal head with objectness and box deltas."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_anchors = _positive_int(num_anchors, "num_anchors")
        self.num_convs = _positive_int(num_convs, "num_convs")
        tower = []
        for _ in range(self.num_convs):
            tower.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1))
            tower.append(nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*tower)
        self.objectness_logits = nn.Conv2d(self.in_channels, self.num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, self.num_anchors * 4, kernel_size=1)

    def __call__(self, features: list[Any] | tuple[Any, ...]):
        return self.forward(features)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        objectness = []
        bbox_reg = []
        for feature in features:
            proposal_feature = self.tower(feature)
            objectness.append(self.objectness_logits(proposal_feature))
            bbox_reg.append(self.bbox_pred(proposal_feature))
        return {"objectness_logits": objectness, "bbox_regression": bbox_reg}


@HEADS.register(
    "VFNetHead",
    aliases=("VFNet", "vfnet_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="VFNet-family dense head using the native ATSS output contract with varifocal loss support.",
)
class VFNetHead(ATSSDenseHead):
    """Dense alias for the VFNet head family."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 4,
        loss_cls: str = "varifocal",
        loss_bbox: str = "iou",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_convs=num_convs,
        )
        LOSSES.import_modules("simpledet.native.losses")
        self.loss_cls = LOSSES.get(loss_cls)()
        self.loss_bbox = LOSSES.get(loss_bbox)()


@HEADS.register(
    "RepPointsHead",
    aliases=("RepPoints", "reppoints_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs", "point_generator"),
    validation_status="runtime_validated",
    family="dense",
    summary="RepPoints-family point-based dense head using the native FCOS output contract.",
)
class RepPointsHead(FCOSDenseHead):
    """RepPoints-style point-based dense head with explicit point generator settings."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        point_strides: tuple[int, ...] | list[int] | None = None,
        point_base_scale: int = 4,
        num_convs: int = 4,
    ) -> None:
        if point_strides is None:
            raise ValueError(
                "RepPointsHead requires point-generation settings: provide point_strides."
            )
        self.point_strides = _positive_int_tuple(point_strides, "point_strides")
        self.point_base_scale = _positive_int(point_base_scale, "point_base_scale")
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            num_convs=num_convs,
        )

    def forward(self, features: list[Any] | tuple[Any, ...]):
        if len(features) != len(self.point_strides):
            raise ValueError(
                "RepPointsHead point-generation settings mismatch: point_strides "
                f"defines {len(self.point_strides)} levels but received {len(features)} feature levels."
            )
        return super().forward(features)


class ReppointsHead(RepPointsHead):
    """Alternate alias for Reppoints naming."""


@HEADS.register(
    "FoveaHead",
    aliases=("FOVEA", "fovea_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="Fovea-family anchor-free dense head.",
)
class FoveaHead(FCOSDenseHead):
    """Anchor-free dense head used by Fovea-style detectors."""


@HEADS.register(
    "YOLOFHead",
    aliases=("YOLOF", "yolof_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="YOLOF-family dense head using the native FCOS output contract.",
)
class YOLOFHead(FCOSDenseHead):
    """Anchor-free YOLOF-style head alias."""


@HEADS.register(
    "YOLOXHead",
    aliases=("yolox", "yolox_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "yolox_anchor_free_outputs", "sim_ota_targets"),
    validation_status="runtime_validated",
    family="dense",
    summary="YOLOX-family anchor-free dense head with explicit objectness, class, and bbox branches.",
)
class YOLOXHead(FCOSDenseHead):
    """YOLOX-style anchor-free head with a dedicated objectness branch."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, num_convs=num_convs)
        self.objectness_logits = nn.Conv2d(self.in_channels, 1, kernel_size=3, padding=1)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        logits = []
        bbox_reg = []
        objectness = []
        for feature in features:
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            logits.append(self.cls_logits(cls_feature))
            bbox_reg.append(self.bbox_pred(box_feature))
            objectness.append(self.objectness_logits(box_feature))
        return {
            "cls_logits": logits,
            "bbox_regression": bbox_reg,
            "objectness_logits": objectness,
        }


@HEADS.register("YOLOHead")
class YOLOHead(YOLOXHead):
    """Compatibility alias for generic YOLO-style names."""


@HEADS.register(
    "RTMDetHead",
    aliases=("rtmdet", "rtmdet_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "rtmdet_anchor_free_outputs", "task_aligned_targets"),
    validation_status="runtime_validated",
    family="dense",
    summary="RTMDet-family anchor-free dense head with class and bbox branches.",
)
class RTMDetHead(FCOSDenseHead):
    """RTMDet-style dense head using anchor-free class and bbox branches."""

    def forward(self, features: list[Any] | tuple[Any, ...]):
        logits = []
        bbox_reg = []
        for feature in features:
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            logits.append(self.cls_logits(cls_feature))
            bbox_reg.append(self.bbox_pred(box_feature))
        return {"cls_logits": logits, "bbox_regression": bbox_reg}


class _AnchorBoxDenseHead(nn.Module):
    """Anchor-based class and box head shared by SSD-style detectors."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_anchors = _positive_int(num_anchors, "num_anchors")
        self.num_convs = _positive_int(num_convs, "num_convs")
        cls_tower = []
        box_tower = []
        for _ in range(self.num_convs):
            cls_tower.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1))
            cls_tower.append(nn.ReLU(inplace=True))
            box_tower.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1))
            box_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        self.cls_logits = nn.Conv2d(
            self.in_channels,
            self.num_anchors * self.num_classes,
            kernel_size=3,
            padding=1,
        )
        self.bbox_pred = nn.Conv2d(self.in_channels, self.num_anchors * 4, kernel_size=3, padding=1)

    def __call__(self, features: list[Any] | tuple[Any, ...]):
        return self.forward(features)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.box_tower(feature)))
        return {"cls_logits": logits, "bbox_regression": bbox_reg}


@HEADS.register(
    "SSDHead",
    aliases=("ssd", "ssd_head", "ssd300", "ssd512", "ssdlite"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "ssd_anchor_outputs", "max_iou_targets"),
    validation_status="runtime_validated",
    family="dense",
    summary="SSD-family anchor-based dense head with class and bbox branches.",
)
class SSDHead(_AnchorBoxDenseHead):
    """SSD-style anchor-based dense head."""


@HEADS.register(
    "EfficientDetHead",
    aliases=("efficientdet", "efficientdet_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "efficientdet_anchor_outputs", "max_iou_targets"),
    validation_status="runtime_validated",
    family="dense",
    summary="EfficientDet-family anchor-based dense head with repeated class and box subnets.",
)
class EfficientDetHead(_AnchorBoxDenseHead):
    """EfficientDet-style anchor head with deeper shared subnets."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_convs=num_convs,
        )


@HEADS.register("SABLHead")
class SABLHead(ATSSDenseHead):
    """SABL-style head alias."""


@HEADS.register(
    "TOODHead",
    aliases=("TOOD", "tood_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="TOOD-family task-aligned dense head using the native FCOS output contract.",
)
class TOODHead(FCOSDenseHead):
    """TOOD-style anchor-free head alias."""


@HEADS.register("SOLOV2Head")
class SOLOV2Head(FCOSDenseHead):
    """SOLOv2-style head alias for a single-stage segmentation-friendly path."""


@HEADS.register(
    "CenterNetHead",
    aliases=("CenterNet",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="compatibility_alias",
    family="dense",
    summary="CenterNet-family head alias routed through FCOS-compatible outputs.",
)
class CenterNetHead(FCOSDenseHead):
    """Compatibility alias for CenterNet-style dense heads."""


@HEADS.register(
    "GFLHead",
    aliases=("gfl", "gfl_head", "gfocal", "gfocal_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="GFL-style dense head using the native ATSS output contract.",
)
class GFLDenseHead(ATSSDenseHead):
    """GFL-style dense head on the current anchor-based dense seam."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 4,
        loss_cls: str = "quality_focal",
        loss_dfl: str = "distribution_focal",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_convs=num_convs,
        )
        LOSSES.import_modules("simpledet.native.losses")
        self.loss_cls = LOSSES.get(loss_cls)()
        self.loss_dfl = LOSSES.get(loss_dfl)()


@HEADS.register(
    "GFLV2Head",
    aliases=("GFLV2", "gflv2_head", "gfocalv2", "gfocalv2_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="GFLv2-style dense head using the native ATSS output contract.",
)
class GFLV2Head(GFLDenseHead):
    """Compatibility alias for GFLv2 variants."""


@HEADS.register(
    "PAAHead",
    aliases=("PAA", "paa_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="PAA-family probabilistic anchor-assignment dense head using the native ATSS output contract.",
)
class PAAHead(ATSSDenseHead):
    """PAA-style anchor-based dense head."""


@HEADS.register(
    "DDODHead",
    aliases=("DDOD", "ddod_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="DDOD-family decoupled dense head using the native ATSS output contract.",
)
class DDODHead(ATSSDenseHead):
    """DDOD-style anchor-based dense head."""


@HEADS.register(
    "AutoAssignHead",
    aliases=("AutoAssign", "auto_assign_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="AutoAssign-family anchor-free dense head using the native FCOS output contract.",
)
class AutoAssignHead(FCOSDenseHead):
    """AutoAssign-style anchor-free dense head."""


@HEADS.register(
    "NASFCOSHead",
    aliases=("NAS-FCOS", "nas_fcos_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="NAS-FCOS-family anchor-free dense head using the native FCOS output contract.",
)
class NASFCOSHead(FCOSDenseHead):
    """NAS-FCOS-style anchor-free dense head."""


def _resolve_head_name(requested: str) -> str:
    normalized = "".join(char for char in str(requested).lower() if char.isalnum())
    if not normalized:
        raise ValueError("Head type cannot be empty.")

    try:
        return HEADS.resolve_name(str(requested))
    except KeyError:
        pass

    if normalized in {"retina", "retinanet", "retinanethead", "retinahead"}:
        return "RetinaHead"
    if normalized in {
        "freeanchor",
        "freeanchorhead",
        "freeanchorretina",
        "freeanchorretinahead",
    }:
        return "FreeAnchorRetinaHead"
    if normalized in {"rpn", "rpnhead"}:
        return "RPNHead"
    if normalized.startswith("fsaf"):
        return "FSAFHead"
    if normalized.startswith("fcos"):
        if normalized in {"fcosheadv2", "fcosv2head"}:
            return "FCOSHeadV2"
        return "FCOSHead"
    if normalized.startswith("atss"):
        return "ATSSHead"
    if normalized.startswith("gflv2") or normalized.startswith("gfocalv2"):
        return "GFLV2Head"
    if normalized.startswith("gfl"):
        return "GFLHead"
    if normalized.startswith("paa"):
        return "PAAHead"
    if normalized.startswith("ddod"):
        return "DDODHead"
    if normalized.startswith("autoassign"):
        return "AutoAssignHead"
    if normalized.startswith("nasfcos"):
        return "NASFCOSHead"
    if normalized in {"yolof", "yolofhead"}:
        return "YOLOFHead"
    if normalized.startswith("yolo"):
        return "YOLOXHead"
    if normalized.startswith("rtmdet"):
        return "RTMDetHead"
    if normalized.startswith("centernet"):
        return "CenterNetHead"
    if normalized.startswith("vfnet"):
        return "VFNetHead"
    if normalized.startswith("reppoints"):
        return "RepPointsHead"
    if normalized.startswith("fovea"):
        return "FoveaHead"
    if normalized.startswith("efficientdet"):
        return "EfficientDetHead"
    if normalized.startswith("ssd"):
        return "SSDHead"
    if normalized.startswith("sabl"):
        return "SABLHead"
    if normalized.startswith("tood"):
        return "TOODHead"
    if normalized.startswith("solov2"):
        return "SOLOV2Head"

    return HEADS.resolve_name(str(requested))


def build_native_head(head_plan, *, out_channels: int, num_classes: int):
    if head_plan is None or str(head_plan.type).strip().lower() in {"", "auto"}:
        head_plan = type("AutoHeadPlan", (), {"type": "RetinaHead", "params": {}})()

    head_type = str(head_plan.type)
    head_type = _resolve_head_name(head_type)
    params = dict(getattr(head_plan, "params", {}))
    params.setdefault("in_channels", int(out_channels))
    params.setdefault("num_classes", int(num_classes))
    params["in_channels"] = _positive_int(params["in_channels"], "in_channels")
    params["num_classes"] = _positive_int(params["num_classes"], "num_classes")
    factory = HEADS.get(head_type)
    if head_type in set(HEADS.names()) and _head_accepts_num_anchors(factory):
        params = _default_num_anchors(params)
        params["num_anchors"] = _positive_int(params["num_anchors"], "num_anchors")
    head = factory(**params)
    spec = HeadSpec(
        name=head_type,
        num_classes=int(params["num_classes"]),
        in_channels=int(params["in_channels"]),
        num_anchors=int(params.get("num_anchors", 1)),
    )
    return head, spec
