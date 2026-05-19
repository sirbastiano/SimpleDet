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
        require_dependency("torchvision", "native heads")
        from torchvision.models.detection.retinanet import RetinaNetHead

        self.in_channels = int(in_channels)
        self.num_anchors = int(num_anchors)
        self.num_classes = int(num_classes)
        self.head = RetinaNetHead(self.in_channels, self.num_anchors, self.num_classes)

    def __call__(self, features: list[Any] | tuple[Any, ...]):
        return self.forward(features)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        return self.head(list(features))


@HEADS.register("RetinaNetHead")
class RetinaNetHead(RetinaDenseHead):
    """Compatibility alias used by external detector configs."""


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

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_convs = int(num_convs)
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
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        self.num_convs = int(num_convs)
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
    "VFNetHead",
    aliases=("VFNet",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="compatibility_alias",
    family="dense",
    summary="VFNet-family head alias routed through ATSS-compatible outputs.",
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
    aliases=("RepPoints", "ReppointsHead"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="compatibility_alias",
    family="dense",
    summary="RepPoints-family head alias routed through ATSS-compatible outputs.",
)
class RepPointsHead(ATSSDenseHead):
    """Compatibility alias for Reppoints-style dense heads."""


class ReppointsHead(RepPointsHead):
    """Alternate alias for Reppoints naming."""


@HEADS.register(
    "FoveaHead",
    aliases=("FOVEA",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="compatibility_alias",
    family="dense",
    summary="Fovea-family head alias routed through ATSS-compatible outputs.",
)
class FoveaHead(ATSSDenseHead):
    """Compatibility alias for Fovea-style dense heads."""


@HEADS.register(
    "YOLOFHead",
    aliases=("YOLOF",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_free_outputs"),
    validation_status="compatibility_alias",
    family="dense",
    summary="YOLOF-family head alias routed through FCOS-compatible outputs.",
)
class YOLOFHead(FCOSDenseHead):
    """Anchor-free YOLOF-style head alias."""


@HEADS.register("YOLOXHead")
class YOLOXHead(FCOSDenseHead):
    """YOLOX-style anchor-free head alias."""


@HEADS.register("YOLOHead")
class YOLOHead(YOLOXHead):
    """Compatibility alias for generic YOLO-style names."""


@HEADS.register("SSDHead")
class SSDHead(ATSSDenseHead):
    """Legacy SSD-style anchor-based head alias."""


@HEADS.register("SABLHead")
class SABLHead(ATSSDenseHead):
    """SABL-style head alias."""


@HEADS.register("TOODHead")
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
    aliases=("gfl", "gfl_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "dense_anchor_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="GFL-style dense head.",
)
class GFLDenseHead(ATSSDenseHead):
    """GFL-style dense head on the current anchor-based dense seam."""


@HEADS.register("GFLV2Head")
class GFLV2Head(ATSSDenseHead):
    """Compatibility alias for GFLv2 variants."""


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
    if normalized.startswith("fcos"):
        if normalized in {"fcosheadv2", "fcosv2head"}:
            return "FCOSHeadV2"
        return "FCOSHead"
    if normalized.startswith("atss"):
        return "ATSSHead"
    if normalized.startswith("gfl"):
        return "GFLHead"
    if normalized in {"yolof", "yolofhead"}:
        return "YOLOFHead"
    if normalized.startswith("yolo"):
        return "YOLOXHead"
    if normalized.startswith("centernet"):
        return "CenterNetHead"
    if normalized.startswith("vfnet"):
        return "VFNetHead"
    if normalized.startswith("reppoints"):
        return "RepPointsHead"
    if normalized.startswith("fovea"):
        return "FoveaHead"
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
    factory = HEADS.get(head_type)
    if head_type in set(HEADS.names()) and _head_accepts_num_anchors(factory):
        params = _default_num_anchors(params)
    head = factory(**params)
    spec = HeadSpec(
        name=head_type,
        num_classes=int(params["num_classes"]),
        in_channels=int(params["in_channels"]),
        num_anchors=int(params.get("num_anchors", 1)),
    )
    return head, spec
