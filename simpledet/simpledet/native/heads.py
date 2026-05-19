"""Native detection heads for the Lightning backend."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import HEADS, LOSSES

require_dependency("torch", "native heads")
import torch  # noqa: E402
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


def _dropout(value: Any) -> float:
    resolved = float(value)
    if resolved < 0.0 or resolved >= 1.0:
        raise ValueError("dropout must be in the range [0, 1).")
    return resolved


def _validate_attention_config(*, hidden_dim: int, num_heads: int) -> None:
    if hidden_dim % num_heads != 0:
        raise ValueError(
            "Transformer head hidden_dim must be divisible by num_heads before forward execution."
        )


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
    aliases=("CenterNet", "centernet_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "keypoint_heatmap_outputs", "dense_anchor_free_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="CenterNet-family keypoint heatmap head with dense compatibility outputs.",
)
class CenterNetHead(nn.Module):
    """CenterNet-style heatmap head with width-height and offset branches."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_convs: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_convs = _positive_int(num_convs, "num_convs")
        tower = []
        for _ in range(self.num_convs):
            tower.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1))
            tower.append(nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*tower)
        self.heatmap_head = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=3, padding=1)
        self.wh_head = nn.Conv2d(self.in_channels, 2, kernel_size=3, padding=1)
        self.offset_head = nn.Conv2d(self.in_channels, 2, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, 4, kernel_size=3, padding=1)
        if hasattr(nn, "init"):
            nn.init.constant_(self.heatmap_head.bias, -2.19)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        heatmap_logits = []
        heatmaps = []
        widths_heights = []
        offsets = []
        bbox_regression = []
        centerness = []
        for feature in features:
            encoded = self.tower(feature)
            logits = self.heatmap_head(encoded)
            heatmap = logits.sigmoid()
            heatmap_logits.append(logits)
            heatmaps.append(heatmap)
            widths_heights.append(self.wh_head(encoded))
            offsets.append(self.offset_head(encoded))
            bbox_regression.append(self.bbox_pred(encoded))
            centerness.append(heatmap.amax(dim=1, keepdim=True))
        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmaps,
            "wh": widths_heights,
            "offset": offsets,
            "cls_logits": heatmap_logits,
            "bbox_regression": bbox_regression,
            "centerness": centerness,
        }


@HEADS.register(
    "CornerNetHead",
    aliases=("CornerNet", "cornernet_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "corner_keypoint_heatmap_outputs"),
    validation_status="runtime_validated",
    family="dense",
    summary="CornerNet-family paired-corner heatmap head with embeddings and offsets.",
)
class CornerNetHead(nn.Module):
    """CornerNet-style paired top-left and bottom-right heatmap head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_convs: int = 2,
        embedding_dim: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_convs = _positive_int(num_convs, "num_convs")
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.top_left_tower = self._make_tower()
        self.bottom_right_tower = self._make_tower()
        self.top_left_heatmap = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=3, padding=1)
        self.bottom_right_heatmap = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=3, padding=1)
        self.top_left_embedding = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=3, padding=1)
        self.bottom_right_embedding = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=3, padding=1)
        self.top_left_offset = nn.Conv2d(self.in_channels, 2, kernel_size=3, padding=1)
        self.bottom_right_offset = nn.Conv2d(self.in_channels, 2, kernel_size=3, padding=1)
        if hasattr(nn, "init"):
            nn.init.constant_(self.top_left_heatmap.bias, -2.19)
            nn.init.constant_(self.bottom_right_heatmap.bias, -2.19)

    def _make_tower(self) -> nn.Sequential:
        layers = []
        for _ in range(self.num_convs):
            layers.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, features: list[Any] | tuple[Any, ...]):
        top_left_heatmaps = []
        bottom_right_heatmaps = []
        top_left_embeddings = []
        bottom_right_embeddings = []
        top_left_offsets = []
        bottom_right_offsets = []
        for feature in features:
            top_left = self.top_left_tower(feature)
            bottom_right = self.bottom_right_tower(feature)
            top_left_logits = self.top_left_heatmap(top_left)
            bottom_right_logits = self.bottom_right_heatmap(bottom_right)
            top_left_heatmaps.append(top_left_logits)
            bottom_right_heatmaps.append(bottom_right_logits)
            top_left_embeddings.append(self.top_left_embedding(top_left))
            bottom_right_embeddings.append(self.bottom_right_embedding(bottom_right))
            top_left_offsets.append(self.top_left_offset(top_left))
            bottom_right_offsets.append(self.bottom_right_offset(bottom_right))
        return {
            "top_left_heatmap": top_left_heatmaps,
            "bottom_right_heatmap": bottom_right_heatmaps,
            "top_left_embedding": top_left_embeddings,
            "bottom_right_embedding": bottom_right_embeddings,
            "top_left_offset": top_left_offsets,
            "bottom_right_offset": bottom_right_offsets,
        }


@HEADS.register(
    "DETRHead",
    aliases=("DETR", "detr_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "query_set_predictions", "class_box_predictions"),
    validation_status="runtime_validated",
    family="transformer",
    summary="DETR-family transformer head with query class and box predictions.",
)
class DETRHead(nn.Module):
    """Small native DETR-style encoder-decoder head."""

    def __init__(
        self,
        *,
        in_channels: int = 256,
        num_classes: int,
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.num_queries = _positive_int(num_queries, "num_queries")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.num_encoder_layers = _positive_int(num_encoder_layers, "num_encoder_layers")
        self.num_decoder_layers = _positive_int(num_decoder_layers, "num_decoder_layers")
        self.dim_feedforward = _positive_int(dim_feedforward, "dim_feedforward")
        self.dropout = _dropout(dropout)
        self.activation = str(activation)
        _validate_attention_config(hidden_dim=self.hidden_dim, num_heads=self.num_heads)

        self.input_proj = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(self, features: list[Any] | tuple[Any, ...]):
        memory = self.encoder(self._flatten_features(features))
        batch_size = int(memory.shape[0])
        target = self._query_embeddings(batch_size)
        decoded = self.decoder(target, memory)
        outputs = {
            "pred_logits": self.class_embed(decoded),
            "pred_boxes": self.bbox_embed(decoded).sigmoid(),
        }
        outputs["class_logits"] = outputs["pred_logits"]
        outputs["box_predictions"] = outputs["pred_boxes"]
        return self._extra_outputs(outputs, batch_size)

    def _flatten_features(self, features: list[Any] | tuple[Any, ...]):
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError("Transformer heads require a non-empty feature map sequence.")
        feature = features[-1]
        self._validate_feature_channels(feature)
        return self.input_proj(feature).flatten(2).permute(0, 2, 1)

    def _validate_feature_channels(self, feature: Any) -> None:
        if len(getattr(feature, "shape", ())) != 4:
            raise ValueError("Transformer heads expect 4D NCHW feature tensors.")
        channels = int(feature.shape[1])
        if channels != self.in_channels:
            raise ValueError(
                f"Transformer head expected feature channels {self.in_channels}, got {channels}."
            )

    def _query_embeddings(self, batch_size: int):
        return self.query_embed.weight.unsqueeze(0).expand(int(batch_size), -1, -1)

    def _extra_outputs(self, outputs: dict[str, Any], batch_size: int) -> dict[str, Any]:
        return outputs


@HEADS.register(
    "ConditionalDETRHead",
    aliases=("ConditionalDETR", "conditional_detr_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "conditional_query_set_predictions", "class_box_predictions"),
    validation_status="runtime_validated",
    family="transformer",
    summary="Conditional DETR-family transformer head with learned conditional query content.",
)
class ConditionalDETRHead(DETRHead):
    """Conditional DETR-style head with a second learned query-content stream."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conditional_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

    def _query_embeddings(self, batch_size: int):
        conditional = self.conditional_query_embed.weight.unsqueeze(0).expand(int(batch_size), -1, -1)
        return super()._query_embeddings(batch_size) + conditional


@HEADS.register(
    "DABDETRHead",
    aliases=("DAB-DETR", "dab_detr_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid", "anchor_query_set_predictions", "class_box_predictions"),
    validation_status="runtime_validated",
    family="transformer",
    summary="DAB-DETR-family transformer head with dynamic anchor box queries.",
)
class DABDETRHead(DETRHead):
    """DAB-DETR-style head with learned reference boxes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reference_points = nn.Embedding(self.num_queries, 4)
        self.reference_proj = nn.Linear(4, self.hidden_dim)

    def _query_embeddings(self, batch_size: int):
        references = self.reference_points.weight.sigmoid()
        reference_content = self.reference_proj(references).unsqueeze(0).expand(int(batch_size), -1, -1)
        return super()._query_embeddings(batch_size) + reference_content

    def _extra_outputs(self, outputs: dict[str, Any], batch_size: int) -> dict[str, Any]:
        references = self.reference_points.weight.sigmoid().unsqueeze(0).expand(int(batch_size), -1, -1)
        outputs["reference_points"] = references
        outputs["pred_boxes"] = ((outputs["pred_boxes"] + references) * 0.5).clamp(0.0, 1.0)
        outputs["box_predictions"] = outputs["pred_boxes"]
        return outputs


@HEADS.register(
    "DeformableDETRHead",
    aliases=("DeformableDETR", "deformable_detr_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("multi_scale_feature_pyramid", "query_set_predictions", "class_box_predictions"),
    validation_status="runtime_validated",
    family="transformer",
    summary="Deformable DETR-family multi-scale transformer head with level embeddings.",
)
class DeformableDETRHead(DETRHead):
    """Multi-scale transformer head for Deformable DETR-style set prediction."""

    def __init__(self, *, num_feature_levels: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_feature_levels = _positive_int(num_feature_levels, "num_feature_levels")
        self.level_projections = nn.ModuleList(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=1)
            for _ in range(self.num_feature_levels)
        )
        self.level_embed = nn.Embedding(self.num_feature_levels, self.hidden_dim)

    def _flatten_features(self, features: list[Any] | tuple[Any, ...]):
        if not isinstance(features, (list, tuple)) or len(features) < self.num_feature_levels:
            raise ValueError(
                "DeformableDETRHead requires at least "
                f"{self.num_feature_levels} feature levels."
            )
        tokens = []
        selected = list(features)[-self.num_feature_levels :]
        for level_index, (projection, feature) in enumerate(zip(self.level_projections, selected)):
            self._validate_feature_channels(feature)
            projected = projection(feature).flatten(2).permute(0, 2, 1)
            level_bias = self.level_embed.weight[level_index].view(1, 1, self.hidden_dim)
            tokens.append(projected + level_bias)
        return torch.cat(tokens, dim=1)


@HEADS.register(
    "DINOHead",
    aliases=("DINO", "dino_head"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("multi_scale_feature_pyramid", "denoising_query_set_predictions", "class_box_predictions"),
    validation_status="runtime_validated",
    family="transformer",
    summary="DINO-family transformer head with anchor and denoising query embeddings.",
)
class DINOHead(DeformableDETRHead):
    """DINO-style head using multi-scale features plus denoising query content."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reference_points = nn.Embedding(self.num_queries, 4)
        self.reference_proj = nn.Linear(4, self.hidden_dim)
        self.denoising_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

    def _query_embeddings(self, batch_size: int):
        references = self.reference_points.weight.sigmoid()
        reference_content = self.reference_proj(references).unsqueeze(0).expand(int(batch_size), -1, -1)
        denoising = self.denoising_query_embed.weight.unsqueeze(0).expand(int(batch_size), -1, -1)
        return super()._query_embeddings(batch_size) + reference_content + denoising

    def _extra_outputs(self, outputs: dict[str, Any], batch_size: int) -> dict[str, Any]:
        references = self.reference_points.weight.sigmoid().unsqueeze(0).expand(int(batch_size), -1, -1)
        outputs["reference_points"] = references
        outputs["pred_boxes"] = ((outputs["pred_boxes"] + references) * 0.5).clamp(0.0, 1.0)
        outputs["box_predictions"] = outputs["pred_boxes"]
        return outputs


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
    if normalized.startswith("cornernet"):
        return "CornerNetHead"
    if normalized in {"detr", "detrhead"}:
        return "DETRHead"
    if normalized.startswith("conditionaldetr"):
        return "ConditionalDETRHead"
    if normalized.startswith("dabdetr"):
        return "DABDETRHead"
    if normalized.startswith("deformabledetr"):
        return "DeformableDETRHead"
    if normalized.startswith("dino"):
        return "DINOHead"
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
