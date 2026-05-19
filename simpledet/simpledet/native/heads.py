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
import torch.nn.functional as F  # noqa: E402


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


def _roi_feat_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("roi_feat_size must be a positive integer or a pair of positive integers.")
        return (
            _positive_int(value[0], "roi_feat_size"),
            _positive_int(value[1], "roi_feat_size"),
        )
    resolved = _positive_int(value, "roi_feat_size")
    return resolved, resolved


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


_ROI_BBOX_DEPENDENCIES = (("torch", "cpu"),)
_ROI_BBOX_CONTRACTS = ("roi_features", "roi_bbox_head_outputs", "roi_bbox_targets")
_ROI_MASK_CONTRACTS = ("roi_features", "roi_mask_head_outputs", "roi_mask_targets", "roi_mask_decoding")
_ROI_GRID_CONTRACTS = ("roi_features", "roi_grid_head_outputs", "roi_grid_targets", "roi_grid_decoding")


def _non_negative_int(value: Any, name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _as_4d_roi_tensor(roi_features: Any, *, in_channels: int, head_name: str):
    feature = roi_features
    if isinstance(feature, (list, tuple)):
        if len(feature) != 1:
            raise ValueError(f"{head_name} expects a single pooled ROI feature tensor.")
        feature = feature[0]
    if not torch.is_tensor(feature):
        feature = torch.as_tensor(feature, dtype=torch.float32)
    if feature.dim() != 4:
        raise ValueError(f"{head_name} expects pooled ROI features with shape (N, C, H, W).")
    if int(feature.shape[1]) != int(in_channels):
        raise ValueError(
            f"{head_name} expected feature channels {int(in_channels)}, "
            f"got {int(feature.shape[1])}."
        )
    return feature


class _RoIBBoxHeadBase(nn.Module):
    """Shared fully connected ROI bbox head contract."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        roi_feat_size: int | tuple[int, int] = 7,
        conv_out_channels: int | None = None,
        fc_out_channels: int = 1024,
        num_shared_convs: int = 0,
        num_shared_fcs: int = 2,
        with_avg_pool: bool = False,
        reg_class_agnostic: bool = False,
        class_agnostic: bool | None = None,
        loss_weight: float = 1.0,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.roi_feat_size = _roi_feat_size(roi_feat_size)
        self.conv_out_channels = _positive_int(
            conv_out_channels if conv_out_channels is not None else in_channels,
            "conv_out_channels",
        )
        self.fc_out_channels = _positive_int(fc_out_channels, "fc_out_channels")
        self.num_shared_convs = int(num_shared_convs)
        self.num_shared_fcs = int(num_shared_fcs)
        if self.num_shared_convs < 0:
            raise ValueError("num_shared_convs must be non-negative.")
        if self.num_shared_fcs < 0:
            raise ValueError("num_shared_fcs must be non-negative.")
        self.with_avg_pool = bool(with_avg_pool)
        self.reg_class_agnostic = bool(reg_class_agnostic if class_agnostic is None else class_agnostic)
        self.loss_weight = float(loss_weight)

        self.shared_convs = self._make_conv_tower(
            num_convs=self.num_shared_convs,
            in_channels=self.in_channels,
            out_channels=self.conv_out_channels,
        )
        shared_channels = self.conv_out_channels if self.num_shared_convs else self.in_channels
        fc_input_dim = self._flattened_feature_dim(shared_channels)
        self.shared_fcs, shared_output_dim = self._make_fc_tower(
            num_fcs=self.num_shared_fcs,
            input_dim=fc_input_dim,
            output_dim=self.fc_out_channels,
        )
        self.fc_cls = nn.Linear(shared_output_dim, self.num_classes + 1)
        self.fc_reg = nn.Linear(shared_output_dim, self.bbox_pred_channels)

    @property
    def bbox_pred_channels(self) -> int:
        if self.reg_class_agnostic:
            return 4
        return (self.num_classes + 1) * 4

    def forward(self, roi_features: Any) -> dict[str, Any]:
        shared = self._shared_representation(roi_features)
        return {
            "cls_score": self.fc_cls(shared),
            "bbox_pred": self.fc_reg(shared),
        }

    def get_targets(self, proposals: Any, gt_boxes: Any, gt_labels: Any, **kwargs: Any):
        from .roi import build_roi_bbox_targets

        return build_roi_bbox_targets(proposals, gt_boxes, gt_labels, **kwargs)

    def loss(self, outputs: Any, targets: Any) -> dict[str, Any]:
        cls_score, bbox_pred = self._unpack_outputs(outputs)
        labels = self._target_tensor(targets, "labels", dtype=torch.long, device=cls_score.device).reshape(-1)
        label_weights = self._target_tensor(
            targets,
            "label_weights",
            default=cls_score.new_ones((labels.shape[0],)),
            dtype=cls_score.dtype,
            device=cls_score.device,
        ).reshape(-1)
        bbox_targets = self._target_tensor(
            targets,
            "bbox_targets",
            dtype=bbox_pred.dtype,
            device=bbox_pred.device,
        )
        bbox_weights = self._target_tensor(
            targets,
            "bbox_weights",
            default=torch.ones_like(bbox_targets),
            dtype=bbox_pred.dtype,
            device=bbox_pred.device,
        )

        num_rois = int(cls_score.shape[0])
        self._validate_loss_rows(num_rois, labels, label_weights, bbox_pred)
        bbox_targets = self._normalize_bbox_targets(bbox_targets, num_rois, name="bbox_targets")
        bbox_weights = self._normalize_bbox_targets(bbox_weights, num_rois, name="bbox_weights")

        clamped_labels = labels.clamp(min=0, max=self.num_classes)
        cls_losses = F.cross_entropy(cls_score, clamped_labels, reduction="none")
        loss_cls = (cls_losses * label_weights).sum() / label_weights.sum().clamp_min(1.0)

        positive = torch.nonzero(clamped_labels > 0, as_tuple=False).reshape(-1)
        if positive.numel() == 0:
            loss_bbox = bbox_pred.sum() * 0.0
        else:
            positive_labels = clamped_labels.index_select(0, positive)
            positive_bbox_pred = bbox_pred.index_select(0, positive)
            positive_bbox_targets = self._select_bbox_rows(
                bbox_targets,
                positive,
                positive_labels,
                name="bbox_targets",
            )
            positive_bbox_weights = self._select_bbox_rows(
                bbox_weights,
                positive,
                positive_labels,
                name="bbox_weights",
            )
            selected_bbox_pred = self._select_bbox_predictions(
                positive_bbox_pred,
                positive_labels,
            )
            weighted_pred = selected_bbox_pred * positive_bbox_weights
            weighted_target = positive_bbox_targets * positive_bbox_weights
            normalizer = positive_bbox_weights.sum().clamp_min(1.0)
            loss_bbox = F.smooth_l1_loss(weighted_pred, weighted_target, reduction="sum") / normalizer

        loss_bbox = loss_bbox * self.loss_weight
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_total": loss_cls + loss_bbox,
        }

    def _shared_representation(self, roi_features: Any):
        feature = self._as_roi_tensor(roi_features)
        if feature.dim() == 4:
            feature = self.shared_convs(feature)
            flattened = self._flatten_spatial_features(feature)
        elif feature.dim() == 2:
            if self.num_shared_convs:
                raise ValueError("ROI bbox heads with shared convs require 4D NCHW ROI features.")
            flattened = feature
            expected = self._flattened_feature_dim(self.in_channels)
            if int(flattened.shape[1]) != expected:
                raise ValueError(
                    f"ROI bbox head expected flattened feature dimension {expected}, "
                    f"got {int(flattened.shape[1])}."
                )
        else:
            raise ValueError("ROI bbox heads expect pooled features with shape (N, C, H, W) or (N, C).")
        return self.shared_fcs(flattened)

    def _as_roi_tensor(self, roi_features: Any):
        feature = roi_features
        if isinstance(feature, (list, tuple)):
            if len(feature) != 1:
                raise ValueError("ROI bbox heads expect a single pooled ROI feature tensor.")
            feature = feature[0]
        if not torch.is_tensor(feature):
            feature = torch.as_tensor(feature, dtype=torch.float32)
        if feature.dim() == 4 and int(feature.shape[1]) != self.in_channels:
            raise ValueError(
                f"ROI bbox head expected feature channels {self.in_channels}, "
                f"got {int(feature.shape[1])}."
            )
        return feature

    def _flatten_spatial_features(self, feature: Any):
        if self.with_avg_pool:
            return torch.flatten(F.adaptive_avg_pool2d(feature, (1, 1)), 1)
        expected_h, expected_w = self.roi_feat_size
        actual_h, actual_w = int(feature.shape[-2]), int(feature.shape[-1])
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise ValueError(
                "ROI bbox head expected pooled feature size "
                f"{(expected_h, expected_w)}, got {(actual_h, actual_w)}."
            )
        return torch.flatten(feature, 1)

    def _flattened_feature_dim(self, channels: int) -> int:
        if self.with_avg_pool:
            return int(channels)
        return int(channels) * self.roi_feat_size[0] * self.roi_feat_size[1]

    def _make_conv_tower(self, *, num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_channels = int(in_channels)
        for _ in range(int(num_convs)):
            layers.append(nn.Conv2d(current_channels, int(out_channels), kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = int(out_channels)
        return nn.Sequential(*layers)

    def _make_fc_tower(self, *, num_fcs: int, input_dim: int, output_dim: int) -> tuple[nn.Sequential, int]:
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(int(num_fcs)):
            layers.append(nn.Linear(current_dim, int(output_dim)))
            layers.append(nn.ReLU(inplace=True))
            current_dim = int(output_dim)
        return nn.Sequential(*layers), current_dim

    def _unpack_outputs(self, outputs: Any) -> tuple[Any, Any]:
        if isinstance(outputs, dict):
            return outputs["cls_score"], outputs["bbox_pred"]
        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            return outputs[0], outputs[1]
        raise ValueError("ROI bbox head loss expects outputs with cls_score and bbox_pred.")

    def _target_tensor(
        self,
        targets: Any,
        name: str,
        *,
        default: Any | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
    ):
        if isinstance(targets, dict):
            value = targets.get(name, default)
        else:
            value = getattr(targets, name, default)
        if value is None:
            raise ValueError(f"ROI bbox head loss requires targets.{name}.")
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _validate_loss_rows(self, num_rois: int, labels: Any, label_weights: Any, bbox_pred: Any) -> None:
        if int(labels.shape[0]) != num_rois:
            raise ValueError("labels must match the number of ROI predictions.")
        if int(label_weights.shape[0]) != num_rois:
            raise ValueError("label_weights must match the number of ROI predictions.")
        if bbox_pred.dim() != 2 or int(bbox_pred.shape[0]) != num_rois:
            raise ValueError("bbox_pred must have shape (N, 4) or (N, (num_classes + 1) * 4).")
        if int(bbox_pred.shape[1]) != self.bbox_pred_channels:
            raise ValueError(
                f"bbox_pred expected second dimension {self.bbox_pred_channels}, "
                f"got {int(bbox_pred.shape[1])}."
            )

    def _normalize_bbox_targets(self, tensor: Any, num_rois: int, *, name: str):
        if int(tensor.shape[0]) != num_rois:
            raise ValueError(f"{name} must match the number of ROI predictions.")
        if self.reg_class_agnostic:
            if tensor.dim() != 2 or int(tensor.shape[1]) != 4:
                raise ValueError(
                    "class-agnostic regression expects bbox_targets with shape (N, 4)."
                )
            return tensor
        if tensor.dim() == 2:
            width = int(tensor.shape[1])
            if width == 4:
                return tensor
            if width == (self.num_classes + 1) * 4:
                return tensor.reshape(num_rois, self.num_classes + 1, 4)
        elif tensor.dim() == 3 and int(tensor.shape[1]) == self.num_classes + 1 and int(tensor.shape[2]) == 4:
            return tensor
        raise ValueError(
            f"{name} must have shape (N, 4), (N, (num_classes + 1) * 4), "
            "or (N, num_classes + 1, 4)."
        )

    def _select_bbox_predictions(self, bbox_pred: Any, labels: Any):
        if self.reg_class_agnostic:
            return bbox_pred
        from .roi import select_class_specific_bbox_deltas

        return select_class_specific_bbox_deltas(
            bbox_pred,
            labels,
            num_classes=self.num_classes,
        )

    def _select_bbox_rows(self, tensor: Any, indices: Any, labels: Any, *, name: str):
        selected = tensor.index_select(0, indices)
        if selected.dim() == 2:
            if int(selected.shape[1]) != 4:
                raise ValueError(f"{name} selected rows must have 4 bbox columns.")
            return selected
        row_indices = torch.arange(selected.shape[0], dtype=torch.long, device=selected.device)
        label_tensor = labels.clamp(min=0, max=self.num_classes)
        return selected[row_indices, label_tensor]


@HEADS.register(
    "Shared2FCBBoxHead",
    aliases=("shared_2fc_bbox_head", "shared2fc"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=_ROI_BBOX_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
    summary="Native shared two-FC ROI bbox head.",
)
class Shared2FCBBoxHead(_RoIBBoxHeadBase):
    """Two fully connected ROI bbox head used by common two-stage detectors."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("num_shared_convs", 0)
        kwargs.setdefault("num_shared_fcs", 2)
        super().__init__(**kwargs)


@HEADS.register(
    "ConvFCBBoxHead",
    aliases=("convfc_bbox_head", "convfc"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=_ROI_BBOX_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
    summary="Native ROI bbox head with shared convolution and FC towers.",
)
class ConvFCBBoxHead(_RoIBBoxHeadBase):
    """Configurable convolution plus fully connected ROI bbox head."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("num_shared_convs", 1)
        kwargs.setdefault("num_shared_fcs", 1)
        super().__init__(**kwargs)


@HEADS.register(
    "DoubleConvFCBBoxHead",
    aliases=("double_convfc_bbox_head",),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=_ROI_BBOX_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
    summary="Native Double-Head ROI bbox head with separate classification and regression towers.",
)
class DoubleConvFCBBoxHead(_RoIBBoxHeadBase):
    """Double-head style ROI bbox head with distinct cls and reg branches."""

    def __init__(
        self,
        *,
        num_reg_convs: int = 2,
        num_cls_fcs: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_shared_convs", 0)
        kwargs.setdefault("num_shared_fcs", 0)
        super().__init__(**kwargs)
        self.num_reg_convs = _positive_int(num_reg_convs, "num_reg_convs")
        self.num_cls_fcs = _positive_int(num_cls_fcs, "num_cls_fcs")
        self.reg_convs = self._make_conv_tower(
            num_convs=self.num_reg_convs,
            in_channels=self.in_channels,
            out_channels=self.conv_out_channels,
        )
        reg_input_dim = self._flattened_feature_dim(self.conv_out_channels)
        self.reg_fcs, reg_output_dim = self._make_fc_tower(
            num_fcs=1,
            input_dim=reg_input_dim,
            output_dim=self.fc_out_channels,
        )
        cls_input_dim = self._flattened_feature_dim(self.in_channels)
        self.cls_fcs, cls_output_dim = self._make_fc_tower(
            num_fcs=self.num_cls_fcs,
            input_dim=cls_input_dim,
            output_dim=self.fc_out_channels,
        )
        self.fc_cls = nn.Linear(cls_output_dim, self.num_classes + 1)
        self.fc_reg = nn.Linear(reg_output_dim, self.bbox_pred_channels)

    def forward(self, roi_features: Any) -> dict[str, Any]:
        feature = self._as_roi_tensor(roi_features)
        if feature.dim() != 4:
            raise ValueError("DoubleConvFCBBoxHead requires 4D NCHW ROI features.")
        cls_feature = self.cls_fcs(self._flatten_spatial_features(feature))
        reg_feature = self.reg_convs(feature)
        reg_feature = self.reg_fcs(self._flatten_spatial_features(reg_feature))
        return {
            "cls_score": self.fc_cls(cls_feature),
            "bbox_pred": self.fc_reg(reg_feature),
        }


@HEADS.register(
    "DynamicBBoxHead",
    aliases=("dynamic_bbox_head", "dynamic_bbox"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=_ROI_BBOX_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
    summary="Native dynamic ROI bbox head with proposal feature mixing.",
)
class DynamicBBoxHead(Shared2FCBBoxHead):
    """Dynamic R-CNN style bbox head with an extra proposal mixing layer."""

    def __init__(self, *, num_dynamic_fcs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_dynamic_fcs = _positive_int(num_dynamic_fcs, "num_dynamic_fcs")
        layers: list[nn.Module] = []
        for _ in range(self.num_dynamic_fcs):
            layers.append(nn.Linear(self.fc_out_channels, self.fc_out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.dynamic_fcs = nn.Sequential(*layers)

    def _shared_representation(self, roi_features: Any):
        return self.dynamic_fcs(super()._shared_representation(roi_features))


@HEADS.register(
    "CascadeBBoxHead",
    aliases=("cascade_bbox_head", "cascade_bbox"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=(*_ROI_BBOX_CONTRACTS, "cascade_roi_refinement"),
    validation_status="runtime_validated",
    family="roi",
    summary="Native cascade ROI bbox head with stage-weighted bbox loss.",
)
class CascadeBBoxHead(Shared2FCBBoxHead):
    """Cascade R-CNN bbox head using the shared ROI target contract."""

    def __init__(self, *, stage_loss_weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(loss_weight=float(stage_loss_weight), **kwargs)
        self.stage_loss_weight = float(stage_loss_weight)

    def refine_proposals(self, proposals: Any, outputs: Any, labels: Any, **kwargs: Any):
        from .roi import refine_cascade_stage_proposals

        _cls_score, bbox_pred = self._unpack_outputs(outputs)
        return refine_cascade_stage_proposals(proposals, bbox_pred, labels, **kwargs)


@HEADS.register(
    "SABLHead",
    aliases=("sabl_head", "sabl_bbox_head", "side_aware_bbox_head"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=(*_ROI_BBOX_CONTRACTS, "side_aware_boundary_localization"),
    validation_status="runtime_validated",
    family="roi",
    summary="Native side-aware boundary localization ROI bbox head.",
)
class SABLHead(ConvFCBBoxHead):
    """SABL-style ROI bbox head with side-aware boundary projections."""

    def __init__(self, *, bucket_count: int = 7, **kwargs: Any) -> None:
        kwargs.setdefault("num_shared_convs", 1)
        kwargs.setdefault("num_shared_fcs", 1)
        super().__init__(**kwargs)
        self.bucket_count = _positive_int(bucket_count, "bucket_count")
        self.side_confidence = nn.Linear(self.fc_out_channels, self.bucket_count * 4)

    def forward(self, roi_features: Any) -> dict[str, Any]:
        shared = self._shared_representation(roi_features)
        return {
            "cls_score": self.fc_cls(shared),
            "bbox_pred": self.fc_reg(shared),
            "side_confidence": self.side_confidence(shared),
        }


@HEADS.register(
    "SparseRoIHead",
    aliases=("sparse_roi_head", "sparse_bbox_head", "sparse_rcnn_roi_head"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=(*_ROI_BBOX_CONTRACTS, "sparse_query_features"),
    validation_status="runtime_validated",
    family="roi",
    summary="Native Sparse R-CNN style ROI head for learned proposal features.",
)
class SparseRoIHead(DynamicBBoxHead):
    """Sparse R-CNN style ROI head that also accepts proposal feature tensors."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("roi_feat_size", 1)
        kwargs.setdefault("with_avg_pool", True)
        super().__init__(**kwargs)


@HEADS.register(
    "FCNMaskHead",
    aliases=("fcn_mask_head", "fcn_mask", "mask_head"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=_ROI_MASK_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
    summary="Native FCN-style ROI mask head with mask target and decode helpers.",
)
class FCNMaskHead(nn.Module):
    """FCN mask head used by Mask R-CNN style ROI heads."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        roi_feat_size: int | tuple[int, int] = 14,
        conv_out_channels: int = 256,
        num_convs: int = 4,
        upsample_scale: int = 2,
        output_size: int | tuple[int, int] | None = None,
        class_agnostic: bool = False,
        loss_weight: float = 1.0,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.roi_feat_size = _roi_feat_size(roi_feat_size)
        self.conv_out_channels = _positive_int(conv_out_channels, "conv_out_channels")
        self.num_convs = _non_negative_int(num_convs, "num_convs")
        self.upsample_scale = _positive_int(upsample_scale, "upsample_scale")
        self.output_size = _roi_feat_size(
            output_size
            if output_size is not None
            else (
                self.roi_feat_size[0] * self.upsample_scale,
                self.roi_feat_size[1] * self.upsample_scale,
            )
        )
        self.class_agnostic = bool(class_agnostic)
        self.loss_weight = float(loss_weight)
        self.mask_channels = 1 if self.class_agnostic else self.num_classes

        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for _ in range(self.num_convs):
            layers.append(nn.Conv2d(current_channels, self.conv_out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = self.conv_out_channels
        self.convs = nn.Sequential(*layers)
        if self.upsample_scale > 1:
            self.upsample = nn.ConvTranspose2d(
                current_channels,
                self.conv_out_channels,
                kernel_size=self.upsample_scale,
                stride=self.upsample_scale,
            )
            current_channels = self.conv_out_channels
        else:
            self.upsample = nn.Identity()
        self.conv_logits = nn.Conv2d(current_channels, self.mask_channels, kernel_size=1)

    def forward(self, roi_features: Any) -> dict[str, Any]:
        feature = self._as_roi_tensor(roi_features)
        if int(feature.shape[0]) == 0:
            mask_logits = feature.new_zeros((0, self.mask_channels, *self.output_size))
            return {"mask_logits": mask_logits, "mask_pred": mask_logits}
        encoded = self.convs(feature)
        encoded = self.upsample(encoded)
        if self.upsample_scale > 1:
            encoded = F.relu(encoded, inplace=True)
        if tuple(encoded.shape[-2:]) != self.output_size:
            encoded = F.interpolate(encoded, size=self.output_size, mode="bilinear", align_corners=False)
        mask_logits = self.conv_logits(encoded)
        return {"mask_logits": mask_logits, "mask_pred": mask_logits}

    def get_targets(self, proposals: Any, gt_masks: Any, matched_gt_indices: Any, **kwargs: Any):
        from .roi import build_roi_mask_targets

        kwargs.setdefault("output_size", self.output_size)
        return build_roi_mask_targets(proposals, gt_masks, matched_gt_indices, **kwargs)

    def decode(self, outputs: Any, labels: Any | None = None):
        mask_logits = self._unpack_mask_logits(outputs)
        probabilities = mask_logits.sigmoid()
        if labels is None:
            return probabilities[:, 0] if self.class_agnostic else probabilities
        return self._select_class_masks(probabilities, labels)

    def loss(self, outputs: Any, targets: Any, *, labels: Any | None = None) -> dict[str, Any]:
        mask_logits = self._unpack_mask_logits(outputs)
        mask_targets = self._target_tensor(
            targets,
            "mask_targets",
            dtype=mask_logits.dtype,
            device=mask_logits.device,
        )
        if mask_targets.dim() != 3:
            raise ValueError("mask_targets must have shape (N, H, W).")
        positive_indices = self._target_value(targets, "positive_indices", default=None)
        if positive_indices is None:
            selected_logits = mask_logits
            selected_labels = self._labels_for_selected_rows(
                labels,
                None,
                target_count=int(mask_targets.shape[0]),
                prediction_count=int(mask_logits.shape[0]),
                device=mask_logits.device,
            )
        else:
            positive_indices = torch.as_tensor(positive_indices, dtype=torch.long, device=mask_logits.device).reshape(-1)
            selected_logits = mask_logits.index_select(0, positive_indices)
            selected_labels = self._labels_for_selected_rows(
                labels,
                positive_indices,
                target_count=int(mask_targets.shape[0]),
                prediction_count=int(mask_logits.shape[0]),
                device=mask_logits.device,
            )
        if int(mask_targets.shape[0]) != int(selected_logits.shape[0]):
            raise ValueError("mask_targets must match the selected positive ROI predictions.")
        if int(mask_targets.shape[0]) == 0:
            zero = mask_logits.sum() * 0.0
            return {"loss_mask": zero, "loss_total": zero}

        selected = self._select_class_mask_logits(selected_logits, selected_labels)
        if tuple(mask_targets.shape[-2:]) != tuple(selected.shape[-2:]):
            mask_targets = F.interpolate(
                mask_targets.unsqueeze(1),
                size=tuple(selected.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        loss_mask = F.binary_cross_entropy_with_logits(selected, mask_targets, reduction="mean") * self.loss_weight
        return {"loss_mask": loss_mask, "loss_total": loss_mask}

    def _as_roi_tensor(self, roi_features: Any):
        feature = _as_4d_roi_tensor(roi_features, in_channels=self.in_channels, head_name=self.__class__.__name__)
        actual_h, actual_w = int(feature.shape[-2]), int(feature.shape[-1])
        if (actual_h, actual_w) != self.roi_feat_size:
            raise ValueError(
                "ROI mask head expected pooled feature size "
                f"{self.roi_feat_size}, got {(actual_h, actual_w)}."
            )
        return feature

    def _unpack_mask_logits(self, outputs: Any):
        if isinstance(outputs, dict):
            return outputs["mask_logits"]
        return torch.as_tensor(outputs)

    def _target_value(self, targets: Any, name: str, *, default: Any | None = None):
        if isinstance(targets, dict):
            return targets.get(name, default)
        return getattr(targets, name, default)

    def _target_tensor(
        self,
        targets: Any,
        name: str,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ):
        value = self._target_value(targets, name, default=None)
        if value is None:
            raise ValueError(f"ROI mask head loss requires targets.{name}.")
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _labels_for_selected_rows(
        self,
        labels: Any | None,
        positive_indices: Any | None,
        *,
        target_count: int,
        prediction_count: int,
        device: Any,
    ):
        if labels is None:
            return None
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
        if positive_indices is not None and int(label_tensor.shape[0]) == int(prediction_count):
            return label_tensor.index_select(0, positive_indices)
        if int(label_tensor.shape[0]) == int(target_count):
            return label_tensor
        raise ValueError("labels must match selected mask targets or all ROI predictions.")

    def _select_class_masks(self, probabilities: Any, labels: Any):
        if self.class_agnostic:
            return probabilities[:, 0]
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=probabilities.device).reshape(-1)
        if int(label_tensor.shape[0]) != int(probabilities.shape[0]):
            raise ValueError("labels must match the number of mask predictions.")
        channels = (label_tensor - 1).clamp(min=0, max=self.num_classes - 1)
        rows = torch.arange(probabilities.shape[0], dtype=torch.long, device=probabilities.device)
        return probabilities[rows, channels]

    def _select_class_mask_logits(self, mask_logits: Any, labels: Any | None):
        if self.class_agnostic:
            return mask_logits[:, 0]
        if labels is None:
            raise ValueError("class-specific mask loss requires one-based labels for positive ROIs.")
        return self._select_class_masks(mask_logits, labels)


@HEADS.register(
    "CascadeMaskHead",
    aliases=("cascade_mask_head", "cascade_mask"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=(*_ROI_MASK_CONTRACTS, "cascade_mask_stages"),
    validation_status="runtime_validated",
    family="roi",
    summary="Native Cascade Mask R-CNN ROI mask head.",
)
class CascadeMaskHead(FCNMaskHead):
    """Cascade-aware FCN mask head with stage-weighted mask loss."""

    def __init__(self, *, stage_loss_weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(loss_weight=float(stage_loss_weight), **kwargs)
        self.stage_loss_weight = float(stage_loss_weight)


@HEADS.register(
    "GridHead",
    aliases=("grid_head", "grid_roi_head"),
    required_dependencies=_ROI_BBOX_DEPENDENCIES,
    tensor_contracts=_ROI_GRID_CONTRACTS,
    validation_status="runtime_validated",
    family="roi",
    summary="Native Grid R-CNN ROI grid head with grid target and decode helpers.",
)
class GridHead(nn.Module):
    """Grid R-CNN style ROI head that predicts per-point heatmaps."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        grid_size: int | None = None,
        roi_feat_size: int | tuple[int, int] = 14,
        conv_out_channels: int = 256,
        num_convs: int = 4,
        upsample_scale: int = 2,
        output_size: int | tuple[int, int] | None = None,
        loss_weight: float = 1.0,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if grid_size is None:
            raise ValueError("GridHead requires `grid_size`; pass grid_size=<positive integer>.")
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.grid_size = _positive_int(grid_size, "grid_size")
        self.roi_feat_size = _roi_feat_size(roi_feat_size)
        self.conv_out_channels = _positive_int(conv_out_channels, "conv_out_channels")
        self.num_convs = _non_negative_int(num_convs, "num_convs")
        self.upsample_scale = _positive_int(upsample_scale, "upsample_scale")
        self.output_size = _roi_feat_size(
            output_size
            if output_size is not None
            else (
                self.roi_feat_size[0] * self.upsample_scale,
                self.roi_feat_size[1] * self.upsample_scale,
            )
        )
        self.loss_weight = float(loss_weight)
        self.num_grid_points = self.grid_size * self.grid_size

        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for _ in range(self.num_convs):
            layers.append(nn.Conv2d(current_channels, self.conv_out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = self.conv_out_channels
        self.convs = nn.Sequential(*layers)
        if self.upsample_scale > 1:
            self.upsample = nn.ConvTranspose2d(
                current_channels,
                self.conv_out_channels,
                kernel_size=self.upsample_scale,
                stride=self.upsample_scale,
            )
            current_channels = self.conv_out_channels
        else:
            self.upsample = nn.Identity()
        self.conv_logits = nn.Conv2d(current_channels, self.num_grid_points, kernel_size=1)

    def forward(self, roi_features: Any) -> dict[str, Any]:
        feature = self._as_roi_tensor(roi_features)
        if int(feature.shape[0]) == 0:
            grid_logits = feature.new_zeros((0, self.num_grid_points, *self.output_size))
            return {"grid_logits": grid_logits, "grid_pred": grid_logits}
        encoded = self.convs(feature)
        encoded = self.upsample(encoded)
        if self.upsample_scale > 1:
            encoded = F.relu(encoded, inplace=True)
        if tuple(encoded.shape[-2:]) != self.output_size:
            encoded = F.interpolate(encoded, size=self.output_size, mode="bilinear", align_corners=False)
        grid_logits = self.conv_logits(encoded)
        return {"grid_logits": grid_logits, "grid_pred": grid_logits}

    def get_targets(self, proposals: Any, matched_boxes: Any, **kwargs: Any):
        from .roi import build_roi_grid_targets

        kwargs.setdefault("grid_size", self.grid_size)
        return build_roi_grid_targets(proposals, matched_boxes, **kwargs)

    def decode(self, outputs: Any, proposals: Any | None = None) -> dict[str, Any]:
        grid_logits = self._unpack_grid_logits(outputs)
        if int(grid_logits.shape[1]) != self.num_grid_points:
            raise ValueError("grid_logits channel count must equal grid_size * grid_size.")
        probabilities = grid_logits.sigmoid()
        height, width = int(probabilities.shape[-2]), int(probabilities.shape[-1])
        scores, indices = probabilities.flatten(2).max(dim=2)
        ys = (indices // width).to(dtype=probabilities.dtype)
        xs = (indices % width).to(dtype=probabilities.dtype)
        x_denominator = float(max(width - 1, 1))
        y_denominator = float(max(height - 1, 1))
        normalized = torch.stack((xs / x_denominator, ys / y_denominator), dim=-1)
        normalized = normalized.reshape(-1, self.grid_size, self.grid_size, 2)
        decoded: dict[str, Any] = {
            "normalized_offsets": normalized,
            "scores": scores.reshape(-1, self.grid_size, self.grid_size),
        }
        if proposals is not None:
            boxes = torch.as_tensor(proposals, dtype=probabilities.dtype, device=probabilities.device).reshape(-1, 4)
            if int(boxes.shape[0]) != int(normalized.shape[0]):
                raise ValueError("proposals must match the number of grid predictions.")
            widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(torch.finfo(probabilities.dtype).eps)
            heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(torch.finfo(probabilities.dtype).eps)
            decoded["points"] = torch.stack(
                (
                    boxes[:, None, None, 0] + normalized[..., 0] * widths[:, None, None],
                    boxes[:, None, None, 1] + normalized[..., 1] * heights[:, None, None],
                ),
                dim=-1,
            )
        return decoded

    def loss(self, outputs: Any, targets: Any) -> dict[str, Any]:
        grid_logits = self._unpack_grid_logits(outputs)
        normalized_offsets = self._target_tensor(
            targets,
            "normalized_offsets",
            dtype=grid_logits.dtype,
            device=grid_logits.device,
        )
        weights = self._target_tensor(
            targets,
            "weights",
            dtype=grid_logits.dtype,
            device=grid_logits.device,
        )
        expected_shape = (int(grid_logits.shape[0]), self.grid_size, self.grid_size)
        if tuple(normalized_offsets.shape[:3]) != expected_shape or int(normalized_offsets.shape[-1]) != 2:
            raise ValueError("normalized_offsets must have shape (N, grid_size, grid_size, 2).")
        if tuple(weights.shape) != expected_shape:
            raise ValueError("weights must have shape (N, grid_size, grid_size).")
        if int(grid_logits.shape[0]) == 0:
            zero = grid_logits.sum() * 0.0
            return {"loss_grid": zero, "loss_total": zero}

        height, width = int(grid_logits.shape[-2]), int(grid_logits.shape[-1])
        flat_offsets = normalized_offsets.reshape(grid_logits.shape[0], self.num_grid_points, 2).clamp(0.0, 1.0)
        xs = torch.round(flat_offsets[..., 0] * float(max(width - 1, 1))).to(dtype=torch.long).clamp(0, width - 1)
        ys = torch.round(flat_offsets[..., 1] * float(max(height - 1, 1))).to(dtype=torch.long).clamp(0, height - 1)
        target_heatmaps = grid_logits.new_zeros(grid_logits.shape)
        rows = torch.arange(grid_logits.shape[0], dtype=torch.long, device=grid_logits.device)[:, None].expand(-1, self.num_grid_points)
        points = torch.arange(self.num_grid_points, dtype=torch.long, device=grid_logits.device)[None, :].expand_as(rows)
        target_heatmaps[rows.reshape(-1), points.reshape(-1), ys.reshape(-1), xs.reshape(-1)] = 1.0
        point_weights = weights.reshape(grid_logits.shape[0], self.num_grid_points).view(
            grid_logits.shape[0],
            self.num_grid_points,
            1,
            1,
        )
        loss_grid = F.binary_cross_entropy_with_logits(grid_logits, target_heatmaps, reduction="none")
        loss_grid = (loss_grid * point_weights).sum() / point_weights.sum().clamp_min(1.0)
        loss_grid = loss_grid * self.loss_weight
        return {"loss_grid": loss_grid, "loss_total": loss_grid}

    def _as_roi_tensor(self, roi_features: Any):
        feature = _as_4d_roi_tensor(roi_features, in_channels=self.in_channels, head_name=self.__class__.__name__)
        actual_h, actual_w = int(feature.shape[-2]), int(feature.shape[-1])
        if (actual_h, actual_w) != self.roi_feat_size:
            raise ValueError(
                "GridHead expected pooled feature size "
                f"{self.roi_feat_size}, got {(actual_h, actual_w)}."
            )
        return feature

    def _unpack_grid_logits(self, outputs: Any):
        if isinstance(outputs, dict):
            return outputs["grid_logits"]
        return torch.as_tensor(outputs)

    def _target_value(self, targets: Any, name: str, *, default: Any | None = None):
        if isinstance(targets, dict):
            return targets.get(name, default)
        return getattr(targets, name, default)

    def _target_tensor(
        self,
        targets: Any,
        name: str,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ):
        value = self._target_value(targets, name, default=None)
        if value is None:
            raise ValueError(f"GridHead loss requires targets.{name}.")
        return torch.as_tensor(value, dtype=dtype, device=device)


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
    if normalized.startswith("fcnmask"):
        return "FCNMaskHead"
    if normalized.startswith("cascademask"):
        return "CascadeMaskHead"
    if normalized in {"grid", "gridhead", "gridroihead"}:
        return "GridHead"
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
