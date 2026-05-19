"""Native detection losses registered for shared training components."""

from __future__ import annotations

import math
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import LOSSES

require_dependency("torch", "native losses")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class LossContractError(ValueError):
    """Raised when a loss receives tensors that violate its shape contract."""


class _LossBase(nn.Module):
    def __init__(
        self,
        *,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum.")
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def _finish(self, loss, *, weight=None, reduction: str | None = None, avg_factor=None, name: str):
        weighted = _apply_weight(loss, weight, name=name)
        reduced = _reduce_loss(weighted, reduction or self.reduction, avg_factor=avg_factor)
        return reduced * self.loss_weight


@LOSSES.register(
    "FocalLoss",
    aliases=("focal", "focal_loss"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="classification",
    summary="Binary sigmoid focal loss for dense classification logits.",
)
class FocalLoss(_LossBase):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction = _as_float_tensor(prediction, name="prediction")
        target = _as_float_tensor(target, name="target", like=prediction)
        _ensure_same_shape(prediction, target, name="FocalLoss")

        ce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        pred_sigmoid = prediction.sigmoid()
        p_t = pred_sigmoid * target + (1.0 - pred_sigmoid) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = ce_loss * alpha_t * (1.0 - p_t).clamp(min=0.0).pow(self.gamma)
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="FocalLoss")


@LOSSES.register(
    "QualityFocalLoss",
    aliases=("quality_focal", "quality_focal_loss", "qfl"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="classification",
    summary="Quality focal loss for dense class logits with quality-score targets.",
)
class QualityFocalLoss(_LossBase):
    def __init__(
        self,
        *,
        beta: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.beta = float(beta)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction = _as_float_tensor(prediction, name="prediction")
        target = _as_float_tensor(target, name="target", like=prediction)
        _ensure_same_shape(prediction, target, name="QualityFocalLoss")

        pred_sigmoid = prediction.sigmoid()
        scale = (target - pred_sigmoid).abs().pow(self.beta)
        loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * scale
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="QualityFocalLoss")


@LOSSES.register(
    "VarifocalLoss",
    aliases=("varifocal", "varifocal_loss", "vfl"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="classification",
    summary="VFNet-style varifocal loss for class logits with IoU-aware targets.",
)
class VarifocalLoss(_LossBase):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float = 0.75,
        iou_weighted: bool = True,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.iou_weighted = bool(iou_weighted)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction = _as_float_tensor(prediction, name="prediction")
        target = _as_float_tensor(target, name="target", like=prediction)
        _ensure_same_shape(prediction, target, name="VarifocalLoss")

        pred_sigmoid = prediction.sigmoid()
        negative_weight = self.alpha * pred_sigmoid.pow(self.gamma) * (1.0 - target)
        positive_weight = target if self.iou_weighted else (target > 0).to(dtype=prediction.dtype)
        focal_weight = negative_weight + positive_weight
        loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * focal_weight
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="VarifocalLoss")


@LOSSES.register(
    "DistributionFocalLoss",
    aliases=(
        "distribution_focal",
        "distribution_focal_loss",
        "dfl",
        "generalized_focal_distribution",
    ),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_bins_target_values",),
    validation_status="runtime_validated",
    family="bbox",
    summary="Distribution focal loss used by generalized focal dense heads.",
)
class DistributionFocalLoss(_LossBase):
    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction = _as_float_tensor(prediction, name="prediction")
        target = _as_float_tensor(target, name="target", like=prediction)
        if prediction.dim() < 2 or prediction.shape[-1] < 2:
            raise LossContractError(
                "DistributionFocalLoss expects prediction shape (..., num_bins) with num_bins >= 2; "
                f"got prediction shape {_shape_text(prediction)}."
            )
        expected_target_shape = tuple(prediction.shape[:-1])
        if tuple(target.shape) != expected_target_shape:
            raise LossContractError(
                "DistributionFocalLoss expects target shape to match prediction.shape[:-1]; "
                f"got prediction shape {_shape_text(prediction)} and target shape {_shape_text(target)}."
            )
        if target.numel() and (bool((target < 0).any()) or bool((target > prediction.shape[-1] - 1).any())):
            raise LossContractError(
                "DistributionFocalLoss targets must be in [0, num_bins - 1]; "
                f"got target range [{float(target.min())}, {float(target.max())}] for {prediction.shape[-1]} bins."
            )

        bins = int(prediction.shape[-1])
        flat_prediction = prediction.reshape(-1, bins)
        flat_target = target.reshape(-1)
        left = flat_target.floor().long()
        right = (left + 1).clamp(max=bins - 1)
        weight_right = flat_target - left.to(dtype=flat_target.dtype)
        weight_left = 1.0 - weight_right
        loss_left = F.cross_entropy(flat_prediction, left, reduction="none") * weight_left
        loss_right = F.cross_entropy(flat_prediction, right, reduction="none") * weight_right
        loss = (loss_left + loss_right).reshape(expected_target_shape)
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="DistributionFocalLoss")


@LOSSES.register(
    "SmoothL1Loss",
    aliases=("smooth_l1", "smooth_l1_loss"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="bbox",
)
class SmoothL1Loss(_LossBase):
    def __init__(
        self,
        *,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.beta = float(beta)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction, target = _same_shape_pair(prediction, target, name="SmoothL1Loss")
        loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.beta)
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="SmoothL1Loss")


@LOSSES.register(
    "L1Loss",
    aliases=("l1", "l1_loss"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="bbox",
)
class L1Loss(_LossBase):
    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction, target = _same_shape_pair(prediction, target, name="L1Loss")
        loss = F.l1_loss(prediction, target, reduction="none")
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="L1Loss")


@LOSSES.register(
    "IoULoss",
    aliases=("iou", "iou_loss"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("box_pair_same_shape",),
    validation_status="runtime_validated",
    family="bbox",
    summary="Axis-aligned xyxy IoU loss.",
)
class IoULoss(_LossBase):
    mode = "iou"

    def __init__(
        self,
        *,
        mode: str | None = None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.mode = str(mode or self.mode).lower()
        if self.mode not in {"iou", "giou", "diou", "ciou"}:
            raise ValueError("IoULoss mode must be one of: iou, giou, diou, ciou.")
        self.eps = float(eps)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction = _as_float_tensor(prediction, name="prediction")
        target = _as_float_tensor(target, name="target", like=prediction)
        _ensure_same_shape(prediction, target, name=type(self).__name__)
        _ensure_box_shape(prediction, name=type(self).__name__, tensor_name="prediction")

        original_shape = tuple(prediction.shape[:-1])
        pred_boxes = prediction.reshape(-1, 4)
        target_boxes = target.reshape(-1, 4)
        metric = _aligned_box_metric(pred_boxes, target_boxes, mode=self.mode, eps=self.eps)
        loss = (1.0 - metric).reshape(original_shape)
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name=type(self).__name__)


@LOSSES.register(
    "GIoULoss",
    aliases=("giou", "giou_loss", "generalized_iou"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("box_pair_same_shape",),
    validation_status="runtime_validated",
    family="bbox",
)
class GIoULoss(IoULoss):
    mode = "giou"


@LOSSES.register(
    "DIoULoss",
    aliases=("diou", "diou_loss", "distance_iou"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("box_pair_same_shape",),
    validation_status="runtime_validated",
    family="bbox",
)
class DIoULoss(IoULoss):
    mode = "diou"


@LOSSES.register(
    "CIoULoss",
    aliases=("ciou", "ciou_loss", "complete_iou"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("box_pair_same_shape",),
    validation_status="runtime_validated",
    family="bbox",
)
class CIoULoss(IoULoss):
    mode = "ciou"


@LOSSES.register(
    "CrossEntropyLoss",
    aliases=("cross_entropy", "cross_entropy_loss", "ce"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("logits_class_targets",),
    validation_status="runtime_validated",
    family="classification",
)
class CrossEntropyLoss(_LossBase):
    def __init__(
        self,
        *,
        use_sigmoid: bool = False,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.use_sigmoid = bool(use_sigmoid)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction = _as_float_tensor(prediction, name="prediction")
        if self.use_sigmoid:
            target = _as_float_tensor(target, name="target", like=prediction)
            _ensure_same_shape(prediction, target, name="CrossEntropyLoss")
            loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
            return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="CrossEntropyLoss")

        if _is_tensor(target):
            target_tensor = target.to(device=prediction.device)
        else:
            target_tensor = torch.as_tensor(target, device=prediction.device)
        if tuple(target_tensor.shape) == tuple(prediction.shape):
            target_tensor = target_tensor.to(dtype=prediction.dtype)
            loss = F.binary_cross_entropy_with_logits(prediction, target_tensor, reduction="none")
            return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="CrossEntropyLoss")

        if prediction.dim() < 2:
            raise LossContractError(
                "CrossEntropyLoss expects prediction shape (N, C, ...) for class-index targets; "
                f"got prediction shape {_shape_text(prediction)}."
            )
        expected = (prediction.shape[0], *prediction.shape[2:])
        if tuple(target_tensor.shape) != expected:
            raise LossContractError(
                "CrossEntropyLoss expects target shape (N, ...) for class-index targets or the same "
                "shape as prediction for binary targets; "
                f"got prediction shape {_shape_text(prediction)} and target shape {_shape_text(target_tensor)}."
            )
        loss = F.cross_entropy(prediction, target_tensor.long(), reduction="none")
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="CrossEntropyLoss")


@LOSSES.register(
    "DiceLoss",
    aliases=("dice", "dice_loss"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="mask",
)
class DiceLoss(_LossBase):
    def __init__(
        self,
        *,
        eps: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.eps = float(eps)
        self.from_logits = bool(from_logits)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction, target = _same_shape_pair(prediction, target, name="DiceLoss")
        probability = prediction.sigmoid() if self.from_logits else prediction
        flat_prediction, flat_target = _flatten_mask_pair(probability, target)
        intersection = (flat_prediction * flat_target).sum(dim=1)
        denominator = flat_prediction.sum(dim=1) + flat_target.sum(dim=1)
        loss = 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="DiceLoss")


@LOSSES.register(
    "MaskLoss",
    aliases=("mask", "mask_loss", "binary_mask", "binary_mask_loss"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("prediction_target_same_shape",),
    validation_status="runtime_validated",
    family="mask",
)
class MaskLoss(_LossBase):
    def __init__(
        self,
        *,
        from_logits: bool = True,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        self.from_logits = bool(from_logits)

    def forward(self, prediction, target, *, weight=None, avg_factor=None, reduction: str | None = None):
        prediction, target = _same_shape_pair(prediction, target, name="MaskLoss")
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        else:
            loss = F.binary_cross_entropy(prediction, target, reduction="none")
        return self._finish(loss, weight=weight, reduction=reduction, avg_factor=avg_factor, name="MaskLoss")


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Build a registered native loss by name or alias."""

    LOSSES.import_modules("simpledet.native.losses")
    return LOSSES.get(name)(**kwargs)


def _same_shape_pair(prediction, target, *, name: str):
    prediction = _as_float_tensor(prediction, name="prediction")
    target = _as_float_tensor(target, name="target", like=prediction)
    _ensure_same_shape(prediction, target, name=name)
    return prediction, target


def _as_float_tensor(value, *, name: str, like=None):
    if _is_tensor(value):
        tensor = value
    else:
        kwargs = {}
        if like is not None:
            kwargs.update({"device": like.device, "dtype": like.dtype})
        tensor = torch.as_tensor(value, **kwargs)
    if like is not None:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    elif not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor


def _is_tensor(value) -> bool:
    return torch.is_tensor(value)


def _ensure_same_shape(prediction, target, *, name: str) -> None:
    if tuple(prediction.shape) != tuple(target.shape):
        raise LossContractError(
            f"{name} expects prediction and target tensors with identical shapes; "
            f"got prediction shape {_shape_text(prediction)} and target shape {_shape_text(target)}."
        )


def _ensure_box_shape(tensor, *, name: str, tensor_name: str) -> None:
    if tensor.dim() == 0 or tensor.shape[-1] != 4:
        raise LossContractError(
            f"{name} expects {tensor_name} boxes with shape (..., 4); "
            f"got {tensor_name} shape {_shape_text(tensor)}."
        )


def _apply_weight(loss, weight, *, name: str):
    if weight is None:
        return loss
    weight_tensor = _as_float_tensor(weight, name="weight", like=loss)
    try:
        return loss * weight_tensor
    except RuntimeError as exc:
        raise LossContractError(
            f"{name} weight shape {_shape_text(weight_tensor)} is not broadcastable to "
            f"loss shape {_shape_text(loss)}."
        ) from exc


def _reduce_loss(loss, reduction: str, *, avg_factor=None):
    if reduction == "none":
        return loss
    if avg_factor is not None:
        factor = float(avg_factor)
        if factor <= 0:
            return loss.sum() * 0.0
        return loss.sum() / factor
    if reduction == "sum":
        return loss.sum()
    if loss.numel() == 0:
        return loss.sum() * 0.0
    return loss.mean()


def _flatten_mask_pair(prediction, target):
    if prediction.dim() == 0:
        raise LossContractError("Mask-style losses expect at least one tensor dimension.")
    if prediction.dim() == 1:
        return prediction.reshape(1, -1), target.reshape(1, -1)
    return prediction.reshape(prediction.shape[0], -1), target.reshape(target.shape[0], -1)


def _aligned_box_metric(prediction, target, *, mode: str, eps: float):
    if prediction.numel() == 0:
        return prediction.new_zeros((prediction.shape[0],))

    pred_x1, pred_y1, pred_x2, pred_y2 = prediction.unbind(dim=1)
    target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=1)
    pred_w = (pred_x2 - pred_x1).clamp(min=0.0)
    pred_h = (pred_y2 - pred_y1).clamp(min=0.0)
    target_w = (target_x2 - target_x1).clamp(min=0.0)
    target_h = (target_y2 - target_y1).clamp(min=0.0)
    pred_area = pred_w * pred_h
    target_area = target_w * target_h

    inter_w = (torch.minimum(pred_x2, target_x2) - torch.maximum(pred_x1, target_x1)).clamp(min=0.0)
    inter_h = (torch.minimum(pred_y2, target_y2) - torch.maximum(pred_y1, target_y1)).clamp(min=0.0)
    intersection = inter_w * inter_h
    union = (pred_area + target_area - intersection).clamp(min=eps)
    iou = intersection / union
    if mode == "iou":
        return iou

    enclose_x1 = torch.minimum(pred_x1, target_x1)
    enclose_y1 = torch.minimum(pred_y1, target_y1)
    enclose_x2 = torch.maximum(pred_x2, target_x2)
    enclose_y2 = torch.maximum(pred_y2, target_y2)
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0.0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0.0)
    if mode == "giou":
        enclose_area = (enclose_w * enclose_h).clamp(min=eps)
        return iou - (enclose_area - union) / enclose_area

    pred_ctr_x = (pred_x1 + pred_x2) * 0.5
    pred_ctr_y = (pred_y1 + pred_y2) * 0.5
    target_ctr_x = (target_x1 + target_x2) * 0.5
    target_ctr_y = (target_y1 + target_y2) * 0.5
    center_distance = (pred_ctr_x - target_ctr_x).pow(2) + (pred_ctr_y - target_ctr_y).pow(2)
    diagonal = enclose_w.pow(2) + enclose_h.pow(2)
    distance_penalty = center_distance / diagonal.clamp(min=eps)
    if mode == "diou":
        return iou - distance_penalty

    pred_w = pred_w.clamp(min=eps)
    pred_h = pred_h.clamp(min=eps)
    target_w = target_w.clamp(min=eps)
    target_h = target_h.clamp(min=eps)
    aspect = (4.0 / math.pi**2) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)).pow(2)
    with torch.no_grad():
        alpha = aspect / (1.0 - iou + aspect).clamp(min=eps)
    return iou - distance_penalty - alpha * aspect


def _shape_text(tensor) -> str:
    return str(tuple(tensor.shape))


__all__ = [
    "LossContractError",
    "FocalLoss",
    "QualityFocalLoss",
    "VarifocalLoss",
    "DistributionFocalLoss",
    "SmoothL1Loss",
    "L1Loss",
    "IoULoss",
    "GIoULoss",
    "DIoULoss",
    "CIoULoss",
    "CrossEntropyLoss",
    "DiceLoss",
    "MaskLoss",
    "build_loss",
]
