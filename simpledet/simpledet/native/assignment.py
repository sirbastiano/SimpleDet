"""Native target assignment and sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency
from ..extensions import ASSIGNERS

require_dependency("torch", "native assignment")
import torch  # noqa: E402

from .geometry import box_iou  # noqa: E402


BACKGROUND_LABEL = 0
IGNORE_LABEL = -1


@dataclass(slots=True)
class AssignmentResult:
    """Per-prior assignment state using one-based foreground labels."""

    assigned_gt_indices: Any
    labels: Any
    max_overlaps: Any
    matched_boxes: Any

    @property
    def positive_mask(self):
        return self.assigned_gt_indices > 0

    @property
    def negative_mask(self):
        return self.assigned_gt_indices == 0

    @property
    def ignored_mask(self):
        return self.assigned_gt_indices < 0


@dataclass(slots=True)
class SamplingResult:
    """Deterministic positive/negative samples from an assignment result."""

    positive_indices: Any
    negative_indices: Any
    ignored_indices: Any


@ASSIGNERS.register(
    "MaxIoUAssigner",
    aliases=("max_iou",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("xyxy_priors", "xyxy_gt_boxes", "one_based_labels"),
    validation_status="runtime_validated",
    family="dense",
    summary="IoU-threshold assigner with low-quality ground-truth matching.",
)
@dataclass(slots=True)
class MaxIoUAssigner:
    pos_iou_thr: float = 0.5
    neg_iou_thr: float = 0.4
    min_pos_iou: float = 0.0
    match_low_quality: bool = True
    ignore_iou_thr: float = 0.5

    def __call__(self, priors, gt_boxes, gt_labels, *, ignored_boxes=None) -> AssignmentResult:
        return max_iou_assign(
            priors,
            gt_boxes,
            gt_labels,
            pos_iou_thr=self.pos_iou_thr,
            neg_iou_thr=self.neg_iou_thr,
            min_pos_iou=self.min_pos_iou,
            match_low_quality=self.match_low_quality,
            ignored_boxes=ignored_boxes,
            ignore_iou_thr=self.ignore_iou_thr,
        )


@ASSIGNERS.register(
    "ATSSAssigner",
    aliases=("atss",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("feature_pyramid_priors", "xyxy_gt_boxes", "one_based_labels"),
    validation_status="runtime_validated",
    family="dense",
    summary="Adaptive Training Sample Selection for multilevel anchor priors.",
)
@dataclass(slots=True)
class ATSSAssigner:
    topk: int = 9
    ignore_iou_thr: float = 0.5

    def __call__(
        self,
        priors,
        gt_boxes,
        gt_labels,
        *,
        num_level_priors,
        ignored_boxes=None,
    ) -> AssignmentResult:
        return atss_assign(
            priors,
            gt_boxes,
            gt_labels,
            num_level_priors=num_level_priors,
            topk=self.topk,
            ignored_boxes=ignored_boxes,
            ignore_iou_thr=self.ignore_iou_thr,
        )


@ASSIGNERS.register(
    "CenterRegionAssigner",
    aliases=("center_region",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("point_priors", "xyxy_gt_boxes", "one_based_labels"),
    validation_status="runtime_validated",
    family="dense",
    summary="Anchor-free center-region assignment for FCOS-like heads.",
)
@dataclass(slots=True)
class CenterRegionAssigner:
    center_radius: float = 1.5

    def __call__(self, points, gt_boxes, gt_labels, *, strides=None) -> AssignmentResult:
        return center_region_assign(
            points,
            gt_boxes,
            gt_labels,
            center_radius=self.center_radius,
            strides=strides,
        )


@ASSIGNERS.register(
    "PointAssigner",
    aliases=("point", "point_based"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("point_priors", "xyxy_gt_boxes", "one_based_labels"),
    validation_status="runtime_validated",
    family="dense",
    summary="Smallest-covering-box assignment for point priors.",
)
class PointAssigner:
    def __call__(self, points, gt_boxes, gt_labels) -> AssignmentResult:
        return point_assign(points, gt_boxes, gt_labels)


@ASSIGNERS.register(
    "TaskAlignedAssigner",
    aliases=("task_aligned", "tal"),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("xyxy_priors", "predicted_scores", "predicted_boxes"),
    validation_status="runtime_validated",
    family="dense",
    summary="TOOD-style task-aligned assignment using class confidence and IoU.",
)
@dataclass(slots=True)
class TaskAlignedAssigner:
    topk: int = 13
    alpha: float = 1.0
    beta: float = 6.0

    def __call__(self, priors, pred_scores, pred_boxes, gt_boxes, gt_labels) -> AssignmentResult:
        return task_aligned_assign(
            priors,
            pred_scores,
            pred_boxes,
            gt_boxes,
            gt_labels,
            topk=self.topk,
            alpha=self.alpha,
            beta=self.beta,
        )


@ASSIGNERS.register(
    "SimOTAAssigner",
    aliases=("sim_ota",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("xyxy_priors", "predicted_scores", "predicted_boxes"),
    validation_status="runtime_validated",
    family="dense",
    summary="YOLOX-style dynamic-k simOTA assignment.",
)
@dataclass(slots=True)
class SimOTAAssigner:
    center_radius: float = 2.5
    candidate_topk: int = 10

    def __call__(self, priors, pred_scores, pred_boxes, gt_boxes, gt_labels) -> AssignmentResult:
        return sim_ota_assign(
            priors,
            pred_scores,
            pred_boxes,
            gt_boxes,
            gt_labels,
            center_radius=self.center_radius,
            candidate_topk=self.candidate_topk,
        )


@ASSIGNERS.register(
    "HungarianAssigner",
    aliases=("hungarian",),
    required_dependencies=(("torch", "cpu"),),
    tensor_contracts=("set_predictions", "xyxy_gt_boxes", "one_based_labels"),
    validation_status="runtime_validated",
    family="transformer",
    summary="DETR-style one-to-one assignment with an exact Hungarian solver.",
)
@dataclass(slots=True)
class HungarianAssigner:
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    iou_cost: float = 2.0

    def __call__(self, pred_logits, pred_boxes, gt_boxes, gt_labels) -> AssignmentResult:
        return hungarian_assign(
            pred_logits,
            pred_boxes,
            gt_boxes,
            gt_labels,
            class_cost=self.class_cost,
            bbox_cost=self.bbox_cost,
            iou_cost=self.iou_cost,
        )


@dataclass(slots=True)
class BalancedSampler:
    """Deterministic sampler for positive and negative assignment indices."""

    num_samples: int
    positive_fraction: float = 0.5

    def __call__(self, assignment: AssignmentResult) -> SamplingResult:
        return sample_assignment(
            assignment,
            num_samples=self.num_samples,
            positive_fraction=self.positive_fraction,
        )


def max_iou_assign(
    priors,
    gt_boxes,
    gt_labels,
    *,
    pos_iou_thr: float = 0.5,
    neg_iou_thr: float = 0.4,
    min_pos_iou: float = 0.0,
    match_low_quality: bool = True,
    ignored_boxes=None,
    ignore_iou_thr: float = 0.5,
) -> AssignmentResult:
    priors = _as_boxes(priors, name="priors")
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=priors.device, dtype=priors.dtype)
    gt_labels = _as_labels(gt_labels, device=priors.device)
    _validate_gt(gt_boxes, gt_labels)
    result = _empty_assignment(priors, gt_boxes)
    if priors.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return _apply_ignored_boxes(result, priors, ignored_boxes, ignore_iou_thr)

    overlaps = box_iou(priors, gt_boxes)
    max_overlaps, argmax_overlaps = overlaps.max(dim=1)
    assigned = torch.full((priors.shape[0],), IGNORE_LABEL, dtype=torch.long, device=priors.device)
    assigned[max_overlaps < float(neg_iou_thr)] = 0
    assigned[max_overlaps >= float(pos_iou_thr)] = argmax_overlaps[max_overlaps >= float(pos_iou_thr)] + 1

    if match_low_quality:
        gt_max_overlaps, gt_argmax_priors = overlaps.max(dim=0)
        for gt_index, prior_index in enumerate(gt_argmax_priors.tolist()):
            if float(gt_max_overlaps[gt_index]) >= float(min_pos_iou):
                assigned[prior_index] = gt_index + 1
                max_overlaps[prior_index] = gt_max_overlaps[gt_index]

    result = _result_from_assigned(priors, gt_boxes, gt_labels, assigned, max_overlaps, argmax_overlaps)
    return _apply_ignored_boxes(result, priors, ignored_boxes, ignore_iou_thr)


def atss_assign(
    priors,
    gt_boxes,
    gt_labels,
    *,
    num_level_priors,
    topk: int = 9,
    ignored_boxes=None,
    ignore_iou_thr: float = 0.5,
) -> AssignmentResult:
    priors = _as_boxes(priors, name="priors")
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=priors.device, dtype=priors.dtype)
    gt_labels = _as_labels(gt_labels, device=priors.device)
    _validate_gt(gt_boxes, gt_labels)
    level_counts = _normalize_level_counts(num_level_priors, priors.shape[0])
    result = _empty_assignment(priors, gt_boxes)
    if priors.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return _apply_ignored_boxes(result, priors, ignored_boxes, ignore_iou_thr)

    overlaps = box_iou(priors, gt_boxes)
    prior_centers = _box_centers(priors)
    gt_centers = _box_centers(gt_boxes)
    distances = ((prior_centers[:, None, :] - gt_centers[None, :, :]) ** 2).sum(dim=2).sqrt()
    candidate_mask = torch.zeros_like(overlaps, dtype=torch.bool)
    start = 0
    for count in level_counts:
        end = start + count
        level_distances = distances[start:end]
        level_topk = min(int(topk), count)
        if level_topk > 0:
            topk_indices = level_distances.topk(level_topk, dim=0, largest=False).indices + start
            candidate_mask[topk_indices, torch.arange(gt_boxes.shape[0], device=priors.device)] = True
        start = end

    candidate_overlaps = torch.where(candidate_mask, overlaps, overlaps.new_zeros(()))
    overlap_sums = candidate_overlaps.sum(dim=0)
    candidate_counts = candidate_mask.sum(dim=0).clamp(min=1)
    means = overlap_sums / candidate_counts
    variances = (
        torch.where(candidate_mask, (overlaps - means.unsqueeze(0)) ** 2, overlaps.new_zeros(()))
        .sum(dim=0)
        / candidate_counts
    )
    thresholds = means + variances.sqrt()
    centers_in_boxes = _points_in_boxes(prior_centers, gt_boxes)
    positive_candidates = candidate_mask & (overlaps >= thresholds.unsqueeze(0)) & centers_in_boxes
    assigned, max_overlaps, argmax_overlaps = _assign_candidates(overlaps, positive_candidates)
    result = _result_from_assigned(priors, gt_boxes, gt_labels, assigned, max_overlaps, argmax_overlaps)
    return _apply_ignored_boxes(result, priors, ignored_boxes, ignore_iou_thr)


def center_region_assign(
    points,
    gt_boxes,
    gt_labels,
    *,
    center_radius: float = 1.5,
    strides=None,
) -> AssignmentResult:
    points = _as_points(points)
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=points.device, dtype=points.dtype)
    gt_labels = _as_labels(gt_labels, device=points.device)
    _validate_gt(gt_boxes, gt_labels)
    result = _empty_point_assignment(points, gt_boxes)
    if points.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return result

    in_boxes = _points_in_boxes(points, gt_boxes)
    if strides is None:
        in_center = _points_in_boxes(points, _center_boxes(gt_boxes, radius=center_radius))
    else:
        in_center = _points_in_center_regions(points, gt_boxes, strides=strides, radius=center_radius)
    return _assign_points_by_area(points, gt_boxes, gt_labels, in_boxes & in_center)


def point_assign(points, gt_boxes, gt_labels) -> AssignmentResult:
    points = _as_points(points)
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=points.device, dtype=points.dtype)
    gt_labels = _as_labels(gt_labels, device=points.device)
    _validate_gt(gt_boxes, gt_labels)
    if points.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return _empty_point_assignment(points, gt_boxes)
    return _assign_points_by_area(points, gt_boxes, gt_labels, _points_in_boxes(points, gt_boxes))


def task_aligned_assign(
    priors,
    pred_scores,
    pred_boxes,
    gt_boxes,
    gt_labels,
    *,
    topk: int = 13,
    alpha: float = 1.0,
    beta: float = 6.0,
) -> AssignmentResult:
    priors = _as_boxes(priors, name="priors")
    pred_boxes = _as_boxes(pred_boxes, name="pred_boxes", device=priors.device, dtype=priors.dtype)
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=priors.device, dtype=priors.dtype)
    gt_labels = _as_labels(gt_labels, device=priors.device)
    pred_scores = _as_scores(pred_scores, device=priors.device, dtype=priors.dtype)
    _validate_gt(gt_boxes, gt_labels)
    _validate_prediction_count(priors, pred_boxes, pred_scores)
    result = _empty_assignment(priors, gt_boxes)
    if priors.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return result

    overlaps = box_iou(pred_boxes, gt_boxes).clamp(min=0)
    label_scores = _scores_for_labels(pred_scores, gt_labels)
    alignment = label_scores.pow(float(alpha)) * overlaps.pow(float(beta))
    alignment = torch.where(_points_in_boxes(_box_centers(priors), gt_boxes), alignment, alignment.new_zeros(()))
    candidate_mask = _topk_mask(alignment, int(topk))
    assigned, max_overlaps, argmax_overlaps = _assign_candidates(alignment, candidate_mask)
    positive = assigned > 0
    if positive.any():
        assigned_gt = assigned[positive] - 1
        max_overlaps[positive] = overlaps[positive, assigned_gt]
        argmax_overlaps[positive] = assigned_gt
    return _result_from_assigned(priors, gt_boxes, gt_labels, assigned, max_overlaps, argmax_overlaps)


def sim_ota_assign(
    priors,
    pred_scores,
    pred_boxes,
    gt_boxes,
    gt_labels,
    *,
    center_radius: float = 2.5,
    candidate_topk: int = 10,
) -> AssignmentResult:
    priors = _as_boxes(priors, name="priors")
    pred_boxes = _as_boxes(pred_boxes, name="pred_boxes", device=priors.device, dtype=priors.dtype)
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=priors.device, dtype=priors.dtype)
    gt_labels = _as_labels(gt_labels, device=priors.device)
    pred_scores = _as_scores(pred_scores, device=priors.device, dtype=priors.dtype).clamp(min=1e-6, max=1 - 1e-6)
    _validate_gt(gt_boxes, gt_labels)
    _validate_prediction_count(priors, pred_boxes, pred_scores)
    result = _empty_assignment(priors, gt_boxes)
    if priors.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return result

    centers = _box_centers(priors)
    in_boxes = _points_in_boxes(centers, gt_boxes)
    in_centers = _points_in_boxes(centers, _center_boxes(gt_boxes, radius=center_radius))
    candidate_mask = in_boxes | in_centers
    overlaps = box_iou(pred_boxes, gt_boxes).clamp(min=0)
    cls_cost = -torch.log(_scores_for_labels(pred_scores, gt_labels))
    iou_cost = -torch.log(overlaps.clamp(min=1e-8))
    cost = cls_cost + 3.0 * iou_cost + torch.where(candidate_mask, overlaps.new_zeros(()), overlaps.new_full((), 1e6))
    matching = torch.zeros_like(candidate_mask, dtype=torch.bool)
    topk = min(int(candidate_topk), overlaps.shape[0])
    for gt_index in range(gt_boxes.shape[0]):
        if topk <= 0:
            continue
        dynamic_k = int(overlaps[:, gt_index].topk(topk, largest=True).values.sum().clamp(min=1).item())
        dynamic_k = min(dynamic_k, int(candidate_mask[:, gt_index].sum().item()))
        if dynamic_k <= 0:
            continue
        candidate_indices = torch.nonzero(candidate_mask[:, gt_index], as_tuple=False).reshape(-1)
        candidate_costs = cost[candidate_indices, gt_index]
        selected = candidate_indices[candidate_costs.topk(dynamic_k, largest=False).indices]
        matching[selected, gt_index] = True

    conflicts = matching.sum(dim=1) > 1
    if conflicts.any():
        best_gt = cost[conflicts].argmin(dim=1)
        matching[conflicts] = False
        matching[torch.nonzero(conflicts, as_tuple=False).reshape(-1), best_gt] = True

    assigned, max_overlaps, argmax_overlaps = _assign_candidates(overlaps, matching)
    return _result_from_assigned(priors, gt_boxes, gt_labels, assigned, max_overlaps, argmax_overlaps)


def hungarian_assign(
    pred_logits,
    pred_boxes,
    gt_boxes,
    gt_labels,
    *,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    iou_cost: float = 2.0,
) -> AssignmentResult:
    pred_boxes = _as_boxes(pred_boxes, name="pred_boxes")
    gt_boxes = _as_boxes(gt_boxes, name="gt_boxes", device=pred_boxes.device, dtype=pred_boxes.dtype)
    gt_labels = _as_labels(gt_labels, device=pred_boxes.device)
    pred_logits = _as_scores(pred_logits, device=pred_boxes.device, dtype=pred_boxes.dtype)
    _validate_gt(gt_boxes, gt_labels)
    if pred_boxes.shape[0] != pred_logits.shape[0]:
        raise ValueError("pred_logits and pred_boxes must have the same prediction count.")
    result = _empty_assignment(pred_boxes, gt_boxes)
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return result

    probabilities = pred_logits.softmax(dim=-1)
    cls_cost = -_scores_for_labels(probabilities, gt_labels)
    l1_cost = torch.cdist(pred_boxes, gt_boxes, p=1)
    overlap_cost = 1.0 - box_iou(pred_boxes, gt_boxes)
    cost = float(class_cost) * cls_cost + float(bbox_cost) * l1_cost + float(iou_cost) * overlap_cost
    pred_indices, gt_indices = _hungarian_indices(cost)
    assigned = torch.zeros((pred_boxes.shape[0],), dtype=torch.long, device=pred_boxes.device)
    max_overlaps = pred_boxes.new_zeros((pred_boxes.shape[0],))
    argmax_overlaps = torch.zeros((pred_boxes.shape[0],), dtype=torch.long, device=pred_boxes.device)
    if pred_indices.numel() > 0:
        assigned[pred_indices] = gt_indices + 1
        overlaps = box_iou(pred_boxes, gt_boxes)
        max_overlaps[pred_indices] = overlaps[pred_indices, gt_indices]
        argmax_overlaps[pred_indices] = gt_indices
    return _result_from_assigned(pred_boxes, gt_boxes, gt_labels, assigned, max_overlaps, argmax_overlaps)


def sample_assignment(
    assignment: AssignmentResult,
    *,
    num_samples: int,
    positive_fraction: float = 0.5,
) -> SamplingResult:
    if int(num_samples) < 0:
        raise ValueError("num_samples must be non-negative.")
    if not 0.0 <= float(positive_fraction) <= 1.0:
        raise ValueError("positive_fraction must be between 0 and 1.")
    positive = torch.nonzero(assignment.positive_mask, as_tuple=False).reshape(-1)
    negative = torch.nonzero(assignment.negative_mask, as_tuple=False).reshape(-1)
    ignored = torch.nonzero(assignment.ignored_mask, as_tuple=False).reshape(-1)
    max_positive = min(int(round(int(num_samples) * float(positive_fraction))), positive.numel())
    max_negative = min(int(num_samples) - max_positive, negative.numel())
    return SamplingResult(
        positive_indices=positive[:max_positive],
        negative_indices=negative[:max_negative],
        ignored_indices=ignored,
    )


def _as_boxes(value, *, name: str, device=None, dtype=None):
    dtype = dtype or torch.float32
    boxes = torch.as_tensor(value, dtype=dtype, device=device)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4).")
    return boxes


def _as_points(value):
    points = torch.as_tensor(value, dtype=getattr(value, "dtype", torch.float32), device=getattr(value, "device", None))
    if points.numel() == 0:
        return points.reshape(0, 2)
    if points.dim() != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    return points


def _as_labels(value, *, device):
    labels = torch.as_tensor(value, dtype=torch.long, device=device)
    if labels.numel() == 0:
        return labels.reshape(0)
    if labels.dim() != 1:
        raise ValueError("gt_labels must have shape (N,).")
    return labels


def _as_scores(value, *, device, dtype):
    scores = torch.as_tensor(value, dtype=dtype, device=device)
    if scores.numel() == 0:
        return scores.reshape(0, 0)
    if scores.dim() != 2:
        raise ValueError("prediction scores/logits must have shape (N, C).")
    return scores


def _validate_gt(gt_boxes, gt_labels) -> None:
    if gt_boxes.shape[0] != gt_labels.shape[0]:
        raise ValueError("gt_boxes and gt_labels must have the same count.")


def _validate_prediction_count(priors, pred_boxes, pred_scores) -> None:
    if priors.shape[0] != pred_boxes.shape[0] or priors.shape[0] != pred_scores.shape[0]:
        raise ValueError("priors, pred_boxes, and pred_scores must have the same prediction count.")


def _empty_assignment(priors, gt_boxes) -> AssignmentResult:
    count = priors.shape[0]
    return AssignmentResult(
        assigned_gt_indices=torch.zeros((count,), dtype=torch.long, device=priors.device),
        labels=torch.zeros((count,), dtype=torch.long, device=priors.device),
        max_overlaps=priors.new_zeros((count,)),
        matched_boxes=priors.new_zeros((count, 4)) if gt_boxes.numel() == 0 else gt_boxes.new_zeros((count, 4)),
    )


def _empty_point_assignment(points, gt_boxes) -> AssignmentResult:
    count = points.shape[0]
    return AssignmentResult(
        assigned_gt_indices=torch.zeros((count,), dtype=torch.long, device=points.device),
        labels=torch.zeros((count,), dtype=torch.long, device=points.device),
        max_overlaps=points.new_zeros((count,)),
        matched_boxes=points.new_zeros((count, 4)) if gt_boxes.numel() == 0 else gt_boxes.new_zeros((count, 4)),
    )


def _result_from_assigned(priors, gt_boxes, gt_labels, assigned, max_overlaps, argmax_overlaps) -> AssignmentResult:
    labels = torch.zeros((assigned.shape[0],), dtype=torch.long, device=priors.device)
    matched_boxes = priors.new_zeros((assigned.shape[0], 4))
    positive = assigned > 0
    ignored = assigned < 0
    if positive.any():
        positive_gt = assigned[positive] - 1
        labels[positive] = gt_labels[positive_gt]
        matched_boxes[positive] = gt_boxes[positive_gt]
    labels[ignored] = IGNORE_LABEL
    background = (assigned == 0) & (gt_boxes.shape[0] > 0)
    if background.any():
        matched_boxes[background] = gt_boxes[argmax_overlaps[background]]
    return AssignmentResult(
        assigned_gt_indices=assigned,
        labels=labels,
        max_overlaps=max_overlaps,
        matched_boxes=matched_boxes,
    )


def _apply_ignored_boxes(
    result: AssignmentResult,
    priors,
    ignored_boxes,
    ignore_iou_thr: float,
) -> AssignmentResult:
    if ignored_boxes is None or float(ignore_iou_thr) < 0:
        return result
    ignored_boxes = _as_boxes(ignored_boxes, name="ignored_boxes", device=priors.device, dtype=priors.dtype)
    if ignored_boxes.numel() == 0 or priors.numel() == 0:
        return result
    ignored = box_iou(priors, ignored_boxes).max(dim=1).values >= float(ignore_iou_thr)
    if not ignored.any():
        return result
    assigned = result.assigned_gt_indices.clone()
    labels = result.labels.clone()
    max_overlaps = result.max_overlaps.clone()
    matched_boxes = result.matched_boxes.clone()
    assigned[ignored] = IGNORE_LABEL
    labels[ignored] = IGNORE_LABEL
    max_overlaps[ignored] = 0
    matched_boxes[ignored] = 0
    return AssignmentResult(assigned, labels, max_overlaps, matched_boxes)


def _normalize_level_counts(num_level_priors, total: int) -> tuple[int, ...]:
    counts = tuple(int(value) for value in num_level_priors)
    if any(count < 0 for count in counts):
        raise ValueError("num_level_priors cannot contain negative counts.")
    if sum(counts) != int(total):
        raise ValueError(f"num_level_priors must sum to {total}, got {sum(counts)}.")
    return counts


def _box_centers(boxes):
    return torch.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5), dim=1)


def _points_in_boxes(points, boxes):
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return torch.zeros((points.shape[0], boxes.shape[0]), dtype=torch.bool, device=points.device)
    return (
        (points[:, None, 0] >= boxes[None, :, 0])
        & (points[:, None, 0] <= boxes[None, :, 2])
        & (points[:, None, 1] >= boxes[None, :, 1])
        & (points[:, None, 1] <= boxes[None, :, 3])
    )


def _box_areas(boxes):
    return ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0))


def _center_boxes(gt_boxes, *, radius: float = 1.5):
    centers = _box_centers(gt_boxes)
    widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    half_w = widths * 0.5 * min(float(radius), 1.0)
    half_h = heights * 0.5 * min(float(radius), 1.0)
    return torch.stack(
        (
            (centers[:, 0] - half_w).clamp(min=gt_boxes[:, 0]),
            (centers[:, 1] - half_h).clamp(min=gt_boxes[:, 1]),
            (centers[:, 0] + half_w).clamp(max=gt_boxes[:, 2]),
            (centers[:, 1] + half_h).clamp(max=gt_boxes[:, 3]),
        ),
        dim=1,
    )


def _points_in_center_regions(points, gt_boxes, *, strides, radius: float):
    stride_tensor = torch.as_tensor(strides, dtype=points.dtype, device=points.device)
    if stride_tensor.numel() == 1:
        stride_tensor = stride_tensor.expand(points.shape[0])
    if stride_tensor.dim() != 1 or stride_tensor.shape[0] != points.shape[0]:
        raise ValueError("strides must be a scalar or one stride per point.")
    centers = _box_centers(gt_boxes)
    half = stride_tensor[:, None] * float(radius)
    left = torch.maximum(centers[None, :, 0] - half, gt_boxes[None, :, 0])
    top = torch.maximum(centers[None, :, 1] - half, gt_boxes[None, :, 1])
    right = torch.minimum(centers[None, :, 0] + half, gt_boxes[None, :, 2])
    bottom = torch.minimum(centers[None, :, 1] + half, gt_boxes[None, :, 3])
    return (
        (points[:, None, 0] >= left)
        & (points[:, None, 0] <= right)
        & (points[:, None, 1] >= top)
        & (points[:, None, 1] <= bottom)
    )


def _assign_points_by_area(points, gt_boxes, gt_labels, candidate_mask) -> AssignmentResult:
    areas = _box_areas(gt_boxes)
    candidate_areas = torch.where(
        candidate_mask,
        areas.unsqueeze(0),
        areas.new_full((points.shape[0], gt_boxes.shape[0]), float("inf")),
    )
    matched_areas, matched_indices = candidate_areas.min(dim=1)
    positive = torch.isfinite(matched_areas)
    assigned = torch.zeros((points.shape[0],), dtype=torch.long, device=points.device)
    assigned[positive] = matched_indices[positive] + 1
    max_overlaps = points.new_zeros((points.shape[0],))
    return _result_from_assigned(
        points.new_zeros((points.shape[0], 4)),
        gt_boxes,
        gt_labels,
        assigned,
        max_overlaps,
        matched_indices,
    )


def _topk_mask(values, topk: int):
    mask = torch.zeros_like(values, dtype=torch.bool)
    if values.shape[0] == 0 or values.shape[1] == 0 or topk <= 0:
        return mask
    k = min(int(topk), values.shape[0])
    top_indices = values.topk(k, dim=0, largest=True).indices
    positive_values = values[top_indices, torch.arange(values.shape[1], device=values.device)] > 0
    mask[top_indices, torch.arange(values.shape[1], device=values.device)] = positive_values
    return mask


def _assign_candidates(overlaps, candidate_mask):
    assigned = torch.zeros((overlaps.shape[0],), dtype=torch.long, device=overlaps.device)
    masked = torch.where(candidate_mask, overlaps, overlaps.new_full(overlaps.shape, -1))
    max_overlaps, argmax_overlaps = masked.max(dim=1)
    positive = max_overlaps >= 0
    assigned[positive] = argmax_overlaps[positive] + 1
    max_overlaps = torch.where(positive, max_overlaps, overlaps.new_zeros(()))
    return assigned, max_overlaps, argmax_overlaps.clamp(min=0)


def _scores_for_labels(scores, gt_labels):
    if scores.shape[1] == 0:
        return scores.new_zeros((scores.shape[0], gt_labels.shape[0]))
    zero_based = (gt_labels.long() - 1).clamp(min=0, max=scores.shape[1] - 1)
    return scores[:, zero_based]


def _hungarian_indices(cost):
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        empty = torch.empty((0,), dtype=torch.long, device=cost.device)
        return empty, empty
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.t() if transposed else cost
    assignment = _hungarian_rectangular(matrix.detach().cpu().tolist())
    row_indices = []
    col_indices = []
    for row, col in enumerate(assignment):
        if col >= 0:
            row_indices.append(row)
            col_indices.append(col)
    if transposed:
        pred_indices = torch.tensor(col_indices, dtype=torch.long, device=cost.device)
        gt_indices = torch.tensor(row_indices, dtype=torch.long, device=cost.device)
    else:
        pred_indices = torch.tensor(row_indices, dtype=torch.long, device=cost.device)
        gt_indices = torch.tensor(col_indices, dtype=torch.long, device=cost.device)
    return pred_indices, gt_indices


def _hungarian_rectangular(matrix: list[list[float]]) -> list[int]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows > cols:
        raise ValueError("Hungarian solver expects rows <= columns.")
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for row in range(1, rows + 1):
        p[0] = row
        minv = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        column = 0
        while True:
            used[column] = True
            current_row = p[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                cur = float(matrix[current_row - 1][candidate - 1]) - u[current_row] - v[candidate]
                if cur < minv[candidate]:
                    minv[candidate] = cur
                    way[candidate] = column
                if minv[candidate] < delta:
                    delta = minv[candidate]
                    next_column = candidate
            for candidate in range(0, cols + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minv[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            next_column = way[column]
            p[column] = p[next_column]
            column = next_column
            if column == 0:
                break
    assignment = [-1] * rows
    for column in range(1, cols + 1):
        if p[column] > 0:
            assignment[p[column] - 1] = column - 1
    return assignment


__all__ = [
    "ATSSAssigner",
    "AssignmentResult",
    "BalancedSampler",
    "CenterRegionAssigner",
    "HungarianAssigner",
    "IGNORE_LABEL",
    "MaxIoUAssigner",
    "PointAssigner",
    "SamplingResult",
    "SimOTAAssigner",
    "TaskAlignedAssigner",
    "atss_assign",
    "center_region_assign",
    "hungarian_assign",
    "max_iou_assign",
    "point_assign",
    "sample_assignment",
    "sim_ota_assign",
    "task_aligned_assign",
]
