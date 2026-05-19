"""Native ROI-core helpers for the Lightning backend."""

from __future__ import annotations

import math
from contextlib import nullcontext
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency

require_dependency("torch", "native roi")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .assignment import max_iou_assign, sample_assignment  # noqa: E402
from .geometry import (  # noqa: E402
    clip_boxes_to_image,
    decode_boxes as decode_bbox_deltas,
    encode_boxes as encode_bbox_deltas,
)


@dataclass(slots=True, frozen=True)
class RoICoreSpec:
    featmap_names: tuple[str, ...]
    box_output_size: int = 7
    mask_output_size: int = 14
    sampling_ratio: int = 2

    @classmethod
    def from_num_levels(cls, num_levels: int) -> "RoICoreSpec":
        featmap_names = tuple(str(index) for index in range(int(num_levels)))
        return cls(featmap_names=featmap_names)


@dataclass(slots=True, frozen=True)
class BatchedRoIProposals:
    """Proposal tensors formatted for torchvision ROI pooling and ROI ops."""

    proposals: tuple[Any, ...]
    rois: Any
    image_indices: Any
    counts: tuple[int, ...]

    @property
    def num_images(self) -> int:
        return len(self.proposals)

    @property
    def num_proposals(self) -> int:
        return int(sum(self.counts))


@dataclass(slots=True, frozen=True)
class RoIEncodedBBoxTargets:
    bbox_targets: Any
    bbox_weights: Any


@dataclass(slots=True, frozen=True)
class RoIBBoxTargets:
    proposals: Any
    labels: Any
    label_weights: Any
    bbox_targets: Any
    bbox_weights: Any
    matched_gt_indices: Any
    positive_indices: Any
    negative_indices: Any
    sampled_indices: Any


@dataclass(slots=True, frozen=True)
class RoIMaskTargets:
    mask_targets: Any
    positive_indices: Any
    matched_gt_indices: Any


@dataclass(slots=True, frozen=True)
class RoICascadeStage:
    proposals: Any
    labels: Any


@dataclass(slots=True, frozen=True)
class RoIGridTargets:
    points: Any
    normalized_offsets: Any
    weights: Any


def batch_roi_proposals(
    proposals,
    *,
    batch_size: int | None = None,
    reference=None,
    dtype=None,
    device=None,
) -> BatchedRoIProposals:
    """Normalize per-image xyxy proposals and build a batched ``(K, 5)`` ROI tensor."""

    proposal_items = _normalize_proposal_items(proposals, batch_size=batch_size)
    proposal_tensors = tuple(
        _as_roi_boxes(item, name=f"proposals[{index}]", reference=reference, dtype=dtype, device=device)
        for index, item in enumerate(proposal_items)
    )
    counts = tuple(_row_count(tensor) for tensor in proposal_tensors)
    tensor_reference = proposal_tensors[0] if proposal_tensors else reference
    rois_by_image = []
    image_indices_by_image = []
    for image_index, boxes in enumerate(proposal_tensors):
        count = counts[image_index]
        if count == 0:
            continue
        image_column = _full_tensor(
            (count, 1),
            float(image_index),
            reference=boxes,
            dtype=getattr(boxes, "dtype", None),
            device=getattr(boxes, "device", None),
        )
        rois_by_image.append(_cat_tensors((image_column, boxes), dim=1))
        image_indices_by_image.append(
            _full_tensor(
                (count,),
                int(image_index),
                reference=boxes,
                dtype=getattr(torch, "long", None),
                device=getattr(boxes, "device", None),
            )
        )
    rois = (
        _cat_tensors(tuple(rois_by_image), dim=0)
        if rois_by_image
        else _empty_tensor((0, 5), reference=tensor_reference, dtype=dtype or getattr(torch, "float32", None), device=device)
    )
    image_indices = (
        _cat_tensors(tuple(image_indices_by_image), dim=0)
        if image_indices_by_image
        else _empty_tensor((0,), reference=tensor_reference, dtype=getattr(torch, "long", None), device=device)
    )
    return BatchedRoIProposals(
        proposals=proposal_tensors,
        rois=rois,
        image_indices=image_indices,
        counts=counts,
    )


def roi_align_features(pool, features, proposals, image_shapes):
    """Run a torchvision-style ROI pool with empty-proposal handling."""

    reference = _first_feature_tensor(features)
    batched = batch_roi_proposals(
        proposals,
        batch_size=len(image_shapes) if image_shapes is not None else None,
        reference=reference,
    )
    if batched.num_proposals == 0:
        return _empty_roi_pool_output(features, pool)
    return pool(features, list(batched.proposals), list(image_shapes))


def encode_roi_bbox_targets(
    proposals,
    matched_boxes,
    *,
    positive_mask=None,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> RoIEncodedBBoxTargets:
    """Encode matched GT boxes as bbox deltas and weights for positive ROIs."""

    proposal_tensor = _as_roi_boxes(proposals, name="proposals")
    matched_tensor = _as_roi_boxes(
        matched_boxes,
        name="matched_boxes",
        reference=proposal_tensor,
        dtype=getattr(proposal_tensor, "dtype", None),
        device=getattr(proposal_tensor, "device", None),
    )
    if proposal_tensor.shape[0] != matched_tensor.shape[0]:
        raise ValueError("proposals and matched_boxes must have the same count.")
    bbox_targets = proposal_tensor.new_zeros((proposal_tensor.shape[0], 4))
    bbox_weights = proposal_tensor.new_zeros((proposal_tensor.shape[0], 4))
    if proposal_tensor.shape[0] == 0:
        return RoIEncodedBBoxTargets(bbox_targets=bbox_targets, bbox_weights=bbox_weights)
    if positive_mask is None:
        positive_mask = torch.ones((proposal_tensor.shape[0],), dtype=torch.bool, device=proposal_tensor.device)
    else:
        positive_mask = torch.as_tensor(positive_mask, dtype=torch.bool, device=proposal_tensor.device)
    if bool(positive_mask.any()):
        bbox_targets[positive_mask] = encode_bbox_deltas(
            proposal_tensor[positive_mask],
            matched_tensor[positive_mask],
            weights=weights,
        )
        bbox_weights[positive_mask] = 1.0
    return RoIEncodedBBoxTargets(bbox_targets=bbox_targets, bbox_weights=bbox_weights)


def build_roi_bbox_targets(
    proposals,
    gt_boxes,
    gt_labels,
    *,
    num_samples: int | None = None,
    positive_fraction: float = 0.25,
    pos_iou_thr: float = 0.5,
    neg_iou_thr: float = 0.5,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> RoIBBoxTargets:
    """Assign, optionally sample, and encode bbox targets for ROI box heads."""

    proposal_tensor = _as_roi_boxes(proposals, name="proposals")
    gt_box_tensor = _as_roi_boxes(
        gt_boxes,
        name="gt_boxes",
        reference=proposal_tensor,
        dtype=getattr(proposal_tensor, "dtype", None),
        device=getattr(proposal_tensor, "device", None),
    )
    gt_label_tensor = _as_roi_labels(gt_labels, device=proposal_tensor.device)
    assignment = max_iou_assign(
        proposal_tensor,
        gt_box_tensor,
        gt_label_tensor,
        pos_iou_thr=pos_iou_thr,
        neg_iou_thr=neg_iou_thr,
    )
    if num_samples is None:
        sampled_indices = torch.arange(proposal_tensor.shape[0], dtype=torch.long, device=proposal_tensor.device)
    else:
        sample = sample_assignment(
            assignment,
            num_samples=int(num_samples),
            positive_fraction=float(positive_fraction),
        )
        sampled_parts = [sample.positive_indices, sample.negative_indices]
        sampled_indices = (
            torch.cat(sampled_parts, dim=0)
            if any(part.numel() > 0 for part in sampled_parts)
            else torch.empty((0,), dtype=torch.long, device=proposal_tensor.device)
        )

    sampled_proposals = proposal_tensor[sampled_indices]
    sampled_labels = assignment.labels[sampled_indices]
    sampled_matched = assignment.matched_boxes[sampled_indices]
    matched_gt_indices = assignment.assigned_gt_indices[sampled_indices]
    positive_mask = sampled_labels > 0
    encoded = encode_roi_bbox_targets(
        sampled_proposals,
        sampled_matched,
        positive_mask=positive_mask,
        weights=weights,
    )
    label_weights = (sampled_labels >= 0).to(dtype=proposal_tensor.dtype)
    positive_indices = torch.nonzero(positive_mask, as_tuple=False).reshape(-1)
    negative_indices = torch.nonzero(sampled_labels == 0, as_tuple=False).reshape(-1)
    return RoIBBoxTargets(
        proposals=sampled_proposals,
        labels=sampled_labels,
        label_weights=label_weights,
        bbox_targets=encoded.bbox_targets,
        bbox_weights=encoded.bbox_weights,
        matched_gt_indices=matched_gt_indices,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        sampled_indices=sampled_indices,
    )


def build_roi_mask_targets(
    proposals,
    gt_masks,
    matched_gt_indices,
    *,
    output_size: int | Sequence[int] = 28,
) -> RoIMaskTargets:
    """Crop and resize positive ROI mask targets from one-based GT assignments."""

    proposal_tensor = _as_roi_boxes(proposals, name="proposals")
    out_h, out_w = _normalize_output_size(output_size)
    assigned = _matched_indices_tensor(matched_gt_indices, device=proposal_tensor.device)
    if assigned.shape[0] != proposal_tensor.shape[0]:
        raise ValueError("matched_gt_indices must match the number of proposals.")
    positive_indices = torch.nonzero(assigned > 0, as_tuple=False).reshape(-1)
    if positive_indices.numel() == 0:
        return RoIMaskTargets(
            mask_targets=proposal_tensor.new_zeros((0, out_h, out_w)),
            positive_indices=positive_indices,
            matched_gt_indices=assigned,
        )

    masks = _as_mask_tensor(gt_masks, device=proposal_tensor.device, dtype=proposal_tensor.dtype)
    if masks.shape[0] == 0:
        return RoIMaskTargets(
            mask_targets=proposal_tensor.new_zeros((0, out_h, out_w)),
            positive_indices=positive_indices.new_zeros((0,)),
            matched_gt_indices=assigned,
        )

    resized_masks = []
    for proposal_index in positive_indices.tolist():
        gt_index = int(assigned[proposal_index].item()) - 1
        if gt_index < 0 or gt_index >= masks.shape[0]:
            resized_masks.append(proposal_tensor.new_zeros((out_h, out_w)))
            continue
        resized_masks.append(
            _crop_and_resize_mask(
                masks[gt_index],
                proposal_tensor[proposal_index],
                output_size=(out_h, out_w),
            )
        )
    return RoIMaskTargets(
        mask_targets=torch.stack(resized_masks, dim=0),
        positive_indices=positive_indices,
        matched_gt_indices=assigned,
    )


def select_class_specific_bbox_deltas(box_deltas, labels, *, num_classes: int | None = None):
    """Select per-ROI bbox deltas for one-based foreground labels."""

    deltas = torch.as_tensor(box_deltas)
    label_tensor = torch.as_tensor(labels, dtype=torch.long, device=deltas.device).reshape(-1)
    if deltas.numel() == 0:
        return deltas.reshape(0, 4)
    if deltas.dim() == 2 and deltas.shape[1] == 4:
        return deltas.to(device=deltas.device)
    if deltas.dim() == 2:
        if deltas.shape[1] % 4 != 0:
            raise ValueError("box_deltas second dimension must be 4 or a multiple of 4.")
        inferred_classes = deltas.shape[1] // 4 - 1
        if num_classes is None:
            num_classes = inferred_classes
        deltas = deltas.reshape(-1, int(num_classes) + 1, 4)
    elif deltas.dim() != 3 or deltas.shape[2] != 4:
        raise ValueError("box_deltas must have shape (N, 4), (N, C*4), or (N, C, 4).")
    if deltas.shape[0] != label_tensor.shape[0]:
        raise ValueError("labels must match the number of box_deltas rows.")
    max_label = deltas.shape[1] - 1
    label_tensor = label_tensor.clamp(min=0, max=max_label)
    row_indices = torch.arange(deltas.shape[0], dtype=torch.long, device=deltas.device)
    return deltas[row_indices, label_tensor]


def refine_cascade_stage_proposals(
    proposals,
    bbox_deltas,
    labels,
    *,
    image_shape: Sequence[int] | None = None,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    detach: bool = True,
) -> RoICascadeStage:
    """Decode stage-specific bbox deltas into proposals for the next cascade stage."""

    proposal_tensor = _as_roi_boxes(proposals, name="proposals")
    label_tensor = _as_roi_labels(labels, device=proposal_tensor.device)
    selected_deltas = select_class_specific_bbox_deltas(bbox_deltas, label_tensor)
    if proposal_tensor.shape[0] == 0:
        refined = proposal_tensor.new_zeros((0, 4))
    else:
        refined = decode_bbox_deltas(proposal_tensor, selected_deltas.to(device=proposal_tensor.device, dtype=proposal_tensor.dtype), weights=weights)
        if image_shape is not None:
            refined = clip_boxes_to_image(refined, image_shape)
    if detach and hasattr(refined, "detach"):
        refined = refined.detach()
    return RoICascadeStage(proposals=refined, labels=label_tensor)


def build_roi_grid_targets(
    proposals,
    matched_boxes,
    *,
    grid_size: int = 7,
) -> RoIGridTargets:
    """Generate Grid R-CNN style target points and normalized offsets."""

    grid_size = _positive_int(grid_size, "grid_size")
    proposal_tensor = _as_roi_boxes(proposals, name="proposals")
    matched_tensor = _as_roi_boxes(
        matched_boxes,
        name="matched_boxes",
        reference=proposal_tensor,
        dtype=getattr(proposal_tensor, "dtype", None),
        device=getattr(proposal_tensor, "device", None),
    )
    if proposal_tensor.shape[0] != matched_tensor.shape[0]:
        raise ValueError("proposals and matched_boxes must have the same count.")
    if proposal_tensor.shape[0] == 0:
        return RoIGridTargets(
            points=proposal_tensor.new_zeros((0, grid_size, grid_size, 2)),
            normalized_offsets=proposal_tensor.new_zeros((0, grid_size, grid_size, 2)),
            weights=proposal_tensor.new_zeros((0, grid_size, grid_size)),
        )
    steps = torch.linspace(0.0, 1.0, grid_size, dtype=proposal_tensor.dtype, device=proposal_tensor.device)
    gt_widths = matched_tensor[:, 2] - matched_tensor[:, 0]
    gt_heights = matched_tensor[:, 3] - matched_tensor[:, 1]
    xs = matched_tensor[:, 0:1] + steps.reshape(1, -1) * gt_widths.reshape(-1, 1)
    ys = matched_tensor[:, 1:2] + steps.reshape(1, -1) * gt_heights.reshape(-1, 1)
    x_grid = xs[:, None, :].expand(-1, grid_size, -1)
    y_grid = ys[:, :, None].expand(-1, -1, grid_size)
    points = torch.stack((x_grid, y_grid), dim=-1)
    eps = torch.finfo(proposal_tensor.dtype).eps
    proposal_widths = (proposal_tensor[:, 2] - proposal_tensor[:, 0]).clamp(min=eps)
    proposal_heights = (proposal_tensor[:, 3] - proposal_tensor[:, 1]).clamp(min=eps)
    normalized_offsets = torch.stack(
        (
            (points[..., 0] - proposal_tensor[:, None, None, 0]) / proposal_widths[:, None, None],
            (points[..., 1] - proposal_tensor[:, None, None, 1]) / proposal_heights[:, None, None],
        ),
        dim=-1,
    ).clamp(min=0.0, max=1.0)
    valid = ((proposal_widths > eps) & (proposal_heights > eps)).to(dtype=proposal_tensor.dtype)
    weights = valid[:, None, None].expand(-1, grid_size, grid_size)
    return RoIGridTargets(points=points, normalized_offsets=normalized_offsets, weights=weights)


class NativeRoIBackbone(nn.Module):
    """Backbone adapter that exposes explicit ROI feature maps."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: nn.Module,
        core_spec: RoICoreSpec,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.core_spec = core_spec

    def forward(self, images):
        features = self.backbone(images)
        pyramid = self.neck(features)
        return _normalize_feature_pyramid(pyramid, self.core_spec.featmap_names)


class TwoStageDetector(nn.Module):
    """Native two-stage detector composition for proposal and ROI families."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: nn.Module | None = None,
        box_roi_pool: nn.Module,
        num_classes: int,
        in_channels: int,
        core_spec: RoICoreSpec,
        rpn_head: nn.Module | None = None,
        bbox_head: nn.Module | None = None,
        mask_roi_pool: nn.Module | None = None,
        mask_head: nn.Module | None = None,
        grid_roi_pool: nn.Module | None = None,
        grid_head: nn.Module | None = None,
        roi_variant: str = "faster_rcnn",
        proposal_source: str = "rpn",
        cascade_num_stages: int = 1,
        roi_sample_size: int = 2,
        grid_size: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.box_roi_pool = box_roi_pool
        self.roi_extractor = box_roi_pool
        self.mask_roi_pool = mask_roi_pool
        self.mask_roi_extractor = mask_roi_pool
        self.grid_roi_pool = grid_roi_pool
        self.grid_roi_extractor = grid_roi_pool
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        self.grid_head = grid_head
        self.num_classes = int(num_classes)
        self.core_spec = core_spec
        self.with_mask = mask_roi_pool is not None or mask_head is not None
        self.roi_variant = str(roi_variant)
        requested_proposal_source = str(proposal_source).strip().lower() or "rpn"
        if self.rpn_head is None and requested_proposal_source == "rpn":
            requested_proposal_source = "learned"
        self.proposal_source = requested_proposal_source
        self.cascade_num_stages = max(1, int(cascade_num_stages))
        self.grid_size = grid_size
        self.proposal_iou_threshold = 0.5
        self.roi_sample_size = max(1, int(roi_sample_size))
        self.postprocess_topk = 4
        hidden_channels = max(128, int(in_channels))
        self.proposal_head = None
        self.proposal_objectness = None
        self.proposal_regressor = None
        if self.rpn_head is None and self.proposal_source == "learned":
            self.proposal_head = nn.Sequential(
                nn.Linear(int(in_channels), hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
            self.proposal_objectness = nn.Linear(hidden_channels, 1)
            self.proposal_regressor = nn.Linear(hidden_channels, 4)
        self.box_head = None
        self.box_classifier = None
        self.box_regressor = None
        if self.bbox_head is None:
            self.box_head = nn.Sequential(
                nn.Linear(int(in_channels), hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
            self.box_classifier = nn.Linear(hidden_channels, self.num_classes + 1)
            self.box_regressor = nn.Linear(hidden_channels, (self.num_classes + 1) * 4)
        self.mask_predictor = None
        if self.with_mask and self.mask_head is None:
            self.mask_head = nn.Sequential(
                nn.Linear(int(in_channels), hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
            self.mask_predictor = nn.Linear(hidden_channels, 1)

    def forward(self, images, targets=None):
        if targets is not None:
            return self.forward_loss(images, targets)
        return self.predict(images)

    def forward_loss(self, images, targets):
        self._validate_inputs(images, targets=targets)
        return self._forward_impl(images, targets=targets)

    def predict(self, images):
        self._validate_inputs(images)
        no_grad = torch.no_grad() if hasattr(torch, "no_grad") else nullcontext()
        with no_grad:
            return self._forward_impl(images, targets=None)

    def _forward_impl(self, images, targets=None):
        detections: list[dict[str, Any]] = []
        losses: list[dict[str, torch.Tensor]] = []
        for image_index, image in enumerate(images):
            features = self.extract_features(image)
            target = targets[image_index] if targets is not None else None
            anchors, anchor_tensor, refined_proposals, proposal_logits = self._run_rpn_stage(
                image,
                features,
                target=target,
            )
            image_shapes = [self._image_shape(image)]
            proposal_scores = self._proposal_scores(proposal_logits)
            if targets is not None:
                sampled_indices, sampled_labels = self._sample_training_proposals(
                    refined_proposals,
                    anchors,
                    proposal_scores,
                    targets[image_index],
                )
                sampled_proposals = refined_proposals[sampled_indices]
                class_logits, box_deltas, sampled_proposals = self._run_cascade_roi_box_head(
                    features,
                    [sampled_proposals],
                    image_shapes,
                    labels=sampled_labels,
                )
                mask_loss = None
                if self.with_mask and self.mask_roi_pool is not None and self.mask_head is not None:
                    mask_loss = self._mask_loss(
                        features,
                        sampled_proposals,
                        sampled_labels,
                        image_shapes,
                        targets[image_index],
                    )
                grid_loss = None
                if self.grid_head is not None and self.grid_roi_pool is not None:
                    grid_loss = self._grid_loss(
                        features,
                        sampled_proposals,
                        image_shapes,
                        targets[image_index],
                    )
                losses.append(
                    self._loss_from_targets(
                        proposal_logits,
                        anchors,
                        class_logits,
                        box_deltas,
                        sampled_proposals,
                        sampled_labels,
                        targets[image_index],
                        mask_loss=mask_loss,
                        grid_loss=grid_loss,
                    )
                )
                continue

            class_logits, box_deltas, detection_proposals = self._run_cascade_roi_box_head(
                features,
                [refined_proposals],
                image_shapes,
            )
            boxes, scores, labels, detection_indices = self._postprocess_detections(
                detection_proposals,
                proposal_scores,
                class_logits,
                box_deltas,
            )
            detection: dict[str, Any] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            if self.with_mask and self.mask_roi_pool is not None and self.mask_head is not None:
                mask_proposals = self._select_rows(detection_proposals, detection_indices, reference=boxes)
                mask_features = self._roi_pool(
                    self.mask_roi_pool,
                    features,
                    [mask_proposals],
                    image_shapes,
                )
                detection["masks"] = self._decode_masks(mask_features, labels)
            if self.grid_head is not None and self.grid_roi_pool is not None:
                grid_proposals = self._select_rows(detection_proposals, detection_indices, reference=boxes)
                grid_features = self._roi_pool(
                    self.grid_roi_pool,
                    features,
                    [grid_proposals],
                    image_shapes,
                )
                grid_outputs = self.grid_head(grid_features)
                if hasattr(self.grid_head, "decode"):
                    detection["grids"] = self.grid_head.decode(grid_outputs, proposals=grid_proposals)
                else:
                    detection["grids"] = grid_outputs

            detections.append(detection)

        if targets is not None:
            return self._merge_losses(losses)
        return detections

    def _validate_inputs(self, images, *, targets=None):
        if not images:
            raise ValueError("TwoStageDetector requires at least one image.")
        for image in images:
            if not hasattr(image, "shape") or not hasattr(image, "unsqueeze"):
                raise TypeError(
                    "TwoStageDetector expects tensor-like images with 'shape' and 'unsqueeze'."
                )
        if targets is not None and len(targets) != len(images):
            raise ValueError("Number of targets must match number of images.")

    def extract_features(self, image):
        batched = image.unsqueeze(0)
        if self.neck is None:
            features = self.backbone(batched)
        else:
            features = self.neck(self.backbone(batched))
        return _normalize_feature_pyramid(features, self.core_spec.featmap_names)

    def _loss_from_targets(
        self,
        proposal_logits,
        anchors,
        class_logits,
        box_deltas,
        sampled_proposals,
        sampled_labels,
        target,
        *,
        mask_loss=None,
        grid_loss=None,
    ):
        proposal_reference = anchors[0] if len(anchors) > 0 else sampled_proposals
        target_box = self._coerce_box(self._first_target_box(target, proposal_reference), reference=class_logits)
        roi_losses = self._roi_losses(
            sampled_proposals=sampled_proposals,
            sampled_labels=sampled_labels,
            class_logits=class_logits,
            box_deltas=box_deltas,
            target_box=target_box,
        )
        losses = dict(roi_losses)
        if self.rpn_head is not None or self.proposal_source == "learned":
            losses.update(self._rpn_losses(proposal_logits, anchors, target_box))
        if self.with_mask:
            losses["loss_mask"] = mask_loss if mask_loss is not None else self._zero_loss_like(roi_losses["loss_roi_classifier"])
        if self.grid_head is not None:
            losses["loss_grid"] = grid_loss if grid_loss is not None else self._zero_loss_like(roi_losses["loss_roi_classifier"])
        losses["loss_total"] = sum(losses.values())
        return losses

    def _first_target_box(self, target, proposal):
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            return target["boxes"][0]
        if _row_count(proposal) == 0:
            return self._zero_box(reference=proposal)
        return proposal[0]

    def _first_target_label(self, target, *, device):
        if target is not None and "labels" in target and len(target["labels"]) > 0:
            label = target["labels"][0]
        else:
            label = self._tensor_from_scalar(1, device=device, dtype=self._long_dtype())
        if hasattr(label, "to"):
            kwargs = {}
            dtype = self._long_dtype()
            if dtype is not None:
                kwargs["dtype"] = dtype
            if device is not None:
                kwargs["device"] = device
            label = label.to(**kwargs)
            if hasattr(torch, "clamp"):
                return torch.clamp(label, min=0, max=self.num_classes)
            return label
        value = max(0, min(int(label), self.num_classes))
        return self._tensor_from_scalar(value, device=device, dtype=self._long_dtype())

    def _merge_losses(self, losses):
        if not losses:
            zero = torch.zeros(())
            return {
                "loss_rpn_objectness": zero,
                "loss_rpn_box_reg": zero,
                "loss_roi_classifier": zero,
                "loss_roi_box_reg": zero,
                "loss_total": zero,
            }
        keys = sorted({key for loss in losses for key in loss.keys()})
        merged: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [loss[key] for loss in losses if key in loss]
            merged[key] = sum(values) / len(values)
        if "loss_total" not in merged:
            merged["loss_total"] = sum(merged.values())
        return merged

    def _generate_candidate_proposals(self, image):
        return self._heuristic_proposals(image)

    def _generate_anchor_proposals(self, image, features=None):
        if features is None or not hasattr(features, "items"):
            return self._heuristic_proposals(image)
        image_height, image_width = self._image_shape(image)
        anchors: list[tuple[float, float, float, float]] = []
        for level_index, (_, feature) in enumerate(features.items()):
            if not hasattr(feature, "shape") or len(feature.shape) < 4:
                return self._heuristic_proposals(image)
            feature_height = max(1, int(feature.shape[-2]))
            feature_width = max(1, int(feature.shape[-1]))
            stride_y = image_height / float(feature_height)
            stride_x = image_width / float(feature_width)
            scale = max(stride_x, stride_y) * (1.0 + 0.25 * float(level_index))
            for row in range(feature_height):
                center_y = (row + 0.5) * stride_y
                for col in range(feature_width):
                    center_x = (col + 0.5) * stride_x
                    anchors.append(
                        self._clip_box(
                            (
                                center_x - scale / 2.0,
                                center_y - scale / 2.0,
                                center_x + scale / 2.0,
                                center_y + scale / 2.0,
                            ),
                            image_width,
                            image_height,
                        )
                    )
        return anchors or self._heuristic_proposals(image)

    def _run_rpn_stage(self, image, features, *, target=None):
        anchors = self._generate_anchor_proposals(image, features)
        anchor_tensor = self._boxes_to_tensor(anchors, reference=image)
        if self.rpn_head is not None:
            rpn_outputs = self.rpn_head(_feature_sequence(features))
            proposal_logits, proposal_deltas = self._flatten_rpn_outputs(rpn_outputs, anchor_tensor)
            matched_count = min(_row_count(anchor_tensor), _row_count(proposal_logits), _row_count(proposal_deltas))
            if matched_count != _row_count(anchor_tensor):
                anchors = anchors[:matched_count]
                anchor_tensor = anchor_tensor[:matched_count]
            refined_proposals = self._decode_boxes(anchor_tensor, proposal_deltas)
            return anchors, anchor_tensor, refined_proposals, proposal_logits
        if self.proposal_source in {"external", "heuristic"}:
            if target is not None and "proposals" in target and _row_count(target["proposals"]) > 0:
                refined_proposals = self._coerce_proposal_tensor(target["proposals"], reference=image)
            elif target is not None and "boxes" in target and _row_count(target["boxes"]) > 0:
                refined_proposals = self._coerce_proposal_tensor(target["boxes"], reference=image)
            else:
                refined_proposals = anchor_tensor
            proposal_logits = refined_proposals.new_zeros((int(refined_proposals.shape[0]),))
            proposal_rows = [tuple(float(value) for value in row.tolist()) for row in refined_proposals]
            return proposal_rows, refined_proposals, refined_proposals, proposal_logits
        if self.proposal_head is None or self.proposal_objectness is None or self.proposal_regressor is None:
            raise ValueError(
                f"Two-stage detector '{self.roi_variant}' requires a native RPN head or explicit proposal source."
            )
        image_shapes = [self._image_shape(image)]
        pooled = self._roi_pool(self.box_roi_pool, features, [anchor_tensor], image_shapes)
        pooled = pooled.mean(dim=(-1, -2))
        proposal_representation = self.proposal_head(pooled)
        proposal_logits = self.proposal_objectness(proposal_representation).squeeze(-1)
        proposal_deltas = self.proposal_regressor(proposal_representation)
        refined_proposals = self._decode_boxes(anchor_tensor, proposal_deltas)
        return anchors, anchor_tensor, refined_proposals, proposal_logits

    def _run_roi_box_head(self, features, proposals, image_shapes):
        pooled = self._roi_pool(self.box_roi_pool, features, proposals, image_shapes)
        if self.bbox_head is not None:
            outputs = self.bbox_head(pooled)
            class_logits, box_deltas = self._unpack_bbox_outputs(outputs)
            return class_logits, box_deltas
        pooled = pooled.mean(dim=(-1, -2))
        representation = self.box_head(pooled)
        class_logits = self.box_classifier(representation)
        box_deltas = self.box_regressor(representation)
        return class_logits, box_deltas

    def _run_cascade_roi_box_head(self, features, proposals, image_shapes, *, labels=None):
        current_proposals = proposals[0]
        class_logits = None
        box_deltas = None
        for stage_index in range(self.cascade_num_stages):
            class_logits, box_deltas = self._run_roi_box_head(features, [current_proposals], image_shapes)
            if stage_index >= self.cascade_num_stages - 1:
                break
            current_labels = labels if labels is not None else self._labels_from_logits(class_logits)
            current_proposals = self._refine_cascade_proposals(
                current_proposals,
                class_logits,
                box_deltas,
                current_labels,
                image_shape=image_shapes[0] if image_shapes else None,
            )
        return class_logits, box_deltas, current_proposals

    def _refine_cascade_proposals(self, proposals, class_logits, box_deltas, labels, *, image_shape):
        if (
            self.bbox_head is not None
            and hasattr(self.bbox_head, "refine_proposals")
            and _has_real_tensor_ops()
            and hasattr(proposals, "shape")
        ):
            stage = self.bbox_head.refine_proposals(
                proposals,
                {"cls_score": class_logits, "bbox_pred": box_deltas},
                labels,
                image_shape=image_shape,
            )
            return stage.proposals
        selected = self._select_class_specific_box_deltas(box_deltas, labels, reference=box_deltas)
        return self._decode_boxes(proposals, selected)

    def _labels_from_logits(self, class_logits):
        if hasattr(torch, "argmax") and hasattr(class_logits, "shape"):
            return torch.argmax(class_logits, dim=1).clamp(min=0, max=self.num_classes)
        probabilities = self._proposal_rows(self._softmax_logits(class_logits))
        return [self._foreground_label_and_score(row)[1] for row in probabilities]

    def _flatten_rpn_outputs(self, outputs, anchor_tensor):
        if not isinstance(outputs, dict):
            raise ValueError("RPN head outputs must include objectness_logits and bbox_regression.")
        objectness_levels = outputs.get("objectness_logits")
        bbox_levels = outputs.get("bbox_regression")
        if objectness_levels is None or bbox_levels is None:
            raise ValueError("RPN head outputs must include objectness_logits and bbox_regression.")
        if not isinstance(objectness_levels, (list, tuple)):
            objectness_levels = (objectness_levels,)
        if not isinstance(bbox_levels, (list, tuple)):
            bbox_levels = (bbox_levels,)
        logits = self._flatten_rpn_objectness(objectness_levels)
        deltas = self._flatten_rpn_bbox_deltas(bbox_levels, reference=anchor_tensor)
        return self._match_rpn_rows(logits, deltas, anchor_tensor)

    def _flatten_rpn_objectness(self, levels):
        rows = []
        for level in levels:
            if hasattr(level, "dim") and level.dim() == 4:
                rows.append(level[0].permute(1, 2, 0).reshape(-1))
            elif hasattr(level, "reshape"):
                rows.append(level.reshape(-1))
            else:
                rows.extend(list(level))
        if rows and hasattr(torch, "cat") and hasattr(rows[0], "shape"):
            return torch.cat(rows, dim=0)
        return self._tensor_from_data(rows, dtype=self._float_dtype())

    def _flatten_rpn_bbox_deltas(self, levels, *, reference):
        rows = []
        for level in levels:
            if hasattr(level, "dim") and level.dim() == 4:
                channels = int(level.shape[1])
                anchors_per_location = max(1, channels // 4)
                rows.append(
                    level[0]
                    .reshape(anchors_per_location, 4, int(level.shape[-2]), int(level.shape[-1]))
                    .permute(2, 3, 0, 1)
                    .reshape(-1, 4)
                )
            elif hasattr(level, "reshape"):
                rows.append(level.reshape(-1, 4))
            else:
                rows.extend(list(level))
        if rows and hasattr(torch, "cat") and hasattr(rows[0], "shape"):
            return torch.cat(rows, dim=0)
        return self._boxes_to_tensor(rows, reference=reference)

    def _match_rpn_rows(self, logits, deltas, anchor_tensor):
        anchor_count = _row_count(anchor_tensor)
        logit_count = _row_count(logits)
        delta_count = _row_count(deltas)
        count = min(anchor_count, logit_count, delta_count)
        if count <= 0:
            zero_logits = anchor_tensor.new_zeros((0,)) if hasattr(anchor_tensor, "new_zeros") else []
            zero_deltas = anchor_tensor.new_zeros((0, 4)) if hasattr(anchor_tensor, "new_zeros") else []
            return zero_logits, zero_deltas
        if logit_count != count:
            logits = logits[:count]
        if delta_count != count:
            deltas = deltas[:count]
        return logits, deltas

    def _unpack_bbox_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs["cls_score"], outputs["bbox_pred"]
        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            return outputs[0], outputs[1]
        raise ValueError("ROI bbox head outputs must include cls_score and bbox_pred.")

    def _heuristic_proposals(self, image):
        height, width = self._image_shape(image)
        proposals = [
            (0.0, 0.0, float(width - 1), float(height - 1)),
            self._centered_box(width, height, 1.0, 0.5, 0.5),
            self._centered_box(width, height, 0.75, 0.5, 0.5),
            self._centered_box(width, height, 0.5, 0.33, 0.33),
            self._centered_box(width, height, 0.5, 0.67, 0.67),
        ]
        unique: list[tuple[float, float, float, float]] = []
        for proposal in proposals:
            clipped = self._clip_box(proposal, width, height)
            if clipped not in unique:
                unique.append(clipped)
        return unique

    def _sample_training_proposals(self, proposals, anchors, proposal_scores, target):
        if _row_count(proposals) == 0:
            return [], []
        if _has_real_tensor_ops() and hasattr(proposals, "shape"):
            gt_boxes = target.get("boxes", []) if target is not None else []
            gt_labels = target.get("labels", []) if target is not None else []
            targets = build_roi_bbox_targets(
                proposals,
                gt_boxes,
                gt_labels,
                num_samples=self.roi_sample_size,
                positive_fraction=0.5,
                pos_iou_thr=self.proposal_iou_threshold,
                neg_iou_thr=self.proposal_iou_threshold,
            )
            return targets.sampled_indices.tolist(), targets.labels.tolist()
        target_box = self._coerce_box(self._first_target_box(target, anchors[0]), reference=proposal_scores)
        target_label = self._first_target_label(target, device=getattr(proposal_scores, "device", None))
        positive_index = self._assign_proposal_index(proposals, target_box)
        negative_index = self._select_negative_proposal_index(
            proposals,
            proposal_scores,
            target_box,
            exclude={positive_index},
        )
        sampled_indices = [positive_index]
        sampled_labels = [int(target_label)]
        if negative_index is not None and negative_index != positive_index:
            sampled_indices.append(negative_index)
            sampled_labels.append(0)
        return sampled_indices[: self.roi_sample_size], sampled_labels[: self.roi_sample_size]

    def _assign_proposal_index(self, proposals, target_box):
        target = self._box_to_tuple(target_box)
        best_index = 0
        best_iou = -1.0
        for index, proposal in enumerate(proposals):
            iou = self._box_iou(proposal, target)
            if iou > best_iou:
                best_index = index
                best_iou = iou
        return best_index

    def _select_proposal_index(self, probabilities):
        if hasattr(probabilities, "tolist"):
            values = probabilities.tolist()
        else:
            values = list(probabilities)
        if not values:
            return 0
        if not isinstance(values[0], (list, tuple)):
            scores = [float(value) for value in values]
            return max(range(len(scores)), key=scores.__getitem__)
        rows = self._probability_rows(values)
        scores = [self._foreground_score(row) for row in rows]
        return max(range(len(scores)), key=scores.__getitem__)

    def _select_negative_proposal_index(self, proposals, proposal_scores, target_box, *, exclude):
        best_index = None
        best_score = -1.0
        for index, proposal in enumerate(proposals):
            if index in exclude:
                continue
            if self._box_iou(proposal, target_box) >= self.proposal_iou_threshold:
                continue
            score = self._proposal_score_value(proposal_scores, index)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None:
            return best_index
        for index, _proposal in enumerate(proposals):
            if index not in exclude:
                return index
        return None

    def _proposal_probabilities(self, class_logits):
        rows = self._probability_rows(class_logits)
        return [self._softmax(row) for row in rows]

    def _proposal_scores(self, objectness_logits):
        if hasattr(torch, "sigmoid") and hasattr(objectness_logits, "shape"):
            return torch.sigmoid(objectness_logits)
        rows = self._probability_rows(objectness_logits)
        if rows and not isinstance(rows[0], (list, tuple)):
            return [float(value) for value in rows]
        return [self._foreground_score(row) for row in rows]

    def _proposal_objectness_targets(self, anchors, target_box, *, reference):
        target = self._coerce_box(target_box, reference=reference)
        labels = [
            1.0 if self._box_iou(anchor, target) >= self.proposal_iou_threshold else 0.0
            for anchor in anchors
        ]
        kwargs = {}
        dtype = getattr(reference, "dtype", None) or self._float_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype
        device = getattr(reference, "device", None)
        if device is not None:
            kwargs["device"] = device
        return self._tensor_from_data(labels, **kwargs)

    def _rpn_losses(self, proposal_logits, anchors, target_box):
        if len(anchors) == 0 or _tensor_numel(proposal_logits) == 0:
            zero = self._zero_loss_like(proposal_logits)
            return {
                "loss_rpn_objectness": zero,
                "loss_rpn_box_reg": zero,
            }
        objectness_targets = self._proposal_objectness_targets(anchors, target_box, reference=proposal_logits)
        objectness_loss = self._binary_cross_entropy_with_logits(proposal_logits, objectness_targets)
        selected_index = self._assign_proposal_index(anchors, target_box)
        selected_proposal = self._boxes_to_tensor([anchors[selected_index]], reference=proposal_logits)
        box_loss = self._smooth_l1_loss(selected_proposal, target_box.unsqueeze(0))
        return {
            "loss_rpn_objectness": objectness_loss,
            "loss_rpn_box_reg": box_loss,
        }

    def _roi_losses(
        self,
        *,
        sampled_proposals,
        sampled_labels,
        class_logits,
        box_deltas,
        target_box,
    ):
        if _row_count(sampled_labels) == 0 or _tensor_numel(class_logits) == 0:
            zero = self._zero_loss_like(class_logits)
            return {
                "loss_roi_classifier": zero,
                "loss_roi_box_reg": zero,
            }
        if _has_real_tensor_ops() and hasattr(class_logits, "shape") and hasattr(box_deltas, "shape"):
            return self._tensor_roi_losses(
                sampled_proposals=sampled_proposals,
                sampled_labels=sampled_labels,
                class_logits=class_logits,
                box_deltas=box_deltas,
                target_box=target_box,
            )
        roi_targets = self._roi_classification_targets(sampled_labels, reference=class_logits)
        classifier_loss = self._cross_entropy_loss(class_logits, roi_targets)
        positive_indices = self._positive_sample_indices(sampled_labels)
        if positive_indices:
            positive_proposals = self._select_rows(sampled_proposals, positive_indices, reference=box_deltas)
            positive_labels = self._select_label_rows(roi_targets, positive_indices, reference=roi_targets)
        else:
            positive_proposals = self._select_rows(sampled_proposals, [0], reference=box_deltas)
            positive_labels = self._select_label_rows(roi_targets, [0], reference=roi_targets)
        positive_box_deltas = self._select_class_specific_box_deltas(
            box_deltas,
            positive_labels,
            reference=box_deltas,
        )
        predicted_box = self._decode_boxes(positive_proposals, positive_box_deltas)
        target_boxes = self._expand_target_box(target_box, len(positive_indices) or 1, reference=predicted_box)
        box_loss = self._smooth_l1_loss(predicted_box, target_boxes)
        return {
            "loss_roi_classifier": classifier_loss,
            "loss_roi_box_reg": box_loss,
        }

    def _tensor_roi_losses(
        self,
        *,
        sampled_proposals,
        sampled_labels,
        class_logits,
        box_deltas,
        target_box,
    ):
        roi_targets = self._roi_classification_targets(sampled_labels, reference=class_logits)
        classifier_loss = self._cross_entropy_loss(class_logits, roi_targets)
        positive_indices = torch.nonzero(roi_targets > 0, as_tuple=False).reshape(-1)
        if positive_indices.numel() == 0:
            box_loss = self._zero_loss_like(class_logits)
        else:
            positive_proposals = sampled_proposals.index_select(0, positive_indices)
            positive_labels = roi_targets.index_select(0, positive_indices)
            positive_box_deltas = select_class_specific_bbox_deltas(
                box_deltas.index_select(0, positive_indices),
                positive_labels,
                num_classes=self.num_classes,
            )
            target_boxes = self._expand_target_box(target_box, int(positive_indices.numel()), reference=positive_proposals)
            encoded = encode_roi_bbox_targets(positive_proposals, target_boxes)
            box_loss = self._smooth_l1_loss(positive_box_deltas, encoded.bbox_targets)
        return {
            "loss_roi_classifier": classifier_loss,
            "loss_roi_box_reg": box_loss,
        }

    def _mask_loss(self, features, sampled_proposals, sampled_labels, image_shapes, target):
        if target is None or "masks" not in target or _row_count(sampled_proposals) == 0:
            return None
        if not (_has_real_tensor_ops() and hasattr(sampled_proposals, "shape")):
            return None
        gt_boxes = target.get("boxes", [])
        gt_labels = target.get("labels", [])
        bbox_targets = build_roi_bbox_targets(
            sampled_proposals,
            gt_boxes,
            gt_labels,
            num_samples=None,
            pos_iou_thr=self.proposal_iou_threshold,
            neg_iou_thr=self.proposal_iou_threshold,
        )
        if hasattr(self.mask_head, "get_targets") and hasattr(self.mask_head, "loss"):
            mask_features = self._roi_pool(self.mask_roi_pool, features, [sampled_proposals], image_shapes)
            mask_outputs = self.mask_head(mask_features)
            mask_targets = self.mask_head.get_targets(
                sampled_proposals,
                target["masks"],
                bbox_targets.matched_gt_indices,
            )
            losses = self.mask_head.loss(mask_outputs, mask_targets, labels=bbox_targets.labels)
            return losses.get("loss_mask", losses.get("loss_total"))
        mask_targets = build_roi_mask_targets(
            sampled_proposals,
            target["masks"],
            bbox_targets.matched_gt_indices,
            output_size=1,
        )
        if mask_targets.mask_targets.shape[0] == 0:
            return self._zero_loss_like(sampled_proposals)
        positive_proposals = sampled_proposals.index_select(0, mask_targets.positive_indices)
        mask_features = self._roi_pool(self.mask_roi_pool, features, [positive_proposals], image_shapes)
        mask_features = mask_features.mean(dim=(-1, -2))
        mask_representation = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_representation).reshape(-1, 1)
        targets = mask_targets.mask_targets.reshape(mask_targets.mask_targets.shape[0], -1).mean(dim=1, keepdim=True)
        return F.binary_cross_entropy_with_logits(mask_logits, targets)

    def _grid_loss(self, features, sampled_proposals, image_shapes, target):
        if target is None or _row_count(sampled_proposals) == 0:
            return None
        if not (_has_real_tensor_ops() and hasattr(sampled_proposals, "shape")):
            return None
        target_box = self._coerce_box(
            self._first_target_box(target, sampled_proposals),
            reference=sampled_proposals,
        )
        matched_boxes = self._expand_target_box(
            target_box,
            _row_count(sampled_proposals),
            reference=sampled_proposals,
        )
        grid_features = self._roi_pool(self.grid_roi_pool, features, [sampled_proposals], image_shapes)
        grid_outputs = self.grid_head(grid_features)
        if hasattr(self.grid_head, "get_targets") and hasattr(self.grid_head, "loss"):
            grid_targets = self.grid_head.get_targets(sampled_proposals, matched_boxes)
            losses = self.grid_head.loss(grid_outputs, grid_targets)
            return losses.get("loss_grid", losses.get("loss_total"))
        return None

    def _decode_masks(self, mask_features, labels):
        if hasattr(self.mask_head, "decode"):
            mask_outputs = self.mask_head(mask_features)
            return self.mask_head.decode(mask_outputs, labels=labels)
        mask_features = mask_features.mean(dim=(-1, -2))
        mask_representation = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_representation)
        return torch.sigmoid(mask_logits).reshape(-1, 1, 1, 1)

    def _image_shape(self, image):
        return int(image.shape[-2]), int(image.shape[-1])

    def _roi_pool(self, pool, features, proposals, image_shapes):
        return roi_align_features(pool, features, proposals, image_shapes)

    def _decode_boxes(self, proposal, box_deltas):
        if not hasattr(proposal, "to") or not hasattr(box_deltas, "reshape"):
            proposal_rows = self._proposal_rows(proposal)
            delta_rows = self._proposal_rows(box_deltas)
            decoded_rows = []
            for proposal_row, delta_row in zip(proposal_rows, delta_rows):
                x1, y1, x2, y2 = self._box_to_tuple(proposal_row)
                if delta_row and isinstance(delta_row[0], (list, tuple)):
                    delta_row = delta_row[0]
                deltas = [float(value) for value in list(delta_row)[:4]]
                while len(deltas) < 4:
                    deltas.append(0.0)
                dx1, dy1, dx2, dy2 = [math.tanh(value) for value in deltas[:4]]
                adjusted = (
                    x1 + dx1,
                    y1 + dy1,
                    max(x2 + dx2, x1 + dx1 + 1.0),
                    max(y2 + dy2, y1 + dy1 + 1.0),
                )
                decoded_rows.append(adjusted)
            return self._boxes_to_tensor(decoded_rows, reference=box_deltas)
        boxes = proposal.to(dtype=box_deltas.dtype, device=box_deltas.device).clone()
        deltas = torch.tanh(box_deltas).reshape_as(boxes)
        boxes = boxes + deltas
        boxes[..., 2] = torch.maximum(boxes[..., 2], boxes[..., 0] + 1.0)
        boxes[..., 3] = torch.maximum(boxes[..., 3], boxes[..., 1] + 1.0)
        return boxes

    def _postprocess_detections(self, proposals, proposal_scores, class_logits, box_deltas):
        proposal_rows = self._proposal_rows(proposals)
        if not proposal_rows:
            return (
                self._boxes_to_tensor([], reference=box_deltas),
                self._tensor_from_data([], dtype=self._float_dtype(), device=getattr(box_deltas, "device", None)),
                self._tensor_from_data([], dtype=self._long_dtype(), device=getattr(box_deltas, "device", None)),
                [],
            )
        proposal_tensor = self._boxes_to_tensor(proposal_rows, reference=box_deltas)
        class_probabilities = self._proposal_rows(self._softmax_logits(class_logits))
        labels: list[int] = []
        class_scores: list[float] = []
        for row in class_probabilities:
            score, label = self._foreground_label_and_score(row)
            labels.append(int(label))
            class_scores.append(float(score))
        selected_box_deltas = self._select_class_specific_box_deltas(
            box_deltas,
            labels,
            reference=box_deltas,
        )
        decoded_boxes = self._decode_boxes(proposal_tensor, selected_box_deltas)
        combined_scores = [
            float(class_scores[index]) * float(self._proposal_score_value(proposal_scores, index))
            for index in range(len(proposal_rows))
        ]
        ranked_indices = sorted(range(len(combined_scores)), key=combined_scores.__getitem__, reverse=True)
        top_indices = ranked_indices[: max(1, min(self.postprocess_topk, len(ranked_indices)))]
        boxes = self._select_rows(decoded_boxes, top_indices, reference=decoded_boxes)
        scores = self._tensor_from_data(
            [combined_scores[index] for index in top_indices],
            dtype=self._float_dtype(),
            device=getattr(box_deltas, "device", None),
        )
        labels_tensor = self._tensor_from_data(
            [labels[index] for index in top_indices],
            dtype=self._long_dtype(),
            device=getattr(box_deltas, "device", None),
        )
        return boxes, scores, labels_tensor, top_indices

    def _boxes_to_tensor(self, proposals, *, reference):
        if not hasattr(torch, "as_tensor"):
            kwargs = {}
            dtype = getattr(reference, "dtype", None) or self._float_dtype()
            if dtype is not None:
                kwargs["dtype"] = dtype
            device = getattr(reference, "device", None)
            if device is not None:
                kwargs["device"] = device
            tensor = self._tensor_from_data(proposals, **kwargs)
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) == 1 and int(shape[0]) == 0 and hasattr(tensor, "reshape"):
                return tensor.reshape(0, 4)
            return tensor
        return _as_roi_boxes(
            proposals,
            name="proposals",
            reference=reference,
            dtype=getattr(reference, "dtype", None) or self._float_dtype(),
            device=getattr(reference, "device", None),
        )

    def _coerce_box(self, value, *, reference):
        if hasattr(value, "to"):
            kwargs = {}
            dtype = getattr(reference, "dtype", None) or self._float_dtype()
            if dtype is not None:
                kwargs["dtype"] = dtype
            device = getattr(reference, "device", None)
            if device is not None:
                kwargs["device"] = device
            return value.to(**kwargs)
        return self._boxes_to_tensor([self._box_to_tuple(value)], reference=reference)[0]

    def _coerce_proposal_tensor(self, value, *, reference):
        if hasattr(value, "to") and hasattr(value, "shape"):
            kwargs = {}
            dtype = getattr(reference, "dtype", None) or self._float_dtype()
            if dtype is not None:
                kwargs["dtype"] = dtype
            device = getattr(reference, "device", None)
            if device is not None:
                kwargs["device"] = device
            proposal_tensor = value.to(**kwargs)
            if proposal_tensor.dim() != 2 or int(proposal_tensor.shape[-1]) != 4:
                raise ValueError("Fast R-CNN proposals must have shape (N, 4).")
            return proposal_tensor
        return self._boxes_to_tensor(value, reference=reference)

    def _scalar_tensor(self, value, *, reference):
        kwargs = {}
        dtype = getattr(reference, "dtype", None) or self._float_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype
        device = getattr(reference, "device", None)
        if device is not None:
            kwargs["device"] = device
        return self._tensor_from_scalar(float(value), **kwargs)

    def _label_tensor(self, value, *, reference):
        kwargs = {}
        dtype = self._long_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype
        device = getattr(reference, "device", None)
        if device is not None:
            kwargs["device"] = device
        return self._tensor_from_scalar(int(value), **kwargs)

    def _proposal_score_value(self, scores, index):
        if hasattr(scores, "tolist"):
            values = scores.tolist()
        else:
            values = list(scores)
        if not values:
            return 0.0
        if not isinstance(values[0], (list, tuple)):
            return float(values[index])
        return self._foreground_score(values[index])

    def _box_to_tuple(self, value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return tuple(float(component) for component in value)

    def _label_values(self, labels):
        if hasattr(labels, "tolist"):
            values = labels.tolist()
        else:
            values = list(labels)
        if values and isinstance(values[0], (list, tuple)):
            values = values[0]
        return [int(value) for value in values]

    def _select_label_rows(self, values, indices, *, reference):
        label_values = self._label_values(values)
        selected_indices = list(indices) if indices else [0]
        selected = [label_values[index] for index in selected_indices if index < len(label_values)]
        if not selected:
            selected = label_values[:1] or [0]
        return self._tensor_from_data(
            [int(value) for value in selected],
            dtype=self._long_dtype(),
            device=getattr(reference, "device", None),
        )

    def _roi_classification_targets(self, labels, *, reference):
        kwargs = {}
        dtype = self._long_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype
        device = getattr(reference, "device", None)
        if device is not None:
            kwargs["device"] = device
        return self._tensor_from_data([int(label) for label in labels], **kwargs)

    def _positive_sample_indices(self, sampled_labels):
        if hasattr(sampled_labels, "tolist"):
            labels = sampled_labels.tolist()
        else:
            labels = list(sampled_labels)
        return [index for index, label in enumerate(labels) if int(label) > 0]

    def _select_rows(self, values, indices, *, reference):
        if hasattr(values, "index_select") and indices:
            index_tensor = self._tensor_from_data(indices, dtype=self._long_dtype(), device=getattr(values, "device", None))
            return values.index_select(0, index_tensor)
        if isinstance(values, (list, tuple)) and values and not isinstance(values[0], (list, tuple)):
            selected_indices = list(indices) if indices else [0]
            selected = [values[index] for index in selected_indices if index < len(values)]
            if not selected:
                selected = list(values[:1])
            return self._tensor_from_data(
                [int(value) for value in selected] if self._is_integer_reference(reference) else [float(value) for value in selected],
                dtype=self._long_dtype() if self._is_integer_reference(reference) else self._float_dtype(),
                device=getattr(reference, "device", None),
            )
        rows = self._proposal_rows(values)
        selected_indices = list(indices) if indices else [0]
        selected = [rows[index] for index in selected_indices if index < len(rows)]
        if rows and not isinstance(rows[0], (list, tuple)):
            if not selected:
                selected = rows[:1]
            return self._tensor_from_data(
                [int(value) for value in selected],
                dtype=self._long_dtype() if self._is_integer_reference(reference) else self._float_dtype(),
                device=getattr(reference, "device", None),
            )
        return self._boxes_to_tensor(selected or rows[:1], reference=reference)

    def _expand_target_box(self, target_box, count, *, reference):
        repeated = [self._box_to_tuple(target_box) for _ in range(max(1, int(count)))]
        return self._boxes_to_tensor(repeated, reference=reference)

    def _select_class_specific_box_deltas(self, box_deltas, labels, *, reference):
        label_values = self._label_values(labels)
        if _has_real_tensor_ops() and hasattr(box_deltas, "reshape") and hasattr(box_deltas, "shape"):
            return select_class_specific_bbox_deltas(
                box_deltas,
                label_values,
                num_classes=self.num_classes,
            )
        if hasattr(box_deltas, "reshape") and hasattr(box_deltas, "shape"):
            reshaped = box_deltas.reshape(-1, self.num_classes + 1, 4)
            row_indices = self._tensor_from_data(
                list(range(len(label_values))),
                dtype=self._long_dtype(),
                device=getattr(box_deltas, "device", None),
            )
            label_tensor = self._tensor_from_data(
                [max(0, min(int(label), self.num_classes)) for label in label_values],
                dtype=self._long_dtype(),
                device=getattr(box_deltas, "device", None),
            )
            return reshaped[row_indices, label_tensor]
        rows = self._proposal_rows(box_deltas)
        selected: list[list[float]] = []
        for row, label in zip(rows, label_values):
            label_index = max(0, min(int(label), self.num_classes))
            if row and isinstance(row[0], (list, tuple)):
                chosen = row[label_index] if label_index < len(row) else row[-1]
                selected.append([float(value) for value in chosen])
                continue
            flat = [float(value) for value in row]
            start = label_index * 4
            chosen = flat[start:start + 4] or flat[:4]
            selected.append([float(value) for value in chosen])
        return self._boxes_to_tensor(selected, reference=reference)

    def _proposal_rows(self, proposals):
        if hasattr(proposals, "tolist"):
            rows = proposals.tolist()
        else:
            rows = list(proposals)
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        return rows

    def _probability_rows(self, values):
        if hasattr(values, "tolist"):
            rows = values.tolist()
        else:
            rows = list(values)
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        return rows

    def _foreground_score(self, row):
        row = [float(value) for value in row]
        if len(row) <= 1:
            return row[0] if row else 0.0
        return max(row[1:])

    def _foreground_label_and_score(self, row):
        row = [float(value) for value in row]
        if len(row) <= 1:
            return (row[0] if row else 0.0, 0)
        foreground = row[1:]
        score = max(foreground)
        label = foreground.index(score) + 1
        return score, label

    def _softmax(self, row):
        row = [float(value) for value in row]
        if not row:
            return []
        baseline = max(row)
        weights = [math.exp(value - baseline) for value in row]
        total = sum(weights) or 1.0
        return [weight / total for weight in weights]

    def _softmax_logits(self, logits):
        if hasattr(torch, "softmax") and hasattr(logits, "shape"):
            return torch.softmax(logits, dim=-1)
        rows = self._proposal_probabilities(logits)
        if not rows:
            return []
        if len(rows) == 1:
            return rows[0]
        return rows

    def _is_integer_reference(self, reference):
        dtype = getattr(reference, "dtype", None)
        return dtype == self._long_dtype() if dtype is not None else False

    def _centered_box(self, width, height, scale, center_x, center_y):
        box_width = max(1.0, float(width) * float(scale))
        box_height = max(1.0, float(height) * float(scale))
        x_center = float(width) * float(center_x)
        y_center = float(height) * float(center_y)
        x1 = x_center - box_width / 2.0
        y1 = y_center - box_height / 2.0
        x2 = x1 + box_width
        y2 = y1 + box_height
        return self._clip_box((x1, y1, x2, y2), width, height)

    def _clip_box(self, box, width, height):
        x1, y1, x2, y2 = box
        limit_x = float(max(0, width - 1))
        limit_y = float(max(0, height - 1))
        x1 = min(max(0.0, float(x1)), limit_x)
        y1 = min(max(0.0, float(y1)), limit_y)
        x2 = min(max(float(x2), x1 + 1.0), limit_x)
        y2 = min(max(float(y2), y1 + 1.0), limit_y)
        return (x1, y1, x2, y2)

    def _box_iou(self, a, b):
        ax1, ay1, ax2, ay2 = self._box_to_tuple(a)
        bx1, by1, bx2, by2 = self._box_to_tuple(b)
        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)
        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            return 0.0
        intersection = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _binary_cross_entropy_with_logits(self, logits, targets):
        if hasattr(F, "binary_cross_entropy_with_logits"):
            return F.binary_cross_entropy_with_logits(logits, targets)
        return self._zero_loss_like(logits)

    def _cross_entropy_loss(self, logits, targets):
        if hasattr(F, "cross_entropy"):
            return F.cross_entropy(logits, targets)
        return self._zero_loss_like(logits)

    def _smooth_l1_loss(self, prediction, target):
        if hasattr(F, "smooth_l1_loss"):
            return F.smooth_l1_loss(prediction, target, reduction="mean")
        return self._zero_loss_like(prediction)

    def _zero_loss_like(self, reference):
        if hasattr(reference, "new_zeros"):
            return reference.new_zeros(())
        return 0.0

    def _float_dtype(self):
        return getattr(torch, "float32", None)

    def _long_dtype(self):
        return getattr(torch, "long", None)

    def _tensor_from_scalar(self, value, **kwargs):
        return self._tensor_from_data([value], **kwargs)

    def _tensor_from_data(self, value, **kwargs):
        if hasattr(torch, "tensor"):
            return torch.tensor(value, **kwargs)
        return value

    def _zero_box(self, *, reference):
        return self._tensor_from_data(
            [0.0, 0.0, 1.0, 1.0],
            dtype=getattr(reference, "dtype", None) or self._float_dtype(),
            device=getattr(reference, "device", None),
        )


class NativeRoIModel(TwoStageDetector):
    """Backward-compatible name for the native two-stage detector."""


def _normalize_proposal_items(proposals, *, batch_size: int | None) -> tuple[Any, ...]:
    if batch_size is not None and int(batch_size) < 0:
        raise ValueError("batch_size must be non-negative.")
    if _is_single_image_proposals(proposals):
        items = (proposals,)
    else:
        items = tuple(proposals)
    if batch_size is None:
        if not items:
            return ([],)
        return items
    batch_size = int(batch_size)
    if len(items) > batch_size:
        raise ValueError(f"received {len(items)} proposal groups for batch_size={batch_size}.")
    if len(items) < batch_size:
        items = items + tuple([] for _ in range(batch_size - len(items)))
    return items


def _is_single_image_proposals(value) -> bool:
    if _is_tensor_box_matrix(value):
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        rows = list(value)
    except TypeError:
        return False
    if not rows:
        return True
    first = rows[0]
    if _is_tensor_box_matrix(first):
        return False
    if isinstance(first, (str, bytes)):
        return False
    try:
        first_values = list(first)
    except TypeError:
        return False
    return len(first_values) == 4 and all(_is_number_like(component) for component in first_values)


def _is_tensor_box_matrix(value) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) == 2 and int(shape[1]) == 4


def _is_number_like(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _as_roi_boxes(value, *, name: str, reference=None, dtype=None, device=None):
    dtype = dtype or getattr(reference, "dtype", None) or getattr(torch, "float32", None)
    device = device if device is not None else getattr(reference, "device", None)
    if _is_tensor_box_matrix(value):
        tensor = value
        if hasattr(tensor, "to"):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if device is not None:
                kwargs["device"] = device
            tensor = tensor.to(**kwargs)
    else:
        tensor = _tensor_from_any(value, dtype=dtype, device=device)
    shape = getattr(tensor, "shape", None)
    if shape is not None and len(shape) == 1 and int(shape[0]) == 0 and hasattr(tensor, "reshape"):
        tensor = tensor.reshape(0, 4)
        shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[1]) != 4:
        raise ValueError(f"{name} must have shape (N, 4).")
    if hasattr(torch, "is_floating_point") and not bool(torch.is_floating_point(tensor)):
        tensor = tensor.to(dtype=getattr(torch, "float32", None), device=device)
    return tensor


def _as_roi_labels(value, *, device=None):
    labels = _tensor_from_any(value, dtype=getattr(torch, "long", None), device=device)
    shape = getattr(labels, "shape", None)
    if shape is not None and len(shape) == 0 and hasattr(labels, "reshape"):
        labels = labels.reshape(1)
        shape = getattr(labels, "shape", None)
    if shape is not None and len(shape) == 1:
        return labels
    if _tensor_numel(labels) == 0 and hasattr(labels, "reshape"):
        return labels.reshape(0)
    raise ValueError("gt_labels must have shape (N,).")


def _tensor_from_any(value, *, dtype=None, device=None):
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    tensor_ctor = getattr(torch, "as_tensor", None) or getattr(torch, "tensor")
    return tensor_ctor(value, **kwargs)


def _empty_tensor(shape, *, reference=None, dtype=None, device=None):
    dtype = dtype or getattr(reference, "dtype", None) or getattr(torch, "float32", None)
    device = device if device is not None else getattr(reference, "device", None)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    if hasattr(torch, "empty"):
        return torch.empty(shape, **kwargs)
    tensor = torch.tensor([], **kwargs)
    return tensor.reshape(*shape) if hasattr(tensor, "reshape") else tensor


def _full_tensor(shape, fill_value, *, reference=None, dtype=None, device=None):
    dtype = dtype or getattr(reference, "dtype", None)
    device = device if device is not None else getattr(reference, "device", None)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    if hasattr(torch, "full"):
        return torch.full(shape, fill_value, **kwargs)
    return torch.tensor(_filled_list(shape, fill_value), **kwargs)


def _filled_list(shape, fill_value):
    if len(shape) == 1:
        return [fill_value for _ in range(int(shape[0]))]
    return [_filled_list(shape[1:], fill_value) for _ in range(int(shape[0]))]


def _cat_tensors(tensors: tuple[Any, ...], *, dim: int):
    if hasattr(torch, "cat"):
        return torch.cat(tensors, dim=dim)
    if dim == 0:
        rows = []
        for tensor in tensors:
            rows.extend(tensor.tolist() if hasattr(tensor, "tolist") else list(tensor))
        dtype = getattr(tensors[0], "dtype", None) if tensors else None
        return torch.tensor(rows, dtype=dtype)
    if dim == 1:
        left, right = tensors
        left_rows = left.tolist() if hasattr(left, "tolist") else list(left)
        right_rows = right.tolist() if hasattr(right, "tolist") else list(right)
        return torch.tensor(
            [list(left_row) + list(right_row) for left_row, right_row in zip(left_rows, right_rows)],
            dtype=getattr(right, "dtype", None),
        )
    raise ValueError("Only dim=0 and dim=1 concatenation are supported without torch.cat.")


def _first_feature_tensor(features):
    if hasattr(features, "values"):
        values = list(features.values())
    else:
        values = list(features)
    return values[0] if values else None


def _feature_sequence(features):
    if hasattr(features, "values"):
        return list(features.values())
    return list(features)


def _empty_roi_pool_output(features, pool):
    feature = _first_feature_tensor(features)
    output_h, output_w = _normalize_output_size(getattr(pool, "output_size", 1))
    channels = 0
    shape = getattr(feature, "shape", None)
    if shape is not None and len(shape) >= 2:
        channels = int(shape[1])
    if feature is not None and hasattr(feature, "new_zeros"):
        return feature.new_zeros((0, channels, output_h, output_w))
    return _empty_tensor((0, channels, output_h, output_w), reference=feature)


def _normalize_output_size(output_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(output_size, Sequence) and not isinstance(output_size, (str, bytes)):
        values = tuple(int(value) for value in output_size)
        if len(values) != 2:
            raise ValueError("output_size must be an int or a (height, width) pair.")
        return _positive_int(values[0], "output_size height"), _positive_int(values[1], "output_size width")
    size = _positive_int(output_size, "output_size")
    return size, size


def _matched_indices_tensor(value, *, device):
    if hasattr(value, "assigned_gt_indices"):
        value = value.assigned_gt_indices
    return _tensor_from_any(value, dtype=getattr(torch, "long", None), device=device).reshape(-1)


def _as_mask_tensor(value, *, device, dtype):
    masks = _tensor_from_any(value, dtype=dtype, device=device)
    if masks.numel() == 0:
        return masks.reshape(0, 0, 0)
    if masks.dim() == 4 and int(masks.shape[1]) == 1:
        masks = masks[:, 0]
    if masks.dim() != 3:
        raise ValueError("gt_masks must have shape (N, H, W) or (N, 1, H, W).")
    return masks


def _crop_and_resize_mask(mask, box, *, output_size: tuple[int, int]):
    height, width = int(mask.shape[-2]), int(mask.shape[-1])
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    crop = mask[top:bottom, left:right]
    if crop.numel() == 0:
        return mask.new_zeros(output_size)
    return F.interpolate(
        crop.reshape(1, 1, crop.shape[-2], crop.shape[-1]),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(output_size)


def _row_count(value) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            return 0
        return int(shape[0])
    try:
        return len(value)
    except TypeError:
        return 0


def _tensor_numel(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return len(value)
        except TypeError:
            return 1
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


def _has_real_tensor_ops() -> bool:
    return all(hasattr(torch, name) for name in ("as_tensor", "arange", "cat", "nonzero"))


def _positive_int(value: Any, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive.")
    return integer


class SparseRCNNDetector(TwoStageDetector):
    """Sparse R-CNN style detector with learned proposal boxes and features."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: nn.Module | None = None,
        box_roi_pool: nn.Module,
        num_classes: int,
        in_channels: int,
        core_spec: RoICoreSpec,
        bbox_head: nn.Module,
        num_proposals: int = 100,
        sparse_num_stages: int = 3,
        roi_sample_size: int = 2,
    ) -> None:
        proposal_count = _positive_int(num_proposals, "num_proposals")
        super().__init__(
            backbone=backbone,
            neck=neck,
            box_roi_pool=box_roi_pool,
            num_classes=num_classes,
            in_channels=in_channels,
            core_spec=core_spec,
            rpn_head=None,
            bbox_head=bbox_head,
            roi_variant="sparse_rcnn",
            proposal_source="sparse",
            cascade_num_stages=max(1, int(sparse_num_stages)),
            roi_sample_size=roi_sample_size,
        )
        self.num_proposals = proposal_count
        self.sparse_proposal_boxes = nn.Parameter(_initial_sparse_proposal_logits(proposal_count))
        self.sparse_proposal_features = nn.Parameter(torch.zeros((proposal_count, int(in_channels))))
        self._active_sparse_indices = None
        if hasattr(nn, "init"):
            nn.init.normal_(self.sparse_proposal_features, std=0.01)

    def _run_rpn_stage(self, image, features, *, target=None):
        del features, target
        self._active_sparse_indices = None
        proposals = self._sparse_proposals_for_image(image)
        proposal_logits = proposals.new_zeros((int(proposals.shape[0]),))
        proposal_rows = [tuple(float(value) for value in row.tolist()) for row in proposals.detach()]
        return proposal_rows, proposals, proposals, proposal_logits

    def _sample_training_proposals(self, proposals, anchors, proposal_scores, target):
        sampled_indices, sampled_labels = super()._sample_training_proposals(
            proposals,
            anchors,
            proposal_scores,
            target,
        )
        self._active_sparse_indices = self._sparse_index_tensor(sampled_indices, reference=proposals)
        return sampled_indices, sampled_labels

    def _run_roi_box_head(self, features, proposals, image_shapes):
        pooled = self._roi_pool(self.box_roi_pool, features, proposals, image_shapes)
        pooled = self._add_sparse_proposal_features(pooled)
        outputs = self.bbox_head(pooled)
        class_logits, box_deltas = self._unpack_bbox_outputs(outputs)
        return class_logits, box_deltas

    def _sparse_proposals_for_image(self, image):
        normalized = torch.sigmoid(self.sparse_proposal_boxes)
        x1 = torch.minimum(normalized[:, 0], normalized[:, 2])
        y1 = torch.minimum(normalized[:, 1], normalized[:, 3])
        x2 = torch.maximum(normalized[:, 0], normalized[:, 2])
        y2 = torch.maximum(normalized[:, 1], normalized[:, 3])
        normalized_boxes = torch.stack((x1, y1, x2, y2), dim=1)
        height, width = self._image_shape(image)
        scale = image.new_tensor((float(width - 1), float(height - 1), float(width - 1), float(height - 1)))
        return clip_boxes_to_image(normalized_boxes.to(device=image.device, dtype=image.dtype) * scale, (height, width))

    def _add_sparse_proposal_features(self, pooled):
        proposal_count = int(pooled.shape[0])
        indices = self._active_sparse_indices
        if indices is None:
            proposal_features = self.sparse_proposal_features[:proposal_count]
        else:
            indices = indices.to(device=self.sparse_proposal_features.device)
            if int(indices.numel()) != proposal_count:
                raise ValueError("Sparse R-CNN proposal features must match pooled proposal rows.")
            proposal_features = self.sparse_proposal_features.index_select(0, indices)
        proposal_features = proposal_features.to(device=pooled.device, dtype=pooled.dtype)
        if pooled.dim() == 4:
            return pooled + proposal_features[:, :, None, None]
        if pooled.dim() == 2:
            return pooled + proposal_features
        return pooled

    def _sparse_index_tensor(self, sampled_indices, *, reference):
        if hasattr(sampled_indices, "to"):
            indices = sampled_indices
        else:
            indices = torch.as_tensor(sampled_indices, dtype=torch.long, device=getattr(reference, "device", None))
        return indices.reshape(-1).long()


def _initial_sparse_proposal_logits(num_proposals: int):
    grid_size = int(math.ceil(math.sqrt(int(num_proposals))))
    boxes: list[tuple[float, float, float, float]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            if len(boxes) >= int(num_proposals):
                break
            center_x = (float(col) + 0.5) / float(grid_size)
            center_y = (float(row) + 0.5) / float(grid_size)
            half_size = 0.35 / float(grid_size)
            boxes.append(
                (
                    max(0.01, center_x - half_size),
                    max(0.01, center_y - half_size),
                    min(0.99, center_x + half_size),
                    min(0.99, center_y + half_size),
                )
            )
    initial = torch.tensor(boxes, dtype=torch.float32).clamp(0.01, 0.99)
    return torch.log(initial / (1.0 - initial))


def build_native_roi_backbone(
    backbone: nn.Module,
    neck: nn.Module,
    *,
    num_levels: int,
) -> NativeRoIBackbone:
    return NativeRoIBackbone(
        backbone=backbone,
        neck=neck,
        core_spec=RoICoreSpec.from_num_levels(num_levels),
    )


def build_native_roi_detector(factory_name: str, components, *, num_classes: int):
    require_dependency("torchvision", "native roi")
    from torchvision.ops import MultiScaleRoIAlign

    normalized_factory = _normalize_roi_factory_name(factory_name)
    core_spec = RoICoreSpec.from_num_levels(components.neck_spec.num_outs)
    featmap_names = list(core_spec.featmap_names)
    kwargs = {
        "backbone": components.backbone,
        "neck": components.neck,
        "num_classes": int(num_classes),
        "in_channels": int(components.neck_spec.out_channels),
        "core_spec": core_spec,
        "rpn_head": components.rpn_head,
        "bbox_head": components.bbox_head,
        "mask_head": components.mask_head,
        "grid_head": components.grid_head,
        "roi_variant": normalized_factory,
        "proposal_source": "external" if normalized_factory == "fast_rcnn" else "rpn",
        "box_roi_pool": MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=core_spec.box_output_size,
            sampling_ratio=core_spec.sampling_ratio,
        ),
    }
    if normalized_factory == "sparse_rcnn":
        overrides = getattr(components.plan, "overrides", {}) or {}
        if components.bbox_head is None:
            raise ValueError("sparse_rcnn detector assembly requires a native SparseRoIHead.")
        return SparseRCNNDetector(
            backbone=components.backbone,
            neck=components.neck,
            box_roi_pool=kwargs["box_roi_pool"],
            num_classes=int(num_classes),
            in_channels=int(components.neck_spec.out_channels),
            core_spec=core_spec,
            bbox_head=components.bbox_head,
            num_proposals=int(overrides.get("num_proposals", overrides.get("num_queries", 100))),
            sparse_num_stages=int(overrides.get("num_stages", 3)),
            roi_sample_size=int(overrides.get("roi_sample_size", 2)),
        )
    if normalized_factory in {"cascade_rcnn", "cascade_mask_rcnn"}:
        kwargs["cascade_num_stages"] = 3
    if normalized_factory == "libra_rcnn":
        kwargs["roi_sample_size"] = 4
    if normalized_factory == "grid_rcnn":
        kwargs["grid_size"] = 7
    if normalized_factory in {"mask_rcnn", "cascade_mask_rcnn"}:
        kwargs["mask_roi_pool"] = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=core_spec.mask_output_size,
            sampling_ratio=core_spec.sampling_ratio,
        )
    if normalized_factory == "grid_rcnn":
        kwargs["grid_roi_pool"] = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=core_spec.mask_output_size,
            sampling_ratio=core_spec.sampling_ratio,
        )
    return TwoStageDetector(**kwargs)


def _normalize_roi_factory_name(factory_name: str) -> str:
    compact = "".join(char for char in str(factory_name).strip().lower() if char.isalnum())
    if compact == "fastrcnn":
        return "fast_rcnn"
    if compact == "fasterrcnn":
        return "faster_rcnn"
    if compact == "maskrcnn":
        return "mask_rcnn"
    if compact == "gridrcnn":
        return "grid_rcnn"
    if compact == "cascadercnn":
        return "cascade_rcnn"
    if compact == "cascademaskrcnn":
        return "cascade_mask_rcnn"
    if compact == "librarcnn":
        return "libra_rcnn"
    if compact == "doubleheadrcnn":
        return "double_head_rcnn"
    if compact == "dynamicrcnn":
        return "dynamic_rcnn"
    if compact == "sparsercnn":
        return "sparse_rcnn"
    return str(factory_name).strip().lower().replace("-", "_")


def _normalize_feature_pyramid(pyramid, featmap_names):
    if hasattr(pyramid, "items"):
        mapping = dict(pyramid.items())
        if all(name in mapping for name in featmap_names):
            return OrderedDict((name, mapping[name]) for name in featmap_names)
        values = list(mapping.values())
    else:
        values = list(pyramid)
    return OrderedDict(
        (name, feature)
        for name, feature in zip(featmap_names, values)
    )
