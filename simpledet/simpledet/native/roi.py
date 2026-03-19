"""Native ROI-core helpers for the Lightning backend."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency

require_dependency("torch", "native roi")
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NativeRoIModel(nn.Module):
    """Compact native ROI detector used for Faster and Mask R-CNN variants."""

    def __init__(
        self,
        *,
        backbone: NativeRoIBackbone,
        box_roi_pool: nn.Module,
        num_classes: int,
        in_channels: int,
        core_spec: RoICoreSpec,
        mask_roi_pool: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.box_roi_pool = box_roi_pool
        self.mask_roi_pool = mask_roi_pool
        self.num_classes = int(num_classes)
        self.core_spec = core_spec
        self.with_mask = mask_roi_pool is not None
        self.proposal_iou_threshold = 0.5
        self.roi_sample_size = 2
        self.postprocess_topk = 4
        hidden_channels = max(128, int(in_channels))
        self.proposal_head = nn.Sequential(
            nn.Linear(int(in_channels), hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.proposal_objectness = nn.Linear(hidden_channels, 1)
        self.proposal_regressor = nn.Linear(hidden_channels, 4)
        self.box_head = nn.Sequential(
            nn.Linear(int(in_channels), hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.box_classifier = nn.Linear(hidden_channels, self.num_classes + 1)
        self.box_regressor = nn.Linear(hidden_channels, (self.num_classes + 1) * 4)
        self.mask_head = None
        self.mask_predictor = None
        if self.with_mask:
            self.mask_head = nn.Sequential(
                nn.Linear(int(in_channels), hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
            self.mask_predictor = nn.Linear(hidden_channels, 1)

    def forward(self, images, targets=None):
        self._validate_inputs(images, targets=targets)
        detections: list[dict[str, Any]] = []
        losses: list[dict[str, torch.Tensor]] = []
        for image_index, image in enumerate(images):
            features = self.backbone(image.unsqueeze(0))
            anchors, anchor_tensor, refined_proposals, proposal_logits = self._run_rpn_stage(image, features)
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
                class_logits, box_deltas = self._run_roi_box_head(
                    features,
                    [sampled_proposals],
                    image_shapes,
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
                    )
                )
                continue

            class_logits, box_deltas = self._run_roi_box_head(
                features,
                [refined_proposals],
                image_shapes,
            )
            boxes, scores, labels, detection_indices = self._postprocess_detections(
                refined_proposals,
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
                mask_proposals = self._select_rows(refined_proposals, detection_indices, reference=boxes)
                mask_features = self._roi_pool(
                    self.mask_roi_pool,
                    features,
                    [mask_proposals],
                    image_shapes,
                )
                mask_features = mask_features.mean(dim=(-1, -2))
                mask_representation = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_representation)
                detection["masks"] = torch.sigmoid(mask_logits).reshape(-1, 1, 1, 1)

            detections.append(detection)

        if targets is not None:
            return self._merge_losses(losses)
        return detections

    def _validate_inputs(self, images, *, targets=None):
        if not images:
            raise ValueError("NativeRoIModel requires at least one image.")
        for image in images:
            if not hasattr(image, "shape") or not hasattr(image, "unsqueeze"):
                raise TypeError(
                    "NativeRoIModel expects tensor-like images with 'shape' and 'unsqueeze'."
                )
        if targets is not None and len(targets) != len(images):
            raise ValueError("Number of targets must match number of images.")

    def _loss_from_targets(
        self,
        proposal_logits,
        anchors,
        class_logits,
        box_deltas,
        sampled_proposals,
        sampled_labels,
        target,
    ):
        target_box = self._coerce_box(self._first_target_box(target, anchors[0]), reference=class_logits)
        rpn_losses = self._rpn_losses(proposal_logits, anchors, target_box)
        roi_losses = self._roi_losses(
            sampled_proposals=sampled_proposals,
            sampled_labels=sampled_labels,
            class_logits=class_logits,
            box_deltas=box_deltas,
            target_box=target_box,
        )
        losses = {**rpn_losses, **roi_losses}
        if self.with_mask:
            losses["loss_mask"] = self._zero_loss_like(roi_losses["loss_roi_classifier"])
        losses["loss_total"] = sum(losses.values())
        return losses

    def _first_target_box(self, target, proposal):
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            return target["boxes"][0]
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

    def _run_rpn_stage(self, image, features):
        anchors = self._generate_anchor_proposals(image, features)
        anchor_tensor = self._boxes_to_tensor(anchors, reference=image)
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
        pooled = pooled.mean(dim=(-1, -2))
        representation = self.box_head(pooled)
        class_logits = self.box_classifier(representation)
        box_deltas = self.box_regressor(representation)
        return class_logits, box_deltas

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

    def _image_shape(self, image):
        return int(image.shape[-2]), int(image.shape[-1])

    def _roi_pool(self, pool, features, proposals, image_shapes):
        return pool(features, proposals, image_shapes)

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
        kwargs = {}
        dtype = getattr(reference, "dtype", None) or self._float_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype
        device = getattr(reference, "device", None)
        if device is not None:
            kwargs["device"] = device
        return self._tensor_from_data(proposals, **kwargs)

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

    normalized_factory = str(factory_name).strip().lower()
    core_spec = RoICoreSpec.from_num_levels(components.neck_spec.num_outs)
    roi_backbone = build_native_roi_backbone(
        components.backbone,
        components.neck,
        num_levels=components.neck_spec.num_outs,
    )
    featmap_names = list(core_spec.featmap_names)
    kwargs = {
        "backbone": roi_backbone,
        "num_classes": int(num_classes),
        "in_channels": int(components.neck_spec.out_channels),
        "core_spec": core_spec,
        "box_roi_pool": MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=core_spec.box_output_size,
            sampling_ratio=core_spec.sampling_ratio,
        ),
    }
    if normalized_factory == "mask_rcnn":
        kwargs["mask_roi_pool"] = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=core_spec.mask_output_size,
            sampling_ratio=core_spec.sampling_ratio,
        )
    return NativeRoIModel(**kwargs)


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
