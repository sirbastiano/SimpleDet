"""Transformer detection ops for the native backend."""

from __future__ import annotations

from ..detectors._deps import require_dependency

require_dependency("torch", "native transformer ops")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .assignment import HungarianAssigner  # noqa: E402


class NativeDetrDecoder(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, num_queries: int = 100) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.projection = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.query_embed = nn.Embedding(self.num_queries, self.in_channels)
        self.class_head = nn.Linear(self.in_channels, self.num_classes + 1)
        self.box_head = nn.Linear(self.in_channels, 4)

    def forward(self, features):
        feature = features[-1]
        projected = self.projection(feature)
        pooled = projected.mean(dim=(-2, -1))
        query_features = self.query_embed.weight.unsqueeze(0) + pooled.unsqueeze(1)
        pred_logits = self.class_head(query_features)
        pred_boxes = torch.sigmoid(self.box_head(query_features))
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }


class NativeDetrLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matcher = HungarianAssigner()

    def forward(self, predictions, targets):
        logits = predictions["pred_logits"]
        boxes = predictions["pred_boxes"]
        loss_cls = logits.sum() * 0.0
        loss_bbox = boxes.sum() * 0.0
        for image_index, target in enumerate(targets or []):
            pred_logits = logits[image_index]
            pred_boxes = boxes[image_index]
            gt_boxes = _normalize_target_boxes(target.get("boxes", []), pred_boxes, target)
            gt_labels = torch.as_tensor(target.get("labels", []), dtype=torch.long, device=pred_boxes.device)
            assignment = self.matcher(pred_logits, pred_boxes, gt_boxes, gt_labels)

            background_index = pred_logits.shape[-1] - 1
            class_targets = torch.full(
                (pred_logits.shape[0],),
                background_index,
                dtype=torch.long,
                device=pred_logits.device,
            )
            if assignment.positive_mask.any():
                class_targets[assignment.positive_mask] = (assignment.labels[assignment.positive_mask] - 1).clamp(
                    min=0,
                    max=background_index - 1,
                )
            loss_cls = loss_cls + F.cross_entropy(pred_logits, class_targets)
            if assignment.positive_mask.any():
                loss_bbox = loss_bbox + F.l1_loss(
                    pred_boxes[assignment.positive_mask],
                    assignment.matched_boxes[assignment.positive_mask],
                    reduction="mean",
                )
        if targets:
            count = max(len(targets), 1)
            loss_cls = loss_cls / count
            loss_bbox = loss_bbox / count
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_total": loss_cls + loss_bbox,
        }


class NativeDetrPostProcessor(nn.Module):
    def __init__(self, *, score_threshold: float = 0.05, detections_per_img: int = 100) -> None:
        super().__init__()
        self.score_threshold = float(score_threshold)
        self.detections_per_img = int(detections_per_img)

    def forward(self, prediction):
        logits = prediction["pred_logits"][0]
        boxes = prediction["pred_boxes"][0]
        scores = torch.softmax(logits, dim=-1)[..., :-1]
        max_scores, labels = scores.max(dim=-1)
        keep = max_scores >= self.score_threshold
        boxes = boxes[keep][: self.detections_per_img]
        max_scores = max_scores[keep][: self.detections_per_img]
        labels = labels[keep][: self.detections_per_img] + 1
        if boxes.numel() == 0:
            return {
                "boxes": boxes.view(0, 4),
                "scores": max_scores,
                "labels": labels,
            }
        return {
            "boxes": boxes,
            "scores": max_scores,
            "labels": labels,
        }


def _normalize_target_boxes(boxes, reference_boxes, target):
    boxes = torch.as_tensor(boxes, dtype=reference_boxes.dtype, device=reference_boxes.device)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError("DETR targets must provide boxes with shape (N, 4).")
    height = target.get("height") or target.get("image_height")
    width = target.get("width") or target.get("image_width")
    image_size = target.get("image_size") or target.get("size")
    if image_size is not None and height is None and width is None:
        height, width = image_size
    if height is not None and width is not None:
        scale = boxes.new_tensor([float(width), float(height), float(width), float(height)]).clamp(min=1.0)
        return (boxes / scale).clamp(min=0.0, max=1.0)
    if float(boxes.max()) <= 1.0:
        return boxes.clamp(min=0.0, max=1.0)
    scale_x = boxes[:, (0, 2)].max().clamp(min=1.0)
    scale_y = boxes[:, (1, 3)].max().clamp(min=1.0)
    scale = torch.stack((scale_x, scale_y, scale_x, scale_y))
    return (boxes / scale).clamp(min=0.0, max=1.0)
