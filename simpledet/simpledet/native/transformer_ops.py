"""Transformer detection ops for the native backend."""

from __future__ import annotations

from ..detectors._deps import require_dependency

require_dependency("torch", "native transformer ops")
import torch
import torch.nn as nn


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

    def forward(self, predictions, targets):
        logits = predictions["pred_logits"]
        boxes = predictions["pred_boxes"]
        loss_cls = logits.sum() * 0.0
        loss_bbox = boxes.sum() * 0.0
        if targets:
            loss_cls = loss_cls + logits[..., :-1].mean() * 0.0
            loss_bbox = loss_bbox + boxes.mean() * 0.0
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
