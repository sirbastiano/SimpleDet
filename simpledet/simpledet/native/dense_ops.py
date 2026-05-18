"""Dense detection ops for the native backend."""

from __future__ import annotations

from ..detectors._deps import require_dependency

require_dependency("torch", "native dense ops")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class DenseRetinaNetDecoder(nn.Module):
    def __init__(
        self,
        *,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.detections_per_img = int(detections_per_img)

    def forward(self, image, feature_maps, head_outputs):
        from torchvision.models.detection._utils import BoxCoder
        from torchvision.models.detection.image_list import ImageList
        from torchvision.ops import batched_nms

        box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        anchors = build_anchor_generator(len(feature_maps))(
            ImageList(image.unsqueeze(0), [tuple(image.shape[-2:])]),
            list(feature_maps),
        )[0]
        cls_logits = flatten_cls_logits(head_outputs["cls_logits"])
        bbox_regression = flatten_bbox_regression(head_outputs["bbox_regression"])
        scores, labels = torch.sigmoid(cls_logits).max(dim=1)
        decoded = box_coder.decode_single(bbox_regression, anchors)

        keep = scores >= self.score_threshold
        decoded = decoded[keep]
        scores = scores[keep]
        labels = labels[keep] + 1
        if decoded.numel() == 0:
            return {
                "boxes": decoded.view(0, 4),
                "scores": scores,
                "labels": labels,
            }

        keep_idx = batched_nms(decoded, scores, labels, self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return {
            "boxes": decoded[keep_idx],
            "scores": scores[keep_idx],
            "labels": labels[keep_idx],
        }


class DenseRetinaNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_cls = torch.tensor(0.0, device=images[0].device)
        total_box = torch.tensor(0.0, device=images[0].device)

        for image, target, feature_maps, head_outputs in zip(
            images,
            targets,
            feature_pyramids,
            head_outputs_per_image,
        ):
            from torchvision.models.detection.image_list import ImageList

            anchors = build_anchor_generator(len(feature_maps))(
                ImageList(image.unsqueeze(0), [tuple(image.shape[-2:])]),
                list(feature_maps),
            )[0]
            cls_logits = flatten_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_bbox_regression(head_outputs["bbox_regression"])

            matched_boxes, matched_labels, positive_mask = match_anchors(anchors, target)
            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = matched_labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0

            total_cls = total_cls + F.binary_cross_entropy_with_logits(
                cls_logits,
                class_targets,
                reduction="mean",
            )

            if positive_mask.any():
                regression_targets = encode_boxes(anchors[positive_mask], matched_boxes[positive_mask])
                total_box = total_box + F.smooth_l1_loss(
                    bbox_regression[positive_mask],
                    regression_targets,
                    reduction="mean",
                )

        num_images = max(len(images), 1)
        loss_cls = total_cls / num_images
        loss_bbox = total_box / num_images
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_total": loss_cls + loss_bbox,
        }


class DenseFCOSDecoder(nn.Module):
    def __init__(
        self,
        *,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.detections_per_img = int(detections_per_img)

    def forward(self, image, feature_maps, head_outputs):
        from torchvision.ops import batched_nms

        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        if "centerness" in head_outputs:
            centerness = head_outputs["centerness"]
        else:
            centerness = [torch.ones_like(level[:, :1]) for level in cls_logits]

        boxes_per_level = []
        scores_per_level = []
        labels_per_level = []
        for feature, logits, bbox, center in zip(feature_maps, cls_logits, bbox_regression, centerness):
            points = build_fcos_points(feature)
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
            bbox_flat = bbox.permute(0, 2, 3, 1).reshape(-1, 4)
            center_flat = center.permute(0, 2, 3, 1).reshape(-1)
            class_scores = torch.sigmoid(logits_flat)
            center_scores = torch.sigmoid(center_flat)
            scores, labels = (class_scores * center_scores.unsqueeze(1)).max(dim=1)
            keep = scores >= self.score_threshold
            if not keep.any():
                continue
            boxes_per_level.append(decode_fcos_boxes(points[keep], bbox_flat[keep]))
            scores_per_level.append(scores[keep])
            labels_per_level.append(labels[keep] + 1)

        if not boxes_per_level:
            empty = image.new_zeros((0, 4))
            return {
                "boxes": empty,
                "scores": image.new_zeros((0,)),
                "labels": image.new_zeros((0,), dtype=torch.long),
            }

        boxes = torch.cat(boxes_per_level, dim=0)
        scores = torch.cat(scores_per_level, dim=0)
        labels = torch.cat(labels_per_level, dim=0)
        keep_idx = batched_nms(boxes, scores, labels, self.nms_threshold)[: self.detections_per_img]
        return {
            "boxes": boxes[keep_idx],
            "scores": scores[keep_idx],
            "labels": labels[keep_idx],
        }


class DenseFCOSLoss(nn.Module):
    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_cls = torch.tensor(0.0, device=images[0].device)
        total_box = torch.tensor(0.0, device=images[0].device)
        total_ctr = torch.tensor(0.0, device=images[0].device)

        for target, feature_maps, head_outputs in zip(targets, feature_pyramids, head_outputs_per_image):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            for feature, logits, bbox, center in zip(
                feature_maps,
                head_outputs["cls_logits"],
                head_outputs["bbox_regression"],
                head_outputs.get("centerness", []),
            ):
                points = build_fcos_points(feature)
                cls_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                bbox_reg = bbox.permute(0, 2, 3, 1).reshape(-1, 4)
                centerness = center.permute(0, 2, 3, 1).reshape(-1)

                matched_boxes, matched_labels, positive_mask = match_points_to_boxes(points, gt_boxes, gt_labels)
                class_targets = torch.zeros_like(cls_logits)
                if positive_mask.any():
                    positive_labels = matched_labels[positive_mask].long().clamp(min=1) - 1
                    class_targets[positive_mask, positive_labels] = 1.0
                total_cls = total_cls + F.binary_cross_entropy_with_logits(cls_logits, class_targets, reduction="mean")

                if positive_mask.any():
                    regression_targets = encode_fcos_boxes(points[positive_mask], matched_boxes[positive_mask])
                    total_box = total_box + F.l1_loss(bbox_reg[positive_mask], regression_targets, reduction="mean")
                    total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                        centerness[positive_mask],
                        torch.ones_like(centerness[positive_mask]),
                        reduction="mean",
                    )

        num_images = max(len(images), 1)
        loss_cls = total_cls / num_images
        loss_bbox = total_box / num_images
        loss_ctr = total_ctr / num_images
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_centerness": loss_ctr,
            "loss_total": loss_cls + loss_bbox + loss_ctr,
        }


class DenseATSSDecoder(nn.Module):
    def __init__(
        self,
        *,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.detections_per_img = int(detections_per_img)

    def forward(self, image, feature_maps, head_outputs):
        from torchvision.models.detection._utils import BoxCoder
        from torchvision.models.detection.image_list import ImageList
        from torchvision.ops import batched_nms

        box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        anchors = build_anchor_generator(len(feature_maps))(
            ImageList(image.unsqueeze(0), [tuple(image.shape[-2:])]),
            list(feature_maps),
        )[0]
        cls_logits = flatten_anchor_cls_logits(head_outputs["cls_logits"])
        bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])
        centerness = flatten_anchor_centerness_logits(head_outputs["centerness"])
        class_scores = torch.sigmoid(cls_logits)
        center_scores = torch.sigmoid(centerness).unsqueeze(1)
        scores, labels = (class_scores * center_scores).max(dim=1)
        decoded = box_coder.decode_single(bbox_regression, anchors)

        keep = scores >= self.score_threshold
        decoded = decoded[keep]
        scores = scores[keep]
        labels = labels[keep] + 1
        if decoded.numel() == 0:
            return {
                "boxes": decoded.view(0, 4),
                "scores": scores,
                "labels": labels,
            }

        keep_idx = batched_nms(decoded, scores, labels, self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return {
            "boxes": decoded[keep_idx],
            "scores": scores[keep_idx],
            "labels": labels[keep_idx],
        }


class DenseATSSLoss(nn.Module):
    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_cls = torch.tensor(0.0, device=images[0].device)
        total_box = torch.tensor(0.0, device=images[0].device)
        total_ctr = torch.tensor(0.0, device=images[0].device)

        for image, target, feature_maps, head_outputs in zip(
            images,
            targets,
            feature_pyramids,
            head_outputs_per_image,
        ):
            from torchvision.models.detection.image_list import ImageList

            anchors = build_anchor_generator(len(feature_maps))(
                ImageList(image.unsqueeze(0), [tuple(image.shape[-2:])]),
                list(feature_maps),
            )[0]
            cls_logits = flatten_anchor_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])
            centerness = flatten_anchor_centerness_logits(head_outputs["centerness"])

            matched_boxes, matched_labels, positive_mask = match_anchors(anchors, target)
            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = matched_labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0

            total_cls = total_cls + F.binary_cross_entropy_with_logits(
                cls_logits,
                class_targets,
                reduction="mean",
            )

            if positive_mask.any():
                regression_targets = encode_boxes(anchors[positive_mask], matched_boxes[positive_mask])
                total_box = total_box + F.smooth_l1_loss(
                    bbox_regression[positive_mask],
                    regression_targets,
                    reduction="mean",
                )
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    centerness[positive_mask],
                    torch.ones_like(centerness[positive_mask]),
                    reduction="mean",
                )

        num_images = max(len(images), 1)
        loss_cls = total_cls / num_images
        loss_bbox = total_box / num_images
        loss_ctr = total_ctr / num_images
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_centerness": loss_ctr,
            "loss_total": loss_cls + loss_bbox + loss_ctr,
        }


class DenseGFLDecoder(DenseATSSDecoder):
    """GFL currently reuses the anchor-based ATSS decode path."""


class DenseGFLLoss(DenseATSSLoss):
    """GFL currently reuses the anchor-based ATSS loss path."""


def build_anchor_generator(num_levels: int):
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    base_sizes = [32, 64, 128, 256, 512]
    sizes = tuple((base_sizes[min(index, len(base_sizes) - 1)],) for index in range(num_levels))
    aspect_ratios = tuple((0.5, 1.0, 2.0) for _ in range(num_levels))
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def flatten_cls_logits(logits_per_level):
    flattened = []
    for level in logits_per_level:
        flattened.append(level.permute(0, 2, 3, 1).reshape(-1, level.shape[1]))
    return torch.cat(flattened, dim=0)


def flatten_bbox_regression(regression_per_level):
    flattened = []
    for level in regression_per_level:
        flattened.append(level.permute(0, 2, 3, 1).reshape(-1, 4))
    return torch.cat(flattened, dim=0)


def flatten_centerness_logits(centerness_per_level):
    flattened = []
    for level in centerness_per_level:
        flattened.append(level.permute(0, 2, 3, 1).reshape(-1))
    return torch.cat(flattened, dim=0)


def flatten_anchor_cls_logits(logits_per_level):
    flattened = []
    for level in logits_per_level:
        batch, channels, height, width = level.shape
        num_anchors = 9 if channels % 9 == 0 else 1
        num_classes = channels // max(num_anchors, 1)
        reshaped = level.view(batch, num_anchors, num_classes, height, width)
        flattened.append(reshaped.permute(0, 3, 4, 1, 2).reshape(-1, num_classes))
    return torch.cat(flattened, dim=0)


def flatten_anchor_bbox_regression(regression_per_level):
    flattened = []
    for level in regression_per_level:
        batch, channels, height, width = level.shape
        num_anchors = channels // 4
        reshaped = level.view(batch, num_anchors, 4, height, width)
        flattened.append(reshaped.permute(0, 3, 4, 1, 2).reshape(-1, 4))
    return torch.cat(flattened, dim=0)


def flatten_anchor_centerness_logits(centerness_per_level):
    flattened = []
    for level in centerness_per_level:
        batch, channels, height, width = level.shape
        reshaped = level.view(batch, channels, height, width)
        flattened.append(reshaped.permute(0, 2, 3, 1).reshape(-1))
    return torch.cat(flattened, dim=0)


def match_anchors(anchors, target):
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    if gt_boxes.numel() == 0:
        zeros_boxes = torch.zeros_like(anchors)
        zeros_labels = torch.zeros((anchors.shape[0],), dtype=torch.long, device=anchors.device)
        positive_mask = torch.zeros((anchors.shape[0],), dtype=torch.bool, device=anchors.device)
        return zeros_boxes, zeros_labels, positive_mask

    anchor_centers_x = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_centers_y = (anchors[:, 1] + anchors[:, 3]) / 2.0
    in_box = (
        (anchor_centers_x[:, None] >= gt_boxes[None, :, 0])
        & (anchor_centers_x[:, None] <= gt_boxes[None, :, 2])
        & (anchor_centers_y[:, None] >= gt_boxes[None, :, 1])
        & (anchor_centers_y[:, None] <= gt_boxes[None, :, 3])
    )
    positive_mask = in_box.any(dim=1)
    matched_indices = in_box.float().argmax(dim=1)
    matched_boxes = gt_boxes[matched_indices]
    matched_labels = gt_labels[matched_indices]
    return matched_boxes, matched_labels, positive_mask


def encode_boxes(anchors, gt_boxes):
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    eps = torch.finfo(anchor_widths.dtype).eps
    anchor_widths = anchor_widths.clamp(min=eps)
    anchor_heights = anchor_heights.clamp(min=eps)
    gt_widths = gt_widths.clamp(min=eps)
    gt_heights = gt_heights.clamp(min=eps)

    dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    dw = torch.log(gt_widths / anchor_widths)
    dh = torch.log(gt_heights / anchor_heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def build_fcos_points(feature_map):
    _, _, height, width = feature_map.shape
    device = feature_map.device
    dtype = feature_map.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1)


def match_points_to_boxes(points, gt_boxes, gt_labels):
    if gt_boxes.numel() == 0:
        zeros_boxes = torch.zeros((points.shape[0], 4), dtype=points.dtype, device=points.device)
        zeros_labels = torch.zeros((points.shape[0],), dtype=torch.long, device=points.device)
        positive_mask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        return zeros_boxes, zeros_labels, positive_mask

    in_box = (
        (points[:, None, 0] >= gt_boxes[None, :, 0])
        & (points[:, None, 0] <= gt_boxes[None, :, 2])
        & (points[:, None, 1] >= gt_boxes[None, :, 1])
        & (points[:, None, 1] <= gt_boxes[None, :, 3])
    )
    box_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    candidate_areas = torch.where(in_box, box_areas.unsqueeze(0), torch.full_like(in_box.float(), float("inf")))
    matched_indices = candidate_areas.argmin(dim=1)
    positive_mask = in_box.any(dim=1)
    matched_boxes = gt_boxes[matched_indices]
    matched_labels = gt_labels[matched_indices]
    return matched_boxes, matched_labels, positive_mask


def encode_fcos_boxes(points, gt_boxes):
    left = points[:, 0] - gt_boxes[:, 0]
    top = points[:, 1] - gt_boxes[:, 1]
    right = gt_boxes[:, 2] - points[:, 0]
    bottom = gt_boxes[:, 3] - points[:, 1]
    return torch.stack((left, top, right, bottom), dim=1).clamp(min=0)


def decode_fcos_boxes(points, deltas):
    x1 = points[:, 0] - deltas[:, 0]
    y1 = points[:, 1] - deltas[:, 1]
    x2 = points[:, 0] + deltas[:, 2]
    y2 = points[:, 1] + deltas[:, 3]
    return torch.stack((x1, y1, x2, y2), dim=1)
