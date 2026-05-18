"""Dense detection ops for the native backend."""

from __future__ import annotations

from ..detectors._deps import require_dependency

require_dependency("torch", "native dense ops")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .geometry import (  # noqa: E402
    DEFAULT_ANCHOR_SIZES,
    build_feature_map_specs,
    clip_boxes_to_image,
    decode_boxes,
    decode_point_boxes,
    encode_boxes,
    encode_point_boxes,
    generate_anchors,
    generate_points,
    make_batched_nms_payload,
    prediction_payload_to_dict,
    select_prediction_payload,
)


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
        from torchvision.ops import batched_nms

        feature_maps = _feature_sequence(feature_maps)
        image_size = _image_size(image)
        anchors = _anchor_priors_for_image(image, feature_maps)
        cls_logits = flatten_cls_logits(head_outputs["cls_logits"])
        bbox_regression = flatten_bbox_regression(head_outputs["bbox_regression"])
        scores, labels = torch.sigmoid(cls_logits).max(dim=1)
        decoded = clip_boxes_to_image(decode_boxes(anchors, bbox_regression), image_size)

        keep = scores >= self.score_threshold
        payload = make_batched_nms_payload(decoded[keep], scores[keep], labels[keep] + 1)
        if payload["boxes"].numel() == 0:
            return prediction_payload_to_dict(payload)

        keep_idx = batched_nms(payload["boxes"], payload["scores"], payload["nms_indices"], self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return prediction_payload_to_dict(select_prediction_payload(payload, keep_idx))


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
            feature_maps = _feature_sequence(feature_maps)
            anchors = _anchor_priors_for_image(image, feature_maps)
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

        feature_maps = _feature_sequence(feature_maps)
        image_size = _image_size(image)
        feature_specs = build_feature_map_specs(feature_maps, image_size=image_size)
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        if "centerness" in head_outputs:
            centerness = head_outputs["centerness"]
        else:
            centerness = [torch.ones_like(level[:, :1]) for level in cls_logits]

        boxes_per_level = []
        scores_per_level = []
        labels_per_level = []
        points_per_level = generate_points(feature_specs, device=feature_maps[0].device, dtype=feature_maps[0].dtype)
        for points, logits, bbox, center in zip(points_per_level, cls_logits, bbox_regression, centerness):
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
            bbox_flat = bbox.permute(0, 2, 3, 1).reshape(-1, 4)
            center_flat = center.permute(0, 2, 3, 1).reshape(-1)
            class_scores = torch.sigmoid(logits_flat)
            center_scores = torch.sigmoid(center_flat)
            scores, labels = (class_scores * center_scores.unsqueeze(1)).max(dim=1)
            keep = scores >= self.score_threshold
            if not keep.any():
                continue
            boxes_per_level.append(clip_boxes_to_image(decode_point_boxes(points[keep], bbox_flat[keep]), image_size))
            scores_per_level.append(scores[keep])
            labels_per_level.append(labels[keep] + 1)

        if not boxes_per_level:
            return prediction_payload_to_dict(
                make_batched_nms_payload(
                    image.new_zeros((0, 4)),
                    image.new_zeros((0,)),
                    image.new_zeros((0,), dtype=torch.long),
                )
            )

        boxes = torch.cat(boxes_per_level, dim=0)
        scores = torch.cat(scores_per_level, dim=0)
        labels = torch.cat(labels_per_level, dim=0)
        payload = make_batched_nms_payload(boxes, scores, labels)
        keep_idx = batched_nms(payload["boxes"], payload["scores"], payload["nms_indices"], self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return prediction_payload_to_dict(select_prediction_payload(payload, keep_idx))


class DenseFCOSLoss(nn.Module):
    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_cls = torch.tensor(0.0, device=images[0].device)
        total_box = torch.tensor(0.0, device=images[0].device)
        total_ctr = torch.tensor(0.0, device=images[0].device)

        for image, target, feature_maps, head_outputs in zip(images, targets, feature_pyramids, head_outputs_per_image):
            feature_maps = _feature_sequence(feature_maps)
            feature_specs = build_feature_map_specs(feature_maps, image_size=_image_size(image))
            points_per_level = generate_points(
                feature_specs,
                device=feature_maps[0].device,
                dtype=feature_maps[0].dtype,
            )
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            for points, logits, bbox, center in zip(
                points_per_level,
                head_outputs["cls_logits"],
                head_outputs["bbox_regression"],
                head_outputs.get("centerness", []),
            ):
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
                    regression_targets = encode_point_boxes(points[positive_mask], matched_boxes[positive_mask])
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
        from torchvision.ops import batched_nms

        feature_maps = _feature_sequence(feature_maps)
        image_size = _image_size(image)
        anchors = _anchor_priors_for_image(image, feature_maps)
        cls_logits = flatten_anchor_cls_logits(head_outputs["cls_logits"])
        bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])
        centerness = flatten_anchor_centerness_logits(head_outputs["centerness"])
        class_scores = torch.sigmoid(cls_logits)
        center_scores = torch.sigmoid(centerness).unsqueeze(1)
        scores, labels = (class_scores * center_scores).max(dim=1)
        decoded = clip_boxes_to_image(decode_boxes(anchors, bbox_regression), image_size)

        keep = scores >= self.score_threshold
        payload = make_batched_nms_payload(decoded[keep], scores[keep], labels[keep] + 1)
        if payload["boxes"].numel() == 0:
            return prediction_payload_to_dict(payload)

        keep_idx = batched_nms(payload["boxes"], payload["scores"], payload["nms_indices"], self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return prediction_payload_to_dict(select_prediction_payload(payload, keep_idx))


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
            feature_maps = _feature_sequence(feature_maps)
            anchors = _anchor_priors_for_image(image, feature_maps)
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


class _NativeAnchorGenerator:
    def __init__(self, num_levels: int) -> None:
        self.num_levels = int(num_levels)
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive.")
        self.base_sizes = tuple(
            DEFAULT_ANCHOR_SIZES[min(index, len(DEFAULT_ANCHOR_SIZES) - 1)]
            for index in range(self.num_levels)
        )

    def __call__(self, image_list, feature_maps):
        feature_maps = _feature_sequence(feature_maps)
        image_sizes = getattr(image_list, "image_sizes", None)
        if image_sizes is None:
            tensors = getattr(image_list, "tensors", None)
            image_sizes = [tuple(tensors.shape[-2:])] if tensors is not None else None
        if image_sizes is None:
            raise ValueError("image_list must expose image_sizes or tensors.")
        anchors = []
        for image_size in image_sizes:
            feature_specs = build_feature_map_specs(feature_maps, image_size=image_size)
            anchors.append(
                torch.cat(
                    generate_anchors(
                        feature_specs,
                        base_sizes=self.base_sizes,
                        device=feature_maps[0].device,
                        dtype=feature_maps[0].dtype,
                    ),
                    dim=0,
                )
            )
        return anchors


def build_anchor_generator(num_levels: int):
    return _NativeAnchorGenerator(num_levels)


def _feature_sequence(feature_maps):
    if isinstance(feature_maps, dict):
        return tuple(feature_maps.values())
    return tuple(feature_maps)


def _image_size(image):
    if not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("image must expose spatial height and width dimensions.")
    return tuple(int(value) for value in image.shape[-2:])


def _anchor_priors_for_image(image, feature_maps):
    feature_specs = build_feature_map_specs(feature_maps, image_size=_image_size(image))
    return torch.cat(
        generate_anchors(
            feature_specs,
            device=feature_maps[0].device,
            dtype=feature_maps[0].dtype,
        ),
        dim=0,
    )


def _is_tensor_like(value):
    return hasattr(value, "shape") and hasattr(value, "reshape") and hasattr(value, "dim")


def flatten_cls_logits(logits_per_level):
    if _is_tensor_like(logits_per_level):
        if logits_per_level.dim() == 3:
            return logits_per_level.reshape(-1, logits_per_level.shape[-1])
        if logits_per_level.dim() == 4:
            return logits_per_level.permute(0, 2, 3, 1).reshape(-1, logits_per_level.shape[1])
        raise ValueError("cls logits must be a 3D flattened tensor or 4D NCHW tensor.")
    flattened = []
    for level in logits_per_level:
        flattened.append(level.permute(0, 2, 3, 1).reshape(-1, level.shape[1]))
    return torch.cat(flattened, dim=0)


def flatten_bbox_regression(regression_per_level):
    if _is_tensor_like(regression_per_level):
        if regression_per_level.dim() == 3:
            return regression_per_level.reshape(-1, 4)
        if regression_per_level.dim() == 4:
            return regression_per_level.permute(0, 2, 3, 1).reshape(-1, 4)
        raise ValueError("bbox regression must be a 3D flattened tensor or 4D NCHW tensor.")
    flattened = []
    for level in regression_per_level:
        flattened.append(level.permute(0, 2, 3, 1).reshape(-1, 4))
    return torch.cat(flattened, dim=0)


def flatten_centerness_logits(centerness_per_level):
    if _is_tensor_like(centerness_per_level):
        if centerness_per_level.dim() == 2:
            return centerness_per_level.reshape(-1)
        if centerness_per_level.dim() == 4:
            return centerness_per_level.permute(0, 2, 3, 1).reshape(-1)
        raise ValueError("centerness logits must be a 2D flattened tensor or 4D NCHW tensor.")
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


def build_fcos_points(feature_map, *, image_size=None, stride=None):
    strides = None if stride is None else (stride,)
    if image_size is None and strides is None:
        strides = ((1.0, 1.0),)
    feature_specs = build_feature_map_specs((feature_map,), image_size=image_size, strides=strides)
    return generate_points(feature_specs, device=feature_map.device, dtype=feature_map.dtype)[0]


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
    return encode_point_boxes(points, gt_boxes)


def decode_fcos_boxes(points, deltas):
    return decode_point_boxes(points, deltas)
