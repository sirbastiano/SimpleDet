"""Dense detection ops for the native backend."""

from __future__ import annotations

from ..detectors._deps import require_dependency
from ..extensions import LOSSES

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
from .assignment import (  # noqa: E402
    atss_assign,
    center_region_assign,
    max_iou_assign,
    sim_ota_assign,
    task_aligned_assign,
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

            assignment = max_iou_assign(
                anchors,
                target["boxes"],
                target["labels"],
                ignored_boxes=_ignored_boxes_from_target(target),
            )
            matched_boxes = assignment.matched_boxes
            matched_labels = assignment.labels
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask
            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = matched_labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0

            total_cls = total_cls + _binary_cross_entropy_valid(cls_logits, class_targets, valid_mask)

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


class DenseFreeAnchorRetinaNetDecoder(DenseRetinaNetDecoder):
    """FreeAnchor uses the Retina-style dense decode contract."""


class DenseFreeAnchorRetinaNetLoss(DenseRetinaNetLoss):
    """Finite FreeAnchor smoke loss on the shared Retina dense target contract."""


class DenseRPNDecoder(nn.Module):
    def __init__(
        self,
        *,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.7,
        detections_per_img: int = 1000,
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
        objectness = flatten_anchor_objectness_logits(head_outputs["objectness_logits"])
        bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])
        decoded = clip_boxes_to_image(decode_boxes(anchors, bbox_regression), image_size)
        scores = torch.sigmoid(objectness)
        labels = torch.ones_like(scores, dtype=torch.long)

        keep = scores >= self.score_threshold
        payload = make_batched_nms_payload(decoded[keep], scores[keep], labels[keep])
        if payload["boxes"].numel() == 0:
            return prediction_payload_to_dict(payload)

        keep_idx = batched_nms(payload["boxes"], payload["scores"], payload["nms_indices"], self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return prediction_payload_to_dict(select_prediction_payload(payload, keep_idx))


class DenseRPNLoss(nn.Module):
    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_objectness = torch.tensor(0.0, device=images[0].device)
        total_box = torch.tensor(0.0, device=images[0].device)

        for image, target, feature_maps, head_outputs in zip(
            images,
            targets,
            feature_pyramids,
            head_outputs_per_image,
        ):
            feature_maps = _feature_sequence(feature_maps)
            anchors = _anchor_priors_for_image(image, feature_maps)
            objectness = flatten_anchor_objectness_logits(head_outputs["objectness_logits"])
            bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])

            labels = target["labels"].new_ones(target["labels"].shape)
            assignment = max_iou_assign(
                anchors,
                target["boxes"],
                labels,
                ignored_boxes=_ignored_boxes_from_target(target),
            )
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask
            objectness_targets = positive_mask.to(dtype=objectness.dtype)
            total_objectness = total_objectness + _binary_cross_entropy_valid(
                objectness,
                objectness_targets,
                valid_mask,
            )

            if positive_mask.any():
                regression_targets = encode_boxes(anchors[positive_mask], assignment.matched_boxes[positive_mask])
                total_box = total_box + F.smooth_l1_loss(
                    bbox_regression[positive_mask],
                    regression_targets,
                    reduction="mean",
                )

        num_images = max(len(images), 1)
        loss_objectness = total_objectness / num_images
        loss_bbox = total_box / num_images
        return {
            "loss_objectness": loss_objectness,
            "loss_bbox": loss_bbox,
            "loss_total": loss_objectness + loss_bbox,
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

                assignment = center_region_assign(points, gt_boxes, gt_labels)
                matched_boxes = assignment.matched_boxes
                matched_labels = assignment.labels
                positive_mask = assignment.positive_mask
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


class DenseCenterNetDecoder(nn.Module):
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
        heatmaps = head_outputs.get("heatmap")
        if heatmaps is None:
            heatmaps = [torch.sigmoid(level) for level in head_outputs["heatmap_logits"]]
        wh_per_level = head_outputs["wh"]
        offset_per_level = head_outputs["offset"]
        points_per_level = generate_points(
            feature_specs,
            device=feature_maps[0].device,
            dtype=feature_maps[0].dtype,
        )

        boxes_per_level = []
        scores_per_level = []
        labels_per_level = []
        for spec, points, heatmap, wh, offset in zip(
            feature_specs,
            points_per_level,
            heatmaps,
            wh_per_level,
            offset_per_level,
        ):
            scores_flat = heatmap.permute(0, 2, 3, 1).reshape(-1, heatmap.shape[1])
            scores, labels = scores_flat.max(dim=1)
            keep = scores >= self.score_threshold
            if not keep.any():
                continue
            wh_flat = wh.permute(0, 2, 3, 1).reshape(-1, 2).abs()
            offset_flat = offset.permute(0, 2, 3, 1).reshape(-1, 2)
            stride = image.new_tensor((float(spec.stride_x), float(spec.stride_y)))
            centers = points + offset_flat * stride
            half_sizes = wh_flat.clamp_min(1.0) * 0.5
            boxes = torch.stack(
                (
                    centers[:, 0] - half_sizes[:, 0],
                    centers[:, 1] - half_sizes[:, 1],
                    centers[:, 0] + half_sizes[:, 0],
                    centers[:, 1] + half_sizes[:, 1],
                ),
                dim=1,
            )
            boxes_per_level.append(clip_boxes_to_image(boxes[keep], image_size))
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


class DenseCenterNetLoss(DenseFCOSLoss):
    """Finite CenterNet smoke loss using dense compatibility targets."""


class DenseCornerNetDecoder(nn.Module):
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
        points_per_level = generate_points(
            feature_specs,
            device=feature_maps[0].device,
            dtype=feature_maps[0].dtype,
        )

        boxes_per_level = []
        scores_per_level = []
        labels_per_level = []
        for (
            spec,
            points,
            top_left,
            bottom_right,
            top_left_offset,
            bottom_right_offset,
            top_left_embedding,
            bottom_right_embedding,
        ) in zip(
            feature_specs,
            points_per_level,
            head_outputs["top_left_heatmap"],
            head_outputs["bottom_right_heatmap"],
            head_outputs["top_left_offset"],
            head_outputs["bottom_right_offset"],
            head_outputs["top_left_embedding"],
            head_outputs["bottom_right_embedding"],
        ):
            top_left_scores, top_left_labels = (
                torch.sigmoid(top_left)
                .permute(0, 2, 3, 1)
                .reshape(-1, top_left.shape[1])
                .max(dim=1)
            )
            bottom_right_scores, bottom_right_labels = (
                torch.sigmoid(bottom_right)
                .permute(0, 2, 3, 1)
                .reshape(-1, bottom_right.shape[1])
                .max(dim=1)
            )
            topk = min(self.detections_per_img, int(top_left_scores.numel()), int(bottom_right_scores.numel()))
            if topk <= 0:
                continue
            top_left_values, top_left_indices = torch.topk(top_left_scores, k=topk)
            bottom_right_values, bottom_right_indices = torch.topk(bottom_right_scores, k=topk)
            point_count = int(points.shape[0])
            top_left_point_indices = torch.remainder(top_left_indices, point_count)
            bottom_right_point_indices = torch.remainder(bottom_right_indices, point_count)
            top_left_offsets = top_left_offset.permute(0, 2, 3, 1).reshape(-1, 2)[top_left_indices]
            bottom_right_offsets = bottom_right_offset.permute(0, 2, 3, 1).reshape(-1, 2)[bottom_right_indices]
            stride = image.new_tensor((float(spec.stride_x), float(spec.stride_y)))
            top_left_points = points[top_left_point_indices] + top_left_offsets * stride
            bottom_right_points = points[bottom_right_point_indices] + bottom_right_offsets * stride

            top_left_embeddings = top_left_embedding.permute(0, 2, 3, 1).reshape(-1, top_left_embedding.shape[1])
            bottom_right_embeddings = bottom_right_embedding.permute(0, 2, 3, 1).reshape(-1, bottom_right_embedding.shape[1])
            top_left_selected_embeddings = top_left_embeddings[top_left_indices]
            bottom_right_selected_embeddings = bottom_right_embeddings[bottom_right_indices]

            pair_scores = (top_left_values[:, None] + bottom_right_values[None, :]) * 0.5
            same_label = top_left_labels[top_left_indices][:, None] == bottom_right_labels[bottom_right_indices][None, :]
            valid_geometry = (
                (bottom_right_points[None, :, 0] >= top_left_points[:, None, 0])
                & (bottom_right_points[None, :, 1] >= top_left_points[:, None, 1])
            )
            embedding_distance = (
                top_left_selected_embeddings[:, None, :]
                - bottom_right_selected_embeddings[None, :, :]
            ).abs().mean(dim=2)
            pair_scores = pair_scores * torch.exp(-embedding_distance)
            keep = (pair_scores >= self.score_threshold) & same_label & valid_geometry
            if not bool(keep.any()):
                continue

            pair_indices = torch.nonzero(keep, as_tuple=False)
            top_left_pair_indices = pair_indices[:, 0]
            bottom_right_pair_indices = pair_indices[:, 1]
            selected_top_left = top_left_points.index_select(0, top_left_pair_indices)
            selected_bottom_right = bottom_right_points.index_select(0, bottom_right_pair_indices)
            x1 = torch.minimum(selected_top_left[:, 0], selected_bottom_right[:, 0])
            y1 = torch.minimum(selected_top_left[:, 1], selected_bottom_right[:, 1])
            x2 = torch.maximum(selected_top_left[:, 0], selected_bottom_right[:, 0])
            y2 = torch.maximum(selected_top_left[:, 1], selected_bottom_right[:, 1])
            boxes = torch.stack((x1, y1, x2, y2), dim=1)
            scores = pair_scores[keep]
            boxes_per_level.append(clip_boxes_to_image(boxes, image_size))
            scores_per_level.append(scores)
            selected_top_left_indices = top_left_indices.index_select(0, top_left_pair_indices)
            labels_per_level.append(top_left_labels[selected_top_left_indices] + 1)

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


class DenseCornerNetLoss(nn.Module):
    """Paired-corner heatmap, offset, pull, and push embedding loss."""

    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_heatmap = images[0].new_tensor(0.0)
        total_offset = images[0].new_tensor(0.0)
        total_pull = images[0].new_tensor(0.0)
        total_push = images[0].new_tensor(0.0)
        positive_count = 0

        for image, target, feature_maps, head_outputs in zip(
            images,
            targets,
            feature_pyramids,
            head_outputs_per_image,
        ):
            feature_maps = _feature_sequence(feature_maps)
            feature_specs = build_feature_map_specs(feature_maps, image_size=_image_size(image))
            points_per_level = generate_points(
                feature_specs,
                device=feature_maps[0].device,
                dtype=feature_maps[0].dtype,
            )
            target_boxes = target.get("boxes", image.new_zeros((0, 4)))
            target_labels = target.get(
                "labels",
                torch.zeros((int(target_boxes.shape[0]),), dtype=torch.long, device=image.device),
            )

            for spec, points, top_left, bottom_right, top_left_offset, bottom_right_offset, top_left_embedding, bottom_right_embedding in zip(
                feature_specs,
                points_per_level,
                head_outputs["top_left_heatmap"],
                head_outputs["bottom_right_heatmap"],
                head_outputs["top_left_offset"],
                head_outputs["bottom_right_offset"],
                head_outputs["top_left_embedding"],
                head_outputs["bottom_right_embedding"],
            ):
                top_left_targets = torch.zeros_like(top_left)
                bottom_right_targets = torch.zeros_like(bottom_right)
                top_left_indices = []
                bottom_right_indices = []
                top_left_offsets = []
                bottom_right_offsets = []
                width = int(top_left.shape[-1])
                stride = image.new_tensor((float(spec.stride_x), float(spec.stride_y)))
                radius = max(float(spec.stride_x), float(spec.stride_y), 1.0)

                for box, label in zip(target_boxes, target_labels):
                    class_index = int(label.clamp(min=1, max=top_left.shape[1]).item()) - 1
                    top_left_index = torch.argmin(((points - box[:2]) ** 2).sum(dim=1))
                    bottom_right_index = torch.argmin(((points - box[2:]) ** 2).sum(dim=1))
                    top_left_row = int(top_left_index.item()) // width
                    top_left_col = int(top_left_index.item()) % width
                    bottom_right_row = int(bottom_right_index.item()) // width
                    bottom_right_col = int(bottom_right_index.item()) % width
                    self._draw_corner_target(
                        top_left_targets,
                        class_index=class_index,
                        corner=box[:2],
                        points=points,
                        radius=radius,
                    )
                    self._draw_corner_target(
                        bottom_right_targets,
                        class_index=class_index,
                        corner=box[2:],
                        points=points,
                        radius=radius,
                    )
                    top_left_targets[0, class_index, top_left_row, top_left_col] = 1.0
                    bottom_right_targets[0, class_index, bottom_right_row, bottom_right_col] = 1.0
                    top_left_indices.append(top_left_index)
                    bottom_right_indices.append(bottom_right_index)
                    top_left_offsets.append((box[:2] - points[top_left_index]) / stride)
                    bottom_right_offsets.append((box[2:] - points[bottom_right_index]) / stride)

                total_heatmap = total_heatmap + self._corner_focal_loss(top_left, top_left_targets)
                total_heatmap = total_heatmap + self._corner_focal_loss(bottom_right, bottom_right_targets)
                if not top_left_indices:
                    continue

                tl_indices = torch.stack(top_left_indices).to(device=top_left.device)
                br_indices = torch.stack(bottom_right_indices).to(device=bottom_right.device)
                tl_offset_targets = torch.stack(top_left_offsets).to(device=top_left_offset.device, dtype=top_left_offset.dtype)
                br_offset_targets = torch.stack(bottom_right_offsets).to(
                    device=bottom_right_offset.device,
                    dtype=bottom_right_offset.dtype,
                )
                tl_offsets = top_left_offset.permute(0, 2, 3, 1).reshape(-1, 2).index_select(0, tl_indices)
                br_offsets = bottom_right_offset.permute(0, 2, 3, 1).reshape(-1, 2).index_select(0, br_indices)
                total_offset = total_offset + F.smooth_l1_loss(tl_offsets, tl_offset_targets, reduction="mean")
                total_offset = total_offset + F.smooth_l1_loss(br_offsets, br_offset_targets, reduction="mean")

                tl_embeddings = top_left_embedding.permute(0, 2, 3, 1).reshape(-1, top_left_embedding.shape[1]).index_select(0, tl_indices)
                br_embeddings = bottom_right_embedding.permute(0, 2, 3, 1).reshape(-1, bottom_right_embedding.shape[1]).index_select(0, br_indices)
                means = (tl_embeddings + br_embeddings) * 0.5
                total_pull = total_pull + (
                    F.smooth_l1_loss(tl_embeddings, means, reduction="mean")
                    + F.smooth_l1_loss(br_embeddings, means, reduction="mean")
                )
                total_push = total_push + self._push_embedding_loss(means)
                positive_count += len(top_left_indices)

        num_images = max(len(images), 1)
        normalizer = max(positive_count, 1)
        loss_heatmap = total_heatmap / num_images
        loss_offset = total_offset / normalizer
        loss_pull = total_pull / normalizer
        loss_push = total_push / normalizer
        return {
            "loss_corner_heatmap": loss_heatmap,
            "loss_corner_offset": loss_offset,
            "loss_corner_embedding": loss_pull + loss_push,
            "loss_total": loss_heatmap + loss_offset + loss_pull + loss_push,
        }

    def _draw_corner_target(self, targets, *, class_index: int, corner, points, radius: float) -> None:
        heat = torch.exp(-((points - corner) ** 2).sum(dim=1) / (2.0 * float(radius) ** 2))
        flat = targets[0, int(class_index)].reshape(-1)
        flat.copy_(torch.maximum(flat, heat.to(device=flat.device, dtype=flat.dtype)))

    def _corner_focal_loss(self, logits, targets):
        probabilities = torch.sigmoid(logits).clamp(min=1e-6, max=1.0 - 1e-6)
        positive = targets.eq(1.0)
        negative = targets.lt(1.0)
        negative_weights = (1.0 - targets).pow(4)
        positive_loss = -torch.log(probabilities) * (1.0 - probabilities).pow(2) * positive
        negative_loss = -torch.log(1.0 - probabilities) * probabilities.pow(2) * negative_weights * negative
        normalizer = positive.to(dtype=logits.dtype).sum().clamp_min(1.0)
        return (positive_loss.sum() + negative_loss.sum()) / normalizer

    def _push_embedding_loss(self, means):
        if int(means.shape[0]) <= 1:
            return means.sum() * 0.0
        distances = (means[:, None, :] - means[None, :, :]).abs().mean(dim=2)
        mask = ~torch.eye(int(means.shape[0]), dtype=torch.bool, device=means.device)
        return F.relu(1.0 - distances[mask]).mean()


class DenseFSAFDecoder(DenseFCOSDecoder):
    """FSAF uses the anchor-free point decode contract."""


class DenseFSAFLoss(DenseFCOSLoss):
    """Finite FSAF smoke loss on the shared anchor-free dense target contract."""


class DenseFoveaDecoder(DenseFCOSDecoder):
    """Fovea uses the anchor-free point decode contract."""


class DenseFoveaLoss(DenseFCOSLoss):
    """Finite Fovea smoke loss on the shared anchor-free dense target contract."""


class DenseYOLOXDecoder(nn.Module):
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
        points = _point_priors_for_image(image, feature_maps)
        cls_logits = flatten_cls_logits(head_outputs["cls_logits"])
        bbox_regression = flatten_bbox_regression(head_outputs["bbox_regression"])
        objectness = _required_objectness_logits(head_outputs)
        class_scores = torch.sigmoid(cls_logits)
        objectness_scores = torch.sigmoid(objectness).unsqueeze(1)
        scores, labels = (class_scores * objectness_scores).max(dim=1)
        decoded = clip_boxes_to_image(
            decode_point_boxes(points, bbox_regression.clamp(min=0)),
            image_size,
        )

        keep = scores >= self.score_threshold
        payload = make_batched_nms_payload(decoded[keep], scores[keep], labels[keep] + 1)
        if payload["boxes"].numel() == 0:
            return prediction_payload_to_dict(payload)

        keep_idx = batched_nms(payload["boxes"], payload["scores"], payload["nms_indices"], self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return prediction_payload_to_dict(select_prediction_payload(payload, keep_idx))


class DenseYOLOXTargetBuilder:
    def __init__(
        self,
        *,
        objectness_target_config=None,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
    ) -> None:
        if objectness_target_config is None:
            raise ValueError(
                "YOLOXHead training-loss setup requires objectness_target_config "
                "for positive and negative objectness targets."
            )
        if not isinstance(objectness_target_config, dict):
            raise ValueError("objectness_target_config must be a mapping.")
        self.positive_value = float(
            objectness_target_config.get(
                "positive",
                objectness_target_config.get("positive_value", 1.0),
            )
        )
        self.negative_value = float(
            objectness_target_config.get(
                "negative",
                objectness_target_config.get("negative_value", 0.0),
            )
        )
        self.center_radius = float(center_radius)
        self.candidate_topk = int(candidate_topk)

    def __call__(self, points, pred_scores, pred_boxes, target):
        priors = _point_boxes(points)
        assignment = sim_ota_assign(
            priors,
            pred_scores,
            pred_boxes,
            target["boxes"],
            target["labels"],
            center_radius=self.center_radius,
            candidate_topk=self.candidate_topk,
        )
        objectness_targets = pred_scores.new_full((points.shape[0],), self.negative_value)
        objectness_targets[assignment.positive_mask] = self.positive_value
        return assignment, objectness_targets


class DenseYOLOXLoss(nn.Module):
    def __init__(
        self,
        *,
        objectness_target_config=None,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
    ) -> None:
        super().__init__()
        self.target_builder = DenseYOLOXTargetBuilder(
            objectness_target_config=objectness_target_config,
            center_radius=center_radius,
            candidate_topk=candidate_topk,
        )

    def forward(self, images, targets, feature_pyramids, head_outputs_per_image):
        total_cls = torch.tensor(0.0, device=images[0].device)
        total_box = torch.tensor(0.0, device=images[0].device)
        total_obj = torch.tensor(0.0, device=images[0].device)

        for image, target, feature_maps, head_outputs in zip(
            images,
            targets,
            feature_pyramids,
            head_outputs_per_image,
        ):
            feature_maps = _feature_sequence(feature_maps)
            points = _point_priors_for_image(image, feature_maps)
            cls_logits = flatten_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_bbox_regression(head_outputs["bbox_regression"])
            objectness = _required_objectness_logits(head_outputs)
            decoded = clip_boxes_to_image(
                decode_point_boxes(points, bbox_regression.clamp(min=0)),
                _image_size(image),
            )
            pred_scores = torch.sigmoid(cls_logits) * torch.sigmoid(objectness).unsqueeze(1)
            assignment, objectness_targets = self.target_builder(points, pred_scores, decoded, target)
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask

            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = assignment.labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0

            total_cls = total_cls + _binary_cross_entropy_valid(cls_logits, class_targets, valid_mask)
            total_obj = total_obj + _binary_cross_entropy_valid(objectness, objectness_targets, valid_mask)

            if positive_mask.any():
                regression_targets = encode_point_boxes(points[positive_mask], assignment.matched_boxes[positive_mask])
                total_box = total_box + F.l1_loss(
                    bbox_regression[positive_mask],
                    regression_targets,
                    reduction="mean",
                )

        num_images = max(len(images), 1)
        loss_cls = total_cls / num_images
        loss_bbox = total_box / num_images
        loss_objectness = total_obj / num_images
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_objectness": loss_objectness,
            "loss_total": loss_cls + loss_bbox + loss_objectness,
        }


class DenseRTMDetDecoder(DenseFCOSDecoder):
    """RTMDet uses class scores and point-box decoding without objectness."""


class DenseRTMDetTargetBuilder:
    def __init__(self, *, topk: int = 13, alpha: float = 1.0, beta: float = 6.0) -> None:
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(self, points, pred_scores, pred_boxes, target):
        return task_aligned_assign(
            _point_boxes(points),
            pred_scores,
            pred_boxes,
            target["boxes"],
            target["labels"],
            topk=self.topk,
            alpha=self.alpha,
            beta=self.beta,
        )


class DenseRTMDetLoss(nn.Module):
    def __init__(self, *, topk: int = 13, alpha: float = 1.0, beta: float = 6.0) -> None:
        super().__init__()
        self.target_builder = DenseRTMDetTargetBuilder(topk=topk, alpha=alpha, beta=beta)

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
            points = _point_priors_for_image(image, feature_maps)
            cls_logits = flatten_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_bbox_regression(head_outputs["bbox_regression"])
            decoded = clip_boxes_to_image(
                decode_point_boxes(points, bbox_regression.clamp(min=0)),
                _image_size(image),
            )
            assignment = self.target_builder(points, torch.sigmoid(cls_logits), decoded, target)
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask

            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = assignment.labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0
            total_cls = total_cls + _binary_cross_entropy_valid(cls_logits, class_targets, valid_mask)

            if positive_mask.any():
                regression_targets = encode_point_boxes(points[positive_mask], assignment.matched_boxes[positive_mask])
                total_box = total_box + F.l1_loss(
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
            anchors_per_level = _anchor_priors_per_level_for_image(image, feature_maps)
            anchors = torch.cat(anchors_per_level, dim=0)
            cls_logits = flatten_anchor_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])
            centerness = flatten_anchor_centerness_logits(head_outputs["centerness"])

            assignment = atss_assign(
                anchors,
                target["boxes"],
                target["labels"],
                num_level_priors=tuple(level.shape[0] for level in anchors_per_level),
                ignored_boxes=_ignored_boxes_from_target(target),
            )
            matched_boxes = assignment.matched_boxes
            matched_labels = assignment.labels
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask
            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = matched_labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0

            total_cls = total_cls + _binary_cross_entropy_valid(cls_logits, class_targets, valid_mask)

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


class DenseGFLV2Decoder(DenseGFLDecoder):
    """GFLv2 uses the native GFL anchor-based decode path."""


class DenseGFLV2Loss(DenseGFLLoss):
    """Finite GFLv2 smoke loss on the shared GFL dense target contract."""


class DensePAADecoder(DenseATSSDecoder):
    """PAA uses the native anchor-based dense decode path."""


class DensePAALoss(DenseATSSLoss):
    """Finite PAA smoke loss on the shared ATSS dense target contract."""


class DenseDDODDecoder(DenseATSSDecoder):
    """DDOD uses the native anchor-based dense decode path."""


class DenseDDODLoss(DenseATSSLoss):
    """Finite DDOD smoke loss on the shared ATSS dense target contract."""


class DenseSSDDecoder(nn.Module):
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
        scores, labels = torch.sigmoid(cls_logits).max(dim=1)
        decoded = clip_boxes_to_image(decode_boxes(anchors, bbox_regression), image_size)

        keep = scores >= self.score_threshold
        payload = make_batched_nms_payload(decoded[keep], scores[keep], labels[keep] + 1)
        if payload["boxes"].numel() == 0:
            return prediction_payload_to_dict(payload)

        keep_idx = batched_nms(payload["boxes"], payload["scores"], payload["nms_indices"], self.nms_threshold)
        keep_idx = keep_idx[: self.detections_per_img]
        return prediction_payload_to_dict(select_prediction_payload(payload, keep_idx))


class DenseSSDTargetBuilder:
    def __init__(
        self,
        *,
        pos_iou_thr: float = 0.5,
        neg_iou_thr: float = 0.4,
    ) -> None:
        self.pos_iou_thr = float(pos_iou_thr)
        self.neg_iou_thr = float(neg_iou_thr)

    def __call__(self, anchors, target):
        return max_iou_assign(
            anchors,
            target["boxes"],
            target["labels"],
            pos_iou_thr=self.pos_iou_thr,
            neg_iou_thr=self.neg_iou_thr,
            ignored_boxes=_ignored_boxes_from_target(target),
        )


class DenseSSDLoss(nn.Module):
    def __init__(
        self,
        *,
        pos_iou_thr: float = 0.5,
        neg_iou_thr: float = 0.4,
    ) -> None:
        super().__init__()
        self.target_builder = DenseSSDTargetBuilder(pos_iou_thr=pos_iou_thr, neg_iou_thr=neg_iou_thr)

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
            cls_logits = flatten_anchor_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])

            assignment = self.target_builder(anchors, target)
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask
            class_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                positive_labels = assignment.labels[positive_mask].long().clamp(min=1) - 1
                class_targets[positive_mask, positive_labels] = 1.0
            total_cls = total_cls + _binary_cross_entropy_valid(cls_logits, class_targets, valid_mask)

            if positive_mask.any():
                regression_targets = encode_boxes(anchors[positive_mask], assignment.matched_boxes[positive_mask])
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


class DenseEfficientDetDecoder(DenseSSDDecoder):
    """EfficientDet uses the native anchor-based class and box decode path."""


class DenseEfficientDetLoss(DenseSSDLoss):
    """EfficientDet target assignment and finite loss on the shared anchor contract."""


class DenseVFNetLoss(nn.Module):
    """VFNet-style dense loss composed from the shared native loss registry."""

    def __init__(self) -> None:
        super().__init__()
        LOSSES.import_modules("simpledet.native.losses")
        self.loss_cls = LOSSES.get("varifocal")()
        self.loss_bbox = LOSSES.get("iou")()

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
            anchors_per_level = _anchor_priors_per_level_for_image(image, feature_maps)
            anchors = torch.cat(anchors_per_level, dim=0)
            cls_logits = flatten_anchor_cls_logits(head_outputs["cls_logits"])
            bbox_regression = flatten_anchor_bbox_regression(head_outputs["bbox_regression"])

            assignment = atss_assign(
                anchors,
                target["boxes"],
                target["labels"],
                num_level_priors=tuple(level.shape[0] for level in anchors_per_level),
                ignored_boxes=_ignored_boxes_from_target(target),
            )
            matched_boxes = assignment.matched_boxes
            matched_labels = assignment.labels
            positive_mask = assignment.positive_mask
            valid_mask = ~assignment.ignored_mask
            class_targets = torch.zeros_like(cls_logits)

            if positive_mask.any():
                positive_labels = matched_labels[positive_mask].long().clamp(min=1) - 1
                decoded_positive = decode_boxes(anchors[positive_mask], bbox_regression[positive_mask])
                with torch.no_grad():
                    quality_targets = _aligned_box_iou(
                        decoded_positive.detach(),
                        matched_boxes[positive_mask],
                    ).clamp(min=0.0, max=1.0)
                class_targets[positive_mask, positive_labels] = quality_targets
                total_box = total_box + self.loss_bbox(decoded_positive, matched_boxes[positive_mask])

            total_cls = total_cls + self.loss_cls(cls_logits[valid_mask], class_targets[valid_mask])

        num_images = max(len(images), 1)
        loss_cls = total_cls / num_images
        loss_bbox = total_box / num_images
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_total": loss_cls + loss_bbox,
        }


class DenseVFNetDecoder(DenseATSSDecoder):
    """VFNet uses the native anchor-based dense decode path."""


class DenseRepPointsDecoder(DenseFCOSDecoder):
    """RepPoints uses the native point-based dense decode path."""


class DenseRepPointsLoss(DenseFCOSLoss):
    """Finite RepPoints smoke loss on the shared point dense target contract."""


class DenseYOLOFDecoder(DenseFCOSDecoder):
    """YOLOF uses the native point-based dense decode path."""


class DenseYOLOFLoss(DenseFCOSLoss):
    """Finite YOLOF smoke loss on the shared point dense target contract."""


class DenseTOODDecoder(DenseFCOSDecoder):
    """TOOD uses the native point-based dense decode path."""


class DenseTOODLoss(DenseFCOSLoss):
    """Finite TOOD smoke loss on the shared point dense target contract."""


class DenseAutoAssignDecoder(DenseFCOSDecoder):
    """AutoAssign uses the native point-based dense decode path."""


class DenseAutoAssignLoss(DenseFCOSLoss):
    """Finite AutoAssign smoke loss on the shared point dense target contract."""


class DenseNASFCOSDecoder(DenseFCOSDecoder):
    """NAS-FCOS uses the native point-based dense decode path."""


class DenseNASFCOSLoss(DenseFCOSLoss):
    """Finite NAS-FCOS smoke loss on the shared point dense target contract."""


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
    return torch.cat(_anchor_priors_from_specs(feature_specs, feature_maps), dim=0)


def _anchor_priors_per_level_for_image(image, feature_maps):
    feature_specs = build_feature_map_specs(feature_maps, image_size=_image_size(image))
    return _anchor_priors_from_specs(feature_specs, feature_maps)


def _anchor_priors_from_specs(feature_specs, feature_maps):
    return generate_anchors(
        feature_specs,
        device=feature_maps[0].device,
        dtype=feature_maps[0].dtype,
    )


def _point_priors_for_image(image, feature_maps):
    feature_specs = build_feature_map_specs(feature_maps, image_size=_image_size(image))
    return torch.cat(
        generate_points(
            feature_specs,
            device=feature_maps[0].device,
            dtype=feature_maps[0].dtype,
        ),
        dim=0,
    )


def _point_boxes(points):
    return torch.cat((points, points), dim=1)


def _required_objectness_logits(head_outputs):
    if "objectness_logits" not in head_outputs:
        raise ValueError("YOLOXHead loss and decode require objectness_logits.")
    return flatten_centerness_logits(head_outputs["objectness_logits"])


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


def flatten_anchor_objectness_logits(objectness_per_level):
    return flatten_anchor_centerness_logits(objectness_per_level)


def _binary_cross_entropy_valid(logits, targets, valid_mask):
    if valid_mask.all():
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    if valid_mask.any():
        return F.binary_cross_entropy_with_logits(logits[valid_mask], targets[valid_mask], reduction="mean")
    return logits.sum() * 0.0


def _aligned_box_iou(boxes1, boxes2):
    if boxes1.shape != boxes2.shape:
        raise ValueError(
            f"aligned IoU expects matching box shapes, got {tuple(boxes1.shape)} and {tuple(boxes2.shape)}."
        )
    if boxes1.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=torch.finfo(boxes1.dtype).eps)


def _ignored_boxes_from_target(target):
    for key in ("ignored_boxes", "ignore_boxes", "gt_bboxes_ignore"):
        if key in target:
            return target[key]
    return None


def match_anchors(anchors, target):
    assignment = max_iou_assign(
        anchors,
        target["boxes"],
        target["labels"],
        ignored_boxes=_ignored_boxes_from_target(target),
    )
    return assignment.matched_boxes, assignment.labels, assignment.positive_mask


def build_fcos_points(feature_map, *, image_size=None, stride=None):
    strides = None if stride is None else (stride,)
    if image_size is None and strides is None:
        strides = ((1.0, 1.0),)
    feature_specs = build_feature_map_specs((feature_map,), image_size=image_size, strides=strides)
    return generate_points(feature_specs, device=feature_map.device, dtype=feature_map.dtype)[0]


def match_points_to_boxes(points, gt_boxes, gt_labels):
    assignment = center_region_assign(points, gt_boxes, gt_labels)
    return assignment.matched_boxes, assignment.labels, assignment.positive_mask


def encode_fcos_boxes(points, gt_boxes):
    return encode_point_boxes(points, gt_boxes)


def decode_fcos_boxes(points, deltas):
    return decode_point_boxes(points, deltas)
