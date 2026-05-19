"""COCO-like bbox evaluation helpers for native SimpleDet outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))
_RECALL_SAMPLES = tuple(index / 100.0 for index in range(101))


def evaluate_coco_bbox_metrics(
    annotation_file: str | Path,
    predictions: Sequence[Mapping[str, Any]] | None,
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate serialized bbox predictions against a COCO-like annotation file.

    Prediction boxes are expected in the SimpleDet public ``xyxy`` payload shape.
    The returned ``prediction_export`` is COCO-compatible ``xywh`` detection JSON.
    """

    categories, ground_truth = _load_coco_ground_truth(annotation_file)
    thresholds = _normalize_iou_thresholds(iou_thresholds)
    prediction_export = _build_prediction_export(predictions or (), categories)
    per_class = []
    valid_class_summaries = []

    for category in categories:
        category_id = int(category["id"])
        category_name = str(category["name"])
        category_ground_truth = ground_truth.get(category_id, {})
        category_predictions = [
            prediction
            for prediction in prediction_export
            if int(prediction["category_id"]) == category_id
        ]
        class_summary = _evaluate_category(
            category_id=category_id,
            category_name=category_name,
            ground_truth_by_image=category_ground_truth,
            predictions=category_predictions,
            thresholds=thresholds,
        )
        per_class.append(class_summary)
        if class_summary["num_ground_truth"] > 0:
            valid_class_summaries.append(class_summary)

    summary = _summarize_metrics(
        per_class=valid_class_summaries,
        thresholds=thresholds,
        num_images=len(_load_coco_image_ids(annotation_file)),
        num_predictions=len(prediction_export),
    )
    if not valid_class_summaries:
        summary["num_ground_truth"] = 0
    else:
        summary["num_ground_truth"] = sum(
            int(item["num_ground_truth"]) for item in valid_class_summaries
        )

    return {
        "summary": summary,
        "per_class": per_class,
        "recall": _recall_summary(valid_class_summaries, thresholds),
        "prediction_export": prediction_export,
    }


def build_coco_prediction_export(
    annotation_file: str | Path,
    predictions: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return COCO detection payloads from serialized SimpleDet predictions."""

    categories, _ground_truth = _load_coco_ground_truth(annotation_file)
    return _build_prediction_export(predictions or (), categories)


def _load_coco_ground_truth(
    annotation_file: str | Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[int, list[list[float]]]]]:
    payload = _load_json_object(annotation_file)
    raw_categories = payload.get("categories", [])
    raw_annotations = payload.get("annotations", [])
    if not isinstance(raw_categories, list):
        raise ValueError("COCO metrics require a list-valued 'categories' field.")
    if not isinstance(raw_annotations, list):
        raise ValueError("COCO metrics require a list-valued 'annotations' field.")

    categories = _parse_categories(raw_categories)
    category_ids = {int(category["id"]) for category in categories}
    ground_truth: dict[int, dict[int, list[list[float]]]] = {
        int(category["id"]): {} for category in categories
    }
    for annotation in raw_annotations:
        if not isinstance(annotation, Mapping):
            continue
        if int(annotation.get("iscrowd", 0) or 0):
            continue
        category_id = _coerce_int(annotation.get("category_id"))
        image_id = _coerce_int(annotation.get("image_id"))
        bbox = _coerce_bbox_xywh(annotation.get("bbox"))
        if category_id is None or image_id is None or bbox is None:
            continue
        if category_id not in category_ids:
            continue
        ground_truth.setdefault(category_id, {}).setdefault(image_id, []).append(bbox)
    return categories, ground_truth


def _load_coco_image_ids(annotation_file: str | Path) -> set[int]:
    payload = _load_json_object(annotation_file)
    raw_images = payload.get("images", [])
    if not isinstance(raw_images, list):
        return set()
    image_ids = set()
    for image in raw_images:
        if not isinstance(image, Mapping):
            continue
        image_id = _coerce_int(image.get("id"))
        if image_id is not None:
            image_ids.add(image_id)
    return image_ids


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"COCO metrics require a JSON object: {resolved}")
    return payload


def _parse_categories(raw_categories: Sequence[Any]) -> list[dict[str, Any]]:
    categories = []
    seen = set()
    for item in raw_categories:
        if not isinstance(item, Mapping):
            continue
        category_id = _coerce_int(item.get("id"))
        name = item.get("name")
        if category_id is None or category_id in seen:
            continue
        categories.append(
            {
                "id": category_id,
                "name": str(name).strip() if str(name).strip() else str(category_id),
            }
        )
        seen.add(category_id)
    categories.sort(key=lambda item: int(item["id"]))
    return categories


def _build_prediction_export(
    predictions: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    category_ids = [int(category["id"]) for category in categories]
    detections = []
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            continue
        image_id = _prediction_image_id(prediction)
        boxes = _as_list(prediction.get("boxes", []))
        scores = _as_list(prediction.get("scores", []))
        labels = _as_list(prediction.get("labels", []))
        for index, raw_box in enumerate(boxes):
            box = _xyxy_to_xywh(raw_box)
            if box is None:
                continue
            category_id = _map_label_to_category_id(_value_at(labels, index), category_ids)
            if category_id is None:
                continue
            score = _coerce_float(_value_at(scores, index, 1.0))
            if score is None:
                continue
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": box,
                    "score": float(score),
                }
            )
    detections.sort(
        key=lambda item: (
            int(item["image_id"]),
            int(item["category_id"]),
            -float(item["score"]),
        )
    )
    return detections


def _evaluate_category(
    *,
    category_id: int,
    category_name: str,
    ground_truth_by_image: Mapping[int, Sequence[Sequence[float]]],
    predictions: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    gt_count = sum(len(boxes) for boxes in ground_truth_by_image.values())
    ap_by_iou: dict[str, float] = {}
    recall_by_iou: dict[str, float] = {}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        precision, recall = _precision_recall_at_iou(
            ground_truth_by_image=ground_truth_by_image,
            predictions=predictions,
            threshold=float(threshold),
            gt_count=gt_count,
        )
        ap_by_iou[key] = _interpolated_average_precision(precision, recall)
        recall_by_iou[key] = float(recall[-1]) if recall else 0.0

    ap = _mean(ap_by_iou.values())
    recall = _mean(recall_by_iou.values())
    return {
        "category_id": int(category_id),
        "name": category_name,
        "ap": ap if gt_count else 0.0,
        "ap_50": ap_by_iou.get("0.50", 0.0) if gt_count else 0.0,
        "ap_75": ap_by_iou.get("0.75", 0.0) if gt_count else 0.0,
        "recall": recall if gt_count else 0.0,
        "recall_50": recall_by_iou.get("0.50", 0.0) if gt_count else 0.0,
        "recall_75": recall_by_iou.get("0.75", 0.0) if gt_count else 0.0,
        "ap_by_iou": ap_by_iou,
        "recall_by_iou": recall_by_iou,
        "num_ground_truth": int(gt_count),
        "num_predictions": int(len(predictions)),
    }


def _precision_recall_at_iou(
    *,
    ground_truth_by_image: Mapping[int, Sequence[Sequence[float]]],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float,
    gt_count: int,
) -> tuple[list[float], list[float]]:
    if gt_count <= 0:
        return [], []

    matched = {
        int(image_id): [False for _box in boxes]
        for image_id, boxes in ground_truth_by_image.items()
    }
    ordered_predictions = sorted(
        predictions,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    true_positives = []
    false_positives = []
    for prediction in ordered_predictions:
        image_id = int(prediction["image_id"])
        gt_boxes = ground_truth_by_image.get(image_id, ())
        best_index = -1
        best_iou = float(threshold)
        for index, gt_box in enumerate(gt_boxes):
            if matched.get(image_id, [])[index]:
                continue
            iou = _bbox_iou_xywh(prediction["bbox"], gt_box)
            if iou >= best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            matched[image_id][best_index] = True
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not true_positives:
        return [0.0], [0.0]

    precision = []
    recall = []
    cumulative_tp = 0.0
    cumulative_fp = 0.0
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        precision.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1.0))
        recall.append(cumulative_tp / float(gt_count))
    return precision, recall


def _interpolated_average_precision(precision: Sequence[float], recall: Sequence[float]) -> float:
    if not precision or not recall:
        return 0.0
    samples = []
    for recall_threshold in _RECALL_SAMPLES:
        candidates = [
            float(prec)
            for prec, rec in zip(precision, recall)
            if float(rec) >= recall_threshold
        ]
        samples.append(max(candidates) if candidates else 0.0)
    return _mean(samples)


def _summarize_metrics(
    *,
    per_class: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    num_images: int,
    num_predictions: int,
) -> dict[str, Any]:
    by_iou = {}
    recall_by_iou = {}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        by_iou[key] = _mean(item.get("ap_by_iou", {}).get(key, 0.0) for item in per_class)
        recall_by_iou[key] = _mean(
            item.get("recall_by_iou", {}).get(key, 0.0) for item in per_class
        )
    return {
        "map": _mean(item.get("ap", 0.0) for item in per_class),
        "map_50": by_iou.get("0.50", 0.0),
        "map_75": by_iou.get("0.75", 0.0),
        "mean_recall": _mean(item.get("recall", 0.0) for item in per_class),
        "recall_50": recall_by_iou.get("0.50", 0.0),
        "recall_75": recall_by_iou.get("0.75", 0.0),
        "ap_by_iou": by_iou,
        "recall_by_iou": recall_by_iou,
        "num_images": int(num_images),
        "num_ground_truth": 0,
        "num_predictions": int(num_predictions),
    }


def _recall_summary(
    per_class: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    by_iou = {}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        by_iou[key] = _mean(item.get("recall_by_iou", {}).get(key, 0.0) for item in per_class)
    return {
        "mean": _mean(item.get("recall", 0.0) for item in per_class),
        "recall_50": by_iou.get("0.50", 0.0),
        "recall_75": by_iou.get("0.75", 0.0),
        "by_iou": by_iou,
        "per_class": [
            {
                "category_id": int(item["category_id"]),
                "name": str(item["name"]),
                "recall": float(item.get("recall", 0.0)),
                "recall_50": float(item.get("recall_50", 0.0)),
                "recall_75": float(item.get("recall_75", 0.0)),
            }
            for item in per_class
        ],
    }


def _xyxy_to_xywh(value: Any) -> list[float] | None:
    box = _as_list(value)
    if len(box) != 4:
        return None
    x_min = _coerce_float(box[0])
    y_min = _coerce_float(box[1])
    x_max = _coerce_float(box[2])
    y_max = _coerce_float(box[3])
    if None in {x_min, y_min, x_max, y_max}:
        return None
    width = float(x_max) - float(x_min)
    height = float(y_max) - float(y_min)
    if width <= 0.0 or height <= 0.0:
        return None
    return [float(x_min), float(y_min), width, height]


def _coerce_bbox_xywh(value: Any) -> list[float] | None:
    box = _as_list(value)
    if len(box) != 4:
        return None
    x_min = _coerce_float(box[0])
    y_min = _coerce_float(box[1])
    width = _coerce_float(box[2])
    height = _coerce_float(box[3])
    if None in {x_min, y_min, width, height}:
        return None
    if float(width) <= 0.0 or float(height) <= 0.0:
        return None
    return [float(x_min), float(y_min), float(width), float(height)]


def _bbox_iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    left_x1, left_y1, left_w, left_h = (float(value) for value in left)
    right_x1, right_y1, right_w, right_h = (float(value) for value in right)
    left_x2 = left_x1 + left_w
    left_y2 = left_y1 + left_h
    right_x2 = right_x1 + right_w
    right_y2 = right_y1 + right_h
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    intersection = inter_w * inter_h
    union = left_w * left_h + right_w * right_h - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _prediction_image_id(prediction: Mapping[str, Any]) -> int:
    image_id = _coerce_int(prediction.get("image_id"))
    return int(image_id) if image_id is not None else 0


def _map_label_to_category_id(label: Any, category_ids: Sequence[int]) -> int | None:
    label_id = _coerce_int(label)
    if label_id is None or label_id <= 0:
        return None
    if label_id in category_ids:
        return int(label_id)
    label_index = label_id - 1
    if 0 <= label_index < len(category_ids):
        return int(category_ids[label_index])
    return None


def _value_at(values: Sequence[Any], index: int, default: Any = None) -> Any:
    if index < len(values):
        return values[index]
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _coerce_float(value: Any) -> float | None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_int(value: Any) -> int | None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = value[0] if value else None
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _normalize_iou_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(threshold) for threshold in thresholds)
    if not normalized:
        raise ValueError("At least one IoU threshold is required for bbox metrics.")
    for threshold in normalized:
        if not math.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
            raise ValueError("IoU thresholds must be finite values in the range (0, 1].")
    return normalized


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "build_coco_prediction_export",
    "evaluate_coco_bbox_metrics",
]
