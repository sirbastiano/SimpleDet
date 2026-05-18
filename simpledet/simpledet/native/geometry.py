"""Native geometry utilities shared by dense detector heads."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..detectors._deps import require_dependency

require_dependency("torch", "native geometry")
import torch  # noqa: E402


DEFAULT_ANCHOR_SIZES = (32.0, 64.0, 128.0, 256.0, 512.0)
DEFAULT_ASPECT_RATIOS = (0.5, 1.0, 2.0)
DEFAULT_ANCHOR_SCALES = (1.0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0))


@dataclass(frozen=True, slots=True)
class FeatureMapSpec:
    """Spatial metadata needed to place dense priors in image coordinates."""

    height: int
    width: int
    stride_y: float
    stride_x: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "height", _positive_int(self.height, "feature height"))
        object.__setattr__(self, "width", _positive_int(self.width, "feature width"))
        object.__setattr__(self, "stride_y", _positive_float(self.stride_y, "stride_y"))
        object.__setattr__(self, "stride_x", _positive_float(self.stride_x, "stride_x"))

    @property
    def size(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def stride(self) -> tuple[float, float]:
        return self.stride_y, self.stride_x

    @property
    def num_locations(self) -> int:
        return self.height * self.width


def build_feature_map_specs(
    feature_maps: Sequence[Any] | None = None,
    *,
    feature_sizes: Sequence[Sequence[int]] | None = None,
    image_size: Sequence[int] | None = None,
    strides: Sequence[float | Sequence[float]] | None = None,
) -> tuple[FeatureMapSpec, ...]:
    """Build validated feature-map specs from tensors or explicit sizes."""

    sizes = _feature_sizes_from_inputs(feature_maps, feature_sizes)
    if not sizes:
        return ()

    if strides is None:
        if image_size is None:
            raise ValueError("strides or image_size are required to build feature map specs.")
        stride_pairs = infer_strides(image_size, sizes)
    else:
        if len(strides) != len(sizes):
            raise ValueError(
                f"stride count mismatch: expected {len(sizes)} entries for feature sizes, got {len(strides)}."
            )
        stride_pairs = tuple(_normalize_stride(stride) for stride in strides)

    return tuple(
        FeatureMapSpec(
            height=height,
            width=width,
            stride_y=stride_y,
            stride_x=stride_x,
        )
        for (height, width), (stride_y, stride_x) in zip(sizes, stride_pairs)
    )


def infer_strides(
    image_size: Sequence[int],
    feature_sizes: Sequence[Sequence[int]],
) -> tuple[tuple[float, float], ...]:
    """Infer positive y/x strides from image size and feature map sizes."""

    image_height, image_width = _normalize_hw(image_size, name="image size")
    strides = []
    for size in feature_sizes:
        height, width = _normalize_hw(size, name="feature size")
        strides.append((float(image_height) / float(height), float(image_width) / float(width)))
    return tuple(strides)


def generate_points(
    feature_specs: Sequence[FeatureMapSpec],
    *,
    offset: float = 0.5,
    device: Any | None = None,
    dtype: Any | None = None,
) -> tuple[Any, ...]:
    """Generate center points for each feature map in image coordinates."""

    offset = float(offset)
    if offset < 0.0 or offset > 1.0:
        raise ValueError("point offset must be between 0 and 1.")
    dtype = dtype or torch.float32
    points = []
    for spec in feature_specs:
        y_positions = (torch.arange(spec.height, device=device, dtype=dtype) + offset) * spec.stride_y
        x_positions = (torch.arange(spec.width, device=device, dtype=dtype) + offset) * spec.stride_x
        ys, xs = torch.meshgrid(y_positions, x_positions, indexing="ij")
        points.append(torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1))
    return tuple(points)


def generate_anchors(
    feature_specs: Sequence[FeatureMapSpec],
    *,
    base_sizes: Sequence[float] | None = None,
    aspect_ratios: Sequence[float] = DEFAULT_ASPECT_RATIOS,
    scales: Sequence[float] = DEFAULT_ANCHOR_SCALES,
    offset: float = 0.5,
    device: Any | None = None,
    dtype: Any | None = None,
) -> tuple[Any, ...]:
    """Generate anchor priors for each feature map in xyxy image coordinates."""

    dtype = dtype or torch.float32
    if base_sizes is None:
        base_sizes = tuple(
            DEFAULT_ANCHOR_SIZES[min(index, len(DEFAULT_ANCHOR_SIZES) - 1)]
            for index, _ in enumerate(feature_specs)
        )
    if len(base_sizes) != len(feature_specs):
        raise ValueError(
            f"anchor base size count mismatch: expected {len(feature_specs)}, got {len(base_sizes)}."
        )
    anchor_levels = []
    point_levels = generate_points(feature_specs, offset=offset, device=device, dtype=dtype)
    for spec, points, base_size in zip(feature_specs, point_levels, base_sizes):
        base_anchors = _base_anchors(
            _positive_float(base_size, "anchor base size"),
            aspect_ratios=aspect_ratios,
            scales=scales,
            device=device,
            dtype=dtype,
        )
        centers = points[:, None, :].expand(spec.num_locations, base_anchors.shape[0], 2)
        center_boxes = torch.cat((centers, centers), dim=2)
        anchor_levels.append((center_boxes + base_anchors[None, :, :]).reshape(-1, 4))
    return tuple(anchor_levels)


def generate_priors(
    kind: str,
    feature_specs: Sequence[FeatureMapSpec],
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Generate anchor or point priors through one validated entrypoint."""

    normalized = str(kind).strip().lower()
    if normalized in {"anchor", "anchors"}:
        return generate_anchors(feature_specs, **kwargs)
    if normalized in {"point", "points"}:
        return generate_points(feature_specs, **kwargs)
    raise ValueError("prior kind must be 'anchor' or 'point'.")


def encode_boxes(
    reference_boxes: Any,
    target_boxes: Any,
    *,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> Any:
    """Encode target boxes as deltas relative to reference boxes."""

    reference_boxes = _as_float_boxes(reference_boxes, name="reference_boxes")
    target_boxes = _as_float_boxes(target_boxes, name="target_boxes")
    target_boxes = target_boxes.to(device=reference_boxes.device, dtype=reference_boxes.dtype)
    _validate_same_box_count(reference_boxes, target_boxes)
    if reference_boxes.numel() == 0:
        return reference_boxes.new_zeros((0, 4))

    wx, wy, ww, wh = _normalize_weights(weights)
    ref_widths, ref_heights, ref_ctr_x, ref_ctr_y = _box_centers(reference_boxes)
    tgt_widths, tgt_heights, tgt_ctr_x, tgt_ctr_y = _box_centers(target_boxes)
    eps = torch.finfo(reference_boxes.dtype).eps
    ref_widths = ref_widths.clamp(min=eps)
    ref_heights = ref_heights.clamp(min=eps)
    tgt_widths = tgt_widths.clamp(min=eps)
    tgt_heights = tgt_heights.clamp(min=eps)

    dx = wx * (tgt_ctr_x - ref_ctr_x) / ref_widths
    dy = wy * (tgt_ctr_y - ref_ctr_y) / ref_heights
    dw = ww * torch.log(tgt_widths / ref_widths)
    dh = wh * torch.log(tgt_heights / ref_heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(
    reference_boxes: Any,
    deltas: Any,
    *,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> Any:
    """Decode box deltas back to xyxy boxes."""

    reference_boxes = _as_float_boxes(reference_boxes, name="reference_boxes")
    deltas = _as_float_boxes(deltas, name="deltas")
    deltas = deltas.to(device=reference_boxes.device, dtype=reference_boxes.dtype)
    _validate_same_box_count(reference_boxes, deltas)
    if reference_boxes.numel() == 0:
        return reference_boxes.new_zeros((0, 4))

    wx, wy, ww, wh = _normalize_weights(weights)
    widths, heights, ctr_x, ctr_y = _box_centers(reference_boxes)
    eps = torch.finfo(reference_boxes.dtype).eps
    widths = widths.clamp(min=eps)
    heights = heights.clamp(min=eps)
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    return torch.stack(
        (
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ),
        dim=1,
    )


def box_iou(boxes1: Any, boxes2: Any) -> Any:
    """Return pairwise IoU for two xyxy box tensors."""

    boxes1 = _as_float_boxes(boxes1, name="boxes1")
    boxes2 = _as_float_boxes(boxes2, name="boxes2")
    boxes2 = boxes2.to(device=boxes1.device, dtype=boxes1.dtype)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=torch.finfo(boxes1.dtype).eps)


def clip_boxes_to_image(boxes: Any, image_size: Sequence[int]) -> Any:
    """Clip xyxy boxes to an image extent."""

    boxes = _as_float_boxes(boxes, name="boxes")
    height, width = _normalize_hw(image_size, name="image size")
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    return torch.stack(
        (
            boxes[:, 0].clamp(min=0.0, max=float(width)),
            boxes[:, 1].clamp(min=0.0, max=float(height)),
            boxes[:, 2].clamp(min=0.0, max=float(width)),
            boxes[:, 3].clamp(min=0.0, max=float(height)),
        ),
        dim=1,
    )


def scale_boxes(boxes: Any, scale_factor: float | Sequence[float]) -> Any:
    """Scale xyxy boxes by a scalar, y/x pair, or explicit xyxy factors."""

    boxes = _as_float_boxes(boxes, name="boxes")
    factors = _scale_factors(scale_factor, device=boxes.device, dtype=boxes.dtype)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    return boxes * factors


def encode_point_boxes(points: Any, target_boxes: Any) -> Any:
    """Encode boxes as left/top/right/bottom distances from point priors."""

    points = _as_points(points)
    target_boxes = _as_float_boxes(target_boxes, name="target_boxes")
    target_boxes = target_boxes.to(device=points.device, dtype=points.dtype)
    if points.shape[0] != target_boxes.shape[0]:
        raise ValueError(
            f"point/box count mismatch: expected {points.shape[0]} boxes, got {target_boxes.shape[0]}."
        )
    if points.numel() == 0:
        return target_boxes.new_zeros((0, 4))
    return torch.stack(
        (
            points[:, 0] - target_boxes[:, 0],
            points[:, 1] - target_boxes[:, 1],
            target_boxes[:, 2] - points[:, 0],
            target_boxes[:, 3] - points[:, 1],
        ),
        dim=1,
    ).clamp(min=0)


def decode_point_boxes(points: Any, distances: Any) -> Any:
    """Decode left/top/right/bottom distances from point priors to boxes."""

    points = _as_points(points)
    distances = _as_float_boxes(distances, name="distances")
    distances = distances.to(device=points.device, dtype=points.dtype)
    if points.shape[0] != distances.shape[0]:
        raise ValueError(
            f"point/distance count mismatch: expected {points.shape[0]} distances, got {distances.shape[0]}."
        )
    if points.numel() == 0:
        return distances.new_zeros((0, 4))
    return torch.stack(
        (
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ),
        dim=1,
    )


def make_batched_nms_payload(
    boxes: Any,
    scores: Any,
    labels: Any,
    *,
    image_indices: Any | None = None,
) -> dict[str, Any]:
    """Validate detection tensors and add grouping indices for batched NMS."""

    boxes = _as_float_boxes(boxes, name="boxes")
    scores = _as_vector(scores, name="scores", dtype=boxes.dtype, device=boxes.device)
    labels = _as_vector(labels, name="labels", dtype=torch.long, device=boxes.device)
    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
        raise ValueError(
            "prediction payload count mismatch: boxes, scores, and labels must have the same length."
        )
    if image_indices is None:
        image_indices = torch.zeros_like(labels)
    else:
        image_indices = _as_vector(image_indices, name="image_indices", dtype=torch.long, device=boxes.device)
    if image_indices.shape[0] != boxes.shape[0]:
        raise ValueError("image_indices must match the number of prediction boxes.")
    if labels.numel() == 0:
        nms_indices = labels.new_zeros((0,))
    else:
        nms_indices = labels + image_indices * (labels.max() + 1)
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "image_indices": image_indices,
        "nms_indices": nms_indices,
    }


def prediction_payload_to_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the public per-image prediction contract from an NMS payload."""

    return {
        "boxes": payload["boxes"],
        "scores": payload["scores"],
        "labels": payload["labels"],
    }


def select_prediction_payload(payload: dict[str, Any], indices: Any) -> dict[str, Any]:
    """Index every tensor in a prediction payload."""

    return {key: value[indices] for key, value in payload.items()}


def _feature_sizes_from_inputs(
    feature_maps: Sequence[Any] | None,
    feature_sizes: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, int], ...]:
    if feature_maps is None and feature_sizes is None:
        raise ValueError("feature_maps or feature_sizes are required.")
    if feature_maps is not None and feature_sizes is not None:
        raise ValueError("Pass feature_maps or feature_sizes, not both.")
    if feature_maps is not None:
        sizes = []
        for feature in feature_maps:
            shape = getattr(feature, "shape", None)
            if shape is None or len(shape) < 2:
                raise ValueError("feature maps must expose at least spatial height and width dimensions.")
            sizes.append(_normalize_hw(shape[-2:], name="feature size"))
        return tuple(sizes)
    return tuple(_normalize_hw(size, name="feature size") for size in feature_sizes or ())


def _base_anchors(
    base_size: float,
    *,
    aspect_ratios: Sequence[float],
    scales: Sequence[float],
    device: Any | None,
    dtype: Any,
) -> Any:
    ratios = _positive_tensor(aspect_ratios, name="aspect ratios", device=device, dtype=dtype)
    scale_tensor = _positive_tensor(scales, name="anchor scales", device=device, dtype=dtype)
    height_ratios = torch.sqrt(ratios)
    width_ratios = 1.0 / height_ratios
    widths = (float(base_size) * width_ratios[:, None] * scale_tensor[None, :]).reshape(-1)
    heights = (float(base_size) * height_ratios[:, None] * scale_tensor[None, :]).reshape(-1)
    return torch.stack((-0.5 * widths, -0.5 * heights, 0.5 * widths, 0.5 * heights), dim=1)


def _box_centers(boxes: Any) -> tuple[Any, Any, Any, Any]:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    return widths, heights, ctr_x, ctr_y


def _box_area(boxes: Any) -> Any:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def _as_float_boxes(value: Any, *, name: str) -> Any:
    if not hasattr(value, "shape") or len(value.shape) != 2 or int(value.shape[1]) != 4:
        raise ValueError(f"{name} must be a tensor with shape (N, 4).")
    if not torch.is_floating_point(value):
        value = value.to(dtype=torch.float32)
    return value


def _as_points(value: Any) -> Any:
    if not hasattr(value, "shape") or len(value.shape) != 2 or int(value.shape[1]) != 2:
        raise ValueError("points must be a tensor with shape (N, 2).")
    if not torch.is_floating_point(value):
        value = value.to(dtype=torch.float32)
    return value


def _as_vector(value: Any, *, name: str, dtype: Any, device: Any) -> Any:
    if not hasattr(value, "shape") or len(value.shape) != 1:
        raise ValueError(f"{name} must be a 1D tensor.")
    if value.device != device or value.dtype != dtype:
        value = value.to(device=device, dtype=dtype)
    return value


def _validate_same_box_count(boxes1: Any, boxes2: Any) -> None:
    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(
            f"box count mismatch: expected {boxes1.shape[0]} boxes, got {boxes2.shape[0]}."
        )


def _normalize_hw(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain height and width.")
    return _positive_int(value[0], f"{name} height"), _positive_int(value[1], f"{name} width")


def _normalize_stride(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != 2:
            raise ValueError("stride entries must be scalars or (stride_y, stride_x) pairs.")
        return _positive_float(values[0], "stride_y"), _positive_float(values[1], "stride_x")
    stride = _positive_float(value, "stride")
    return stride, stride


def _normalize_weights(weights: Sequence[float]) -> tuple[float, float, float, float]:
    if len(weights) != 4:
        raise ValueError("bbox weights must contain four values.")
    return tuple(_positive_float(value, "bbox weight") for value in weights)  # type: ignore[return-value]


def _scale_factors(scale_factor: float | Sequence[float], *, device: Any, dtype: Any) -> Any:
    if isinstance(scale_factor, Iterable) and not isinstance(scale_factor, (str, bytes)):
        values = tuple(float(value) for value in scale_factor)
        if len(values) == 2:
            scale_y, scale_x = values
            values = (scale_x, scale_y, scale_x, scale_y)
        elif len(values) != 4:
            raise ValueError("scale_factor must be a scalar, (scale_y, scale_x), or four xyxy factors.")
    else:
        scalar = float(scale_factor)
        values = (scalar, scalar, scalar, scalar)
    if any(value < 0.0 for value in values):
        raise ValueError("scale_factor values must be non-negative.")
    return torch.tensor(values, device=device, dtype=dtype)


def _positive_tensor(values: Sequence[float], *, name: str, device: Any | None, dtype: Any) -> Any:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")
    tensor = torch.tensor(tuple(float(value) for value in values), device=device, dtype=dtype)
    if bool((tensor <= 0).any()):
        raise ValueError(f"{name} must contain only positive values.")
    return tensor


def _positive_int(value: Any, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive.")
    return integer


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


__all__ = [
    "DEFAULT_ANCHOR_SCALES",
    "DEFAULT_ANCHOR_SIZES",
    "DEFAULT_ASPECT_RATIOS",
    "FeatureMapSpec",
    "box_iou",
    "build_feature_map_specs",
    "clip_boxes_to_image",
    "decode_boxes",
    "decode_point_boxes",
    "encode_boxes",
    "encode_point_boxes",
    "generate_anchors",
    "generate_points",
    "generate_priors",
    "infer_strides",
    "make_batched_nms_payload",
    "prediction_payload_to_dict",
    "scale_boxes",
    "select_prediction_payload",
]
