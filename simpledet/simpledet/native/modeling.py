"""Native model factories for the Lightning backend."""

from __future__ import annotations

from typing import Any

from ..detectors._deps import require_dependency
from ..suite.catalog import resolve_architecture_name
require_dependency("torch", "native")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from .roi import NativeRoIBackbone  # noqa: E402


SUPPORTED_NATIVE_ARCHITECTURES = {
    "retinanet",
    "fcos",
    "atss",
    "gfl",
    "vfnet",
    "fovea",
    "foveabox",
    "reppoints",
    "yolof",
    "centernet",
    "yolo",
    "yolo3",
    "yolo_v3",
    "yolov3",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8",
    "yolox",
    "rtmdet",
    "tood",
    "ssd",
    "sabl",
    "solov2",
    "grid_rcnn",
    "cascade_rcnn",
    "faster_rcnn",
    "mask_rcnn",
    "detr",
    "deformable_detr",
    "conditional_detr",
    "dino",
}


class NativeRetinaNetModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        backbone_spec: Any,
        neck_spec: Any,
        head_spec: Any,
        loss_fn: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.backbone_spec = backbone_spec
        self.neck_spec = neck_spec
        self.head_spec = head_spec
        self.loss_fn = loss_fn
        self.decoder = decoder

    def __call__(self, images, targets=None):
        return self.forward(images, targets=targets)

    def forward(self, images, targets=None):
        self._validate_inputs(images, targets=targets)
        return self._forward_tensor_batch(images, targets=targets)

    def _validate_inputs(self, images, *, targets=None):
        if not images:
            raise ValueError("NativeRetinaNetModel requires at least one image.")
        for image in images:
            if not hasattr(image, "shape") or not hasattr(image, "unsqueeze"):
                raise TypeError(
                    "NativeRetinaNetModel expects tensor-like images with 'shape' and 'unsqueeze'."
                )
        if targets is not None and len(targets) != len(images):
            raise ValueError("Number of targets must match number of images.")

    def _forward_tensor_batch(self, images, *, targets=None):
        feature_pyramids = []
        head_outputs_per_image = []
        for image in images:
            batched = image.unsqueeze(0)
            features = self.backbone(batched)
            pyramid = self.neck(features)
            head_outputs = self.head(pyramid)
            feature_pyramids.append(pyramid)
            head_outputs_per_image.append(head_outputs)

        if targets is not None:
            return self.loss_fn(images, targets, feature_pyramids, head_outputs_per_image)

        return [
            self.decoder(image, pyramid, head_outputs)
            for image, pyramid, head_outputs in zip(images, feature_pyramids, head_outputs_per_image)
        ]


NativeFeatureExtractorBackbone = NativeRoIBackbone


class NativeDetrModel(nn.Module):
    """Reserved transformer-family model boundary for a future DETR rollout."""
    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module,
        postprocessor: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.postprocessor = postprocessor

    def forward(self, images, targets=None):
        if not images:
            raise ValueError("NativeDetrModel requires at least one image.")
        predictions = []
        for image in images:
            if not hasattr(image, "shape") or not hasattr(image, "unsqueeze"):
                raise TypeError(
                    "NativeDetrModel expects tensor-like images with 'shape' and 'unsqueeze'."
                )
            batched = image.unsqueeze(0)
            features = self.backbone(batched)
            pyramid = self.neck(features)
            predictions.append(self.decoder(pyramid))
        if targets is not None:
            if len(targets) != len(images):
                raise ValueError("Number of targets must match number of images.")
            merged = _merge_detr_predictions(predictions)
            return self.loss_fn(merged, targets)
        return [self.postprocessor(prediction) for prediction in predictions]


def build_native_model(
    architecture: str,
    *,
    num_classes: int,
    in_channels: int = 3,
    detector_spec=None,
) -> Any:
    normalized = resolve_architecture_name(architecture)

    if detector_spec is None:
        raise TypeError("Native model construction now requires a detector_spec.")
    spec_architecture = str(getattr(detector_spec, "architecture", "")).strip()
    if spec_architecture:
        if resolve_architecture_name(spec_architecture) != normalized:
            raise ValueError(
                f"Requested architecture '{architecture}' is incompatible with detector_spec "
                f"architecture '{spec_architecture}'."
            )

    from ..extensions import DETECTORS
    from .assemblers import build_native_components

    try:
        detector_name = DETECTORS.resolve_name(normalized)
    except KeyError:
        detector_name = ""

    if normalized not in SUPPORTED_NATIVE_ARCHITECTURES and not detector_name:
        supported = ", ".join(sorted(set(SUPPORTED_NATIVE_ARCHITECTURES) | set(DETECTORS.names())))
        raise ValueError(
            f"Unsupported native architecture '{architecture}'. Supported: {supported}."
        )

    components = build_native_components(detector_spec)
    assembler = DETECTORS.get(detector_name or normalized)
    return assembler(components, num_classes=int(num_classes))


def _merge_detr_predictions(predictions):
    pred_logits = torch.cat([prediction["pred_logits"] for prediction in predictions], dim=0)
    pred_boxes = torch.cat([prediction["pred_boxes"] for prediction in predictions], dim=0)
    return {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
    }
