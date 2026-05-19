"""Native model factories for the Lightning backend."""

from __future__ import annotations

from contextlib import nullcontext
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
    "efficientdet",
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


class SingleStageDetector(nn.Module):
    """Native single-stage detector composition for dense detector families."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        backbone_spec: Any,
        neck_spec: Any,
        head_spec: Any,
        loss_fn: nn.Module,
        postprocessor: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.backbone_spec = backbone_spec
        self.neck_spec = neck_spec
        self.head_spec = head_spec
        self.loss_fn = loss_fn
        self.postprocessor = postprocessor
        self.decoder = postprocessor

    def __call__(self, images, targets=None):
        return self.forward(images, targets=targets)

    def forward(self, images, targets=None):
        if targets is not None:
            return self.forward_loss(images, targets)
        return self.predict(images)

    def forward_loss(self, images, targets):
        """Return dense detector losses with gradient tracking enabled."""

        self._validate_inputs(images, targets=targets)
        feature_pyramids, head_outputs_per_image = self._head_outputs(images)
        return self.loss_fn(images, targets, feature_pyramids, head_outputs_per_image)

    def predict(self, images):
        """Return postprocessed per-image predictions without tracking gradients."""

        self._validate_inputs(images)
        no_grad = torch.no_grad() if hasattr(torch, "no_grad") else nullcontext()
        with no_grad:
            feature_pyramids, head_outputs_per_image = self._head_outputs(images)
            return [
                self.postprocess(image, pyramid, head_outputs)
                for image, pyramid, head_outputs in zip(images, feature_pyramids, head_outputs_per_image)
            ]

    def extract_features(self, image):
        """Run backbone and optional neck for one CHW image tensor."""

        batched = image.unsqueeze(0)
        features = self.backbone(batched)
        if self.neck is None:
            return features
        return self.neck(features)

    def forward_head(self, feature_pyramid):
        """Run the dense prediction head for one feature pyramid."""

        return self.head(feature_pyramid)

    def postprocess(self, image, feature_pyramid, head_outputs):
        """Run the dense postprocess path for one image."""

        return self.postprocessor(image, feature_pyramid, head_outputs)

    def _validate_inputs(self, images, *, targets=None):
        if not images:
            raise ValueError("SingleStageDetector requires at least one image.")
        for image in images:
            if not hasattr(image, "shape") or not hasattr(image, "unsqueeze"):
                raise TypeError(
                    "SingleStageDetector expects tensor-like images with 'shape' and 'unsqueeze'."
                )
        if targets is not None and len(targets) != len(images):
            raise ValueError("Number of targets must match number of images.")

    def _head_outputs(self, images):
        feature_pyramids = []
        head_outputs_per_image = []
        for image in images:
            pyramid = self.extract_features(image)
            head_outputs = self.forward_head(pyramid)
            feature_pyramids.append(pyramid)
            head_outputs_per_image.append(head_outputs)
        return feature_pyramids, head_outputs_per_image


class NativeRetinaNetModel(SingleStageDetector):
    """Backward-compatible RetinaNet model name for dense native detectors."""


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


def build_detector(
    name: str,
    *,
    num_classes: int = 1,
    detector_spec=None,
    in_channels: int = 3,
    pretrained: bool = True,
    **overrides: Any,
) -> SingleStageDetector:
    """Build a native detector module from the public suite defaults."""

    if detector_spec is None:
        from ..suite import build_detector as build_detector_spec

        detector_spec = build_detector_spec(
            name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            **overrides,
        )
    model = build_native_model(
        name,
        num_classes=int(num_classes),
        in_channels=int(in_channels),
        detector_spec=detector_spec,
    )
    if not isinstance(model, SingleStageDetector):
        raise ValueError(
            f"build_detector(name={name!r}) expected a single-stage detector, "
            f"got {type(model).__name__}."
        )
    return model


def _merge_detr_predictions(predictions):
    pred_logits = torch.cat([prediction["pred_logits"] for prediction in predictions], dim=0)
    pred_boxes = torch.cat([prediction["pred_boxes"] for prediction in predictions], dim=0)
    return {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
    }
