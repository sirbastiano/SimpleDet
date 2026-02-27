"""Inference entry points for the detector public API."""

from __future__ import annotations

import os
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._deps import require_dependency

SUPPORTED_CHECKPOINT_FORMATS = (".pth", ".pt", ".ckpt")


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a checkpoint path is missing or uses an unsupported format."""

    def __init__(self, checkpoint: str, *, supported_formats: Sequence[str]):
        self.checkpoint = checkpoint
        self.supported_formats = tuple(supported_formats)
        supported = ", ".join(self.supported_formats) or "<none>"
        super().__init__(
            "Checkpoint not found or unsupported format "
            f"'{checkpoint}'. Supported formats: {supported}"
        )


def _validate_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise CheckpointNotFoundError(str(path), supported_formats=SUPPORTED_CHECKPOINT_FORMATS)
    if path.suffix.lower() not in SUPPORTED_CHECKPOINT_FORMATS:
        raise CheckpointNotFoundError(str(path), supported_formats=SUPPORTED_CHECKPOINT_FORMATS)
    return path


def _coerce_device(device: str) -> str:
    device_value = str(device).strip().lower()
    if device_value == "auto":
        return "cpu"
    return device_value


def _extract_state_dict(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"], payload
    if isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"], payload
    if all(hasattr(value, "shape") for value in payload.values()):
        return dict(payload), payload
    raise ValueError("Invalid checkpoint payload: expected 'model_state_dict' or tensor mapping.")


def _build_torchvision_model(model_name: str, num_classes: int):
    require_dependency("torchvision", "infer")
    import torch
    import torchvision

    model_factories = {
        "faster_rcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
        "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    }
    factory = model_factories.get(model_name.strip())
    if factory is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported checkpoint-compatible models: "
            f"{', '.join(model_factories)}."
        )

    kwargs = {"num_classes": int(num_classes)}
    signature = inspect.signature(factory)
    for key in ("weights", "weights_backbone", "weights_v2"):
        if key in signature.parameters:
            kwargs[key] = None
    return factory(**kwargs)


def _infer_num_classes(model_name: str, state_dict: dict[str, Any]) -> int | None:
    if model_name == "faster_rcnn_resnet50_fpn":
        key = "roi_heads.box_predictor.cls_score.weight"
        weight = state_dict.get(key)
        if hasattr(weight, "shape") and weight.ndim == 2:
            return int(weight.shape[0])

    if model_name == "retinanet_resnet50_fpn":
        key = "head.classification_head.cls_logits.weight"
        weight = state_dict.get(key)
        if hasattr(weight, "shape") and weight.ndim == 2:
            # RetinaNet in torchvision uses a fixed 9 anchors per location by default.
            if weight.shape[0] % 9 == 0:
                return int(weight.shape[0] // 9)

    if model_name == "ssd300_vgg16":
        key = "head.classification_head.conv1.conv_list.0.0.weight"
        weight = state_dict.get(key)
        if hasattr(weight, "shape") and weight.ndim == 4:
            # 4x4 default anchor set is the default per-location anchor count for SSD300.
            if weight.shape[0] % 4 == 0:
                return int(weight.shape[0] // 4)

    return None


def _load_state_dict(model: Any, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict, strict=False)
        return
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                stripped[key[7:]] = value
            else:
                stripped[f"module.{key}"] = value
        model.load_state_dict(stripped, strict=False)


def _to_tensor(image: Any, *, device: str):
    import torch

    if hasattr(image, "shape") and getattr(image, "ndim", 0) == 4:
        return _to_tensor_batch(image, device=device)

    if hasattr(image, "ndim") and getattr(image, "ndim", 0) == 3 and image.shape[0] in {1, 3}:
        tensor = torch.as_tensor(image, dtype=torch.float32)
    elif hasattr(image, "ndim") and getattr(image, "ndim", 0) == 2:
        tensor = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
    elif hasattr(image, "ndim") and getattr(image, "ndim", 0) == 3 and getattr(image, "shape", [None, None, None])[-1] in {1, 3}:
        tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
    else:
        try:
            import torchvision.io as tvio

            if isinstance(image, str | bytes | os.PathLike):
                tensor = tvio.read_image(str(image)).to(dtype=torch.float32)
            else:
                require_dependency("PIL", "infer")
                from PIL import Image

                if isinstance(image, Image.Image):
                    tensor = torch.as_tensor(image.convert("RGB"), dtype=torch.float32).permute(
                        2, 0, 1
                    )
                else:
                    raise TypeError
        except Exception as exc:
            raise TypeError(
                "Unsupported image input type. "
                "Pass a pathlib-style path, PyTorch tensor, NumPy array, or Pillow image."
            ) from exc

    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")

    return tensor.to(device=device)


def _to_tensor_batch(batch_images: Sequence[Any], *, device: str):
    import torch

    if len(batch_images) == 0:
        return torch.empty((0, 3, 1, 1), dtype=torch.float32, device=device)
    tensors = [_to_tensor(image, device=device) for image in batch_images]
    return tensors


def _predict_batch(
    model: Any,
    batch: Sequence[Any],
    *,
    score_threshold: float,
    max_detections: int | None,
):
    import torch

    model.eval()
    with torch.no_grad():
        outputs = model(list(batch))

    results = []
    for output in outputs:
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu().long()
        scores = output["scores"].detach().cpu()
        keep = scores >= score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        if max_detections is not None:
            order = torch.argsort(scores, descending=True)
            keep_k = order[:max_detections]
            boxes = boxes[keep_k]
            labels = labels[keep_k]
            scores = scores[keep_k]
        else:
            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]
            labels = labels[order]
            scores = scores[order]

        results.append(
            {
                "boxes": boxes.tolist(),
                "class_ids": labels.tolist(),
                "scores": scores.tolist(),
            }
        )
    return results


@dataclass(frozen=True)
class _LoadedModel:
    model: Any
    score_threshold: float
    max_detections: int | None
    device: str

    def predict(self, image: Any):
        if isinstance(image, (tuple, list)):
            tensors = _to_tensor_batch(image, device=self.device)
            outputs = _predict_batch(
                self.model,
                tensors,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
            )
            return outputs

        tensor = _to_tensor(image, device=self.device)
        outputs = _predict_batch(
            self.model,
            [tensor],
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
        )
        return outputs[0]


_active_model: _LoadedModel | None = None


def load_model(
    checkpoint: str,
    *,
    device: str = "cpu",
    model_name: str | None = None,
    num_classes: int | None = None,
    score_threshold: float = 0.05,
    max_detections: int | None = None,
):
    """Load a trained torchvision detector checkpoint and return a lightweight predictor."""
    path = _validate_checkpoint_path(checkpoint)
    require_dependency("torch", "infer")
    import torch

    checkpoint_data = torch.load(path, map_location=_coerce_device(device))

    if not isinstance(checkpoint_data, dict):
        raise ValueError("Invalid checkpoint format: expected a mapping payload.")

    state_dict, metadata = _extract_state_dict(checkpoint_data)
    resolved_model_name = model_name or str(metadata.get("model_name", "")).strip()
    if not resolved_model_name:
        raise ValueError("Checkpoint missing required 'model_name'.")

    resolved_num_classes = (
        int(num_classes)
        if num_classes is not None
        else metadata.get("num_classes")
    )
    if resolved_num_classes is None:
        inferred = _infer_num_classes(resolved_model_name, state_dict)
        if inferred is None:
            raise ValueError(
                "Unable to infer checkpoint num_classes. "
                "Pass `num_classes=...` when calling `load_model`."
            )
        resolved_num_classes = inferred

    resolved_num_classes = int(resolved_num_classes)
    if resolved_num_classes <= 0:
        raise ValueError("`num_classes` must be a positive integer.")

    detector = _build_torchvision_model(resolved_model_name, resolved_num_classes)
    _load_state_dict(detector, state_dict)
    detector.to(_coerce_device(device))

    result = _LoadedModel(
        model=detector,
        score_threshold=float(score_threshold),
        max_detections=None if max_detections is None else int(max_detections),
        device=_coerce_device(device),
    )
    global _active_model
    _active_model = result
    return result


def predict(
    image: Any,
    *,
    model: _LoadedModel | None = None,
    score_threshold: float | None = None,
):
    """Run deterministic inference on one image or a batch of images."""
    active_model = model or _active_model
    if active_model is None:
        raise RuntimeError("No model loaded. Call `load_model(checkpoint)` first.")

    if score_threshold is not None:
        active_model = _LoadedModel(
            model=active_model.model,
            score_threshold=float(score_threshold),
            max_detections=active_model.max_detections,
            device=active_model.device,
        )
    return active_model.predict(image)


def detect(
    *,
    pipeline: Any | None = None,
    build: bool = True,
    **pipeline_kwargs: Any,
):
    """Run inference using a detector pipeline and return detection outputs."""
    from ._deps import require_detector_runtime

    require_detector_runtime("detect")

    from simpledet.api import ObjectDetectionPipeline

    if pipeline is None:
        if not pipeline_kwargs:
            raise TypeError(
                "Provide either `pipeline` or the constructor keyword arguments "
                "required by ObjectDetectionPipeline."
            )
        pipeline = ObjectDetectionPipeline(**pipeline_kwargs)

    if build:
        pipeline.build()

    return pipeline.test()
