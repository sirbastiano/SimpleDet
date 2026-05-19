"""Inference entry points for the detector public API."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


class ImageLoadingError(RuntimeError):
    """Raised when an existing image path cannot be decoded for prediction."""

    def __init__(self, image_path: str | Path, reason: str) -> None:
        self.image_path = str(image_path)
        super().__init__(f"Unable to load image '{self.image_path}': {reason}")


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


def _coerce_categories(raw: Any) -> tuple[str, ...]:
    if raw is None:
        raise TypeError("`pipeline_kwargs` must define `categories` for native mode.")
    if isinstance(raw, str):
        return (raw,)
    try:
        normalized = tuple(str(item) for item in raw)
    except TypeError as exc:
        raise TypeError("`categories` must be a sequence of class names.") from exc
    if not normalized:
        raise TypeError("`categories` must contain at least one value.")
    return normalized


def _resolve_class_names(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return (raw,)
    try:
        normalized = tuple(str(item) for item in raw)
    except TypeError as exc:
        raise TypeError("`class_names` must be a sequence of class names.") from exc
    return normalized or None


def _coerce_in_channels(in_channels: Any, *, tif_channels_to_load: Any = None) -> int:
    if in_channels is None:
        if tif_channels_to_load is None:
            raise TypeError(
                "Provide `in_channels` or `tif_channels_to_load` for native inference."
            )
        try:
            in_channels = len(tif_channels_to_load)
        except TypeError as exc:
            raise TypeError("`tif_channels_to_load` must be an iterable.") from exc
    value = int(in_channels)
    if value <= 0:
        raise TypeError("`in_channels` must be a positive integer.")
    return value


def _run_native_inference_from_kwargs(pipeline_kwargs: Mapping[str, Any]) -> Any:
    dataset_root = (
        pipeline_kwargs.get("data_root")
        or pipeline_kwargs.get("dataset_root")
        or pipeline_kwargs.get("data_folder")
    )
    if dataset_root is None:
        raise TypeError(
            "When using `pipeline_kwargs`, pass `data_root` (or legacy `data_folder`) "
            "for native inference."
        )

    from simpledet.api import run_inference

    resolved_kwargs = dict(pipeline_kwargs)
    resolved_kwargs["data_root"] = str(dataset_root)
    resolved_kwargs["categories"] = _coerce_categories(pipeline_kwargs.get("categories"))
    resolved_kwargs["in_channels"] = _coerce_in_channels(
        pipeline_kwargs.get("in_channels"),
        tif_channels_to_load=pipeline_kwargs.get("tif_channels_to_load"),
    )
    resolved_kwargs.pop("dataset_root", None)
    resolved_kwargs.pop("data_folder", None)

    return run_inference(**resolved_kwargs)


def _build_torchvision_model(model_name: str, num_classes: int):
    require_dependency("torchvision", "infer")
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
        if isinstance(image, (str, bytes, os.PathLike)):
            path = _path_from_input(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            try:
                import torchvision.io as tvio
            except Exception as exc:
                raise TypeError(
                    "Unsupported image input type. "
                    "Install torchvision or pass a PyTorch tensor, NumPy array, or Pillow image."
                ) from exc
            try:
                tensor = tvio.read_image(str(path)).to(dtype=torch.float32)
            except Exception as exc:
                raise TypeError(f"Unable to decode image file '{path}'.") from exc
        else:
            try:
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


@dataclass(frozen=True)
class _InferenceImage:
    tensor: Any
    metadata: dict[str, Any]


def _is_path_input(image: Any) -> bool:
    return isinstance(image, (str, bytes, os.PathLike))


def _path_from_input(image: str | bytes | os.PathLike[str]) -> Path:
    return Path(os.fsdecode(image)).expanduser()


def _load_image_for_prediction(
    image: Any,
    *,
    device: str,
    metadata: Mapping[str, Any] | None = None,
) -> _InferenceImage:
    resolved_metadata = dict(metadata or {})
    if _is_path_input(image):
        path = _path_from_input(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            tensor = _to_tensor(path, device=device)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ImageLoadingError(path, str(exc)) from exc
        resolved_metadata.update(_image_metadata(tensor, path=path))
        return _InferenceImage(tensor=tensor, metadata=resolved_metadata)

    tensor = _to_tensor(image, device=device)
    resolved_metadata.update(_image_metadata(tensor, path=None))
    return _InferenceImage(tensor=tensor, metadata=resolved_metadata)


def _image_metadata(tensor: Any, *, path: Path | None) -> dict[str, Any]:
    shape = tuple(int(value) for value in getattr(tensor, "shape", ()) or ())
    metadata: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "file_name": path.name if path is not None else None,
        "shape": list(shape),
    }
    if len(shape) >= 3:
        metadata.update({"channels": shape[-3], "height": shape[-2], "width": shape[-1]})
    elif len(shape) >= 2:
        metadata.update({"channels": None, "height": shape[-2], "width": shape[-1]})
    else:
        metadata.update({"channels": None, "height": None, "width": None})
    return metadata


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
    class_names: tuple[str, ...] | None = None

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
    class_names: Sequence[str] | None = None,
):
    """Load a trained torchvision detector checkpoint and return a lightweight predictor.

    Checkpoints are loaded with ``torch.load``. Only load checkpoint files from
    trusted sources.
    """
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
    resolved_class_names = _resolve_class_names(
        class_names or metadata.get("class_names") or metadata.get("categories")
    )

    detector = _build_torchvision_model(resolved_model_name, resolved_num_classes)
    _load_state_dict(detector, state_dict)
    detector.to(_coerce_device(device))

    result = _LoadedModel(
        model=detector,
        score_threshold=float(score_threshold),
        max_detections=None if max_detections is None else int(max_detections),
        device=_coerce_device(device),
        class_names=resolved_class_names,
    )
    global _active_model
    _active_model = result
    return result


def load_checkpoint_for_inference(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    model_name: str | None = None,
    num_classes: int | None = None,
    score_threshold: float = 0.05,
    max_detections: int | None = None,
    class_names: Sequence[str] | None = None,
):
    """Load a lightweight torchvision-compatible checkpoint for image prediction.

    This delegates to ``load_model`` and inherits its trusted-checkpoint
    requirement.
    """

    return load_model(
        str(checkpoint),
        device=device,
        model_name=model_name,
        num_classes=num_classes,
        score_threshold=score_threshold,
        max_detections=max_detections,
        class_names=class_names,
    )


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
            class_names=active_model.class_names,
        )
    return active_model.predict(image)


def predict_image(
    model: Any,
    image_path: Any,
    *,
    class_names: Sequence[str] | None = None,
    device: str = "cpu",
    score_threshold: float | None = None,
    max_detections: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one structured detection payload for a single image.

    The payload contains ``boxes``, ``scores``, ``labels``, ``class_names``, and
    ``metadata``. Missing paths raise ``FileNotFoundError`` with the path, while
    decode failures raise ``ImageLoadingError`` with the offending path.
    """

    return predict_batch(
        model,
        [image_path],
        class_names=class_names,
        device=device,
        score_threshold=score_threshold,
        max_detections=max_detections,
        metadata=[metadata or {}],
    )[0]


def predict_batch(
    model: Any,
    image_paths: Sequence[Any],
    *,
    class_names: Sequence[str] | None = None,
    device: str = "cpu",
    score_threshold: float | None = None,
    max_detections: int | None = None,
    metadata: Sequence[Mapping[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Return structured detection payloads for a batch of images.

    Batch prediction is fail-fast: one missing or undecodable image aborts the
    batch and the raised exception includes the offending path.
    """

    if _is_path_input(image_paths) or not isinstance(image_paths, Sequence):
        raise TypeError("`image_paths` must be a non-string sequence of images or paths.")
    if metadata is not None and len(metadata) != len(image_paths):
        raise ValueError("`metadata` must match the number of images.")
    if len(image_paths) == 0:
        return []

    loaded_images = [
        _load_image_for_prediction(
            image,
            device=_coerce_device(device),
            metadata=None if metadata is None else metadata[index],
        )
        for index, image in enumerate(image_paths)
    ]
    tensors = [item.tensor for item in loaded_images]
    outputs = _run_inference_model(model, tensors)
    normalized_outputs = _normalize_batch_outputs(outputs, expected=len(tensors))
    resolved_class_names = _resolve_class_names(class_names) or getattr(model, "class_names", None)
    return [
        _structured_prediction(
            output,
            metadata=loaded.metadata,
            class_names=resolved_class_names,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        for output, loaded in zip(normalized_outputs, loaded_images)
    ]


def export_predictions(
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
    *,
    indent: int = 2,
) -> dict[str, Any]:
    """Return and optionally write the JSON-serializable SimpleDet prediction payload."""

    if isinstance(predictions, Mapping):
        prediction_items = [predictions]
    else:
        prediction_items = list(predictions)
    payload = {
        "format": "simpledet_predictions",
        "version": 1,
        "predictions": [_exportable_prediction(item) for item in prediction_items],
    }
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return payload


def _run_inference_model(model: Any, tensors: Sequence[Any]) -> Any:
    if model is None:
        raise TypeError("`model` is required for predict_image and predict_batch.")
    if hasattr(model, "eval"):
        model.eval()
    predict_fn = getattr(model, "predict", None)
    callable_model = predict_fn if callable(predict_fn) else model
    if not callable(callable_model):
        raise TypeError("`model` must be callable or expose a predict(images) method.")
    with _inference_context():
        return callable_model(list(tensors))


def _inference_context():
    try:
        import torch
    except ImportError:
        return nullcontext()
    no_grad = getattr(torch, "no_grad", None)
    return no_grad() if callable(no_grad) else nullcontext()


def _normalize_batch_outputs(outputs: Any, *, expected: int) -> list[Any]:
    if isinstance(outputs, Mapping):
        normalized = [outputs]
    elif isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes, bytearray)):
        normalized = list(outputs)
    else:
        raise TypeError("Inference model must return a mapping or a sequence of mappings.")
    if len(normalized) != expected:
        raise ValueError(
            f"Inference model returned {len(normalized)} payloads for {expected} input images."
        )
    return normalized


def _structured_prediction(
    output: Any,
    *,
    metadata: Mapping[str, Any],
    class_names: Sequence[str] | None,
    score_threshold: float | None,
    max_detections: int | None,
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise TypeError("Each inference output must be a mapping.")
    boxes = _normalize_boxes(output.get("boxes", []))
    scores = _normalize_scores(output.get("scores", []), count=len(boxes))
    labels = _normalize_labels(output.get("labels", output.get("class_ids", [])), count=len(boxes))
    detections = list(zip(boxes, scores, labels))
    if score_threshold is not None:
        threshold = float(score_threshold)
        detections = [item for item in detections if float(item[1]) >= threshold]
    detections.sort(key=lambda item: float(item[1]), reverse=True)
    if max_detections is not None:
        detections = detections[: max(0, int(max_detections))]
    resolved_boxes = [box for box, _score, _label in detections]
    resolved_scores = [score for _box, score, _label in detections]
    resolved_labels = [label for _box, _score, label in detections]
    names = [_class_name_for_label(label, class_names) for label in resolved_labels]
    return {
        "boxes": resolved_boxes,
        "scores": resolved_scores,
        "labels": resolved_labels,
        "class_names": names,
        "metadata": dict(metadata),
    }


def _normalize_boxes(value: Any) -> list[list[float]]:
    data = _to_plain_data(value)
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError("Prediction `boxes` must be a sequence.")
    if not data:
        return []
    if all(not isinstance(item, (list, tuple)) for item in data):
        data = [data]
    return [[float(coord) for coord in box] for box in data]


def _normalize_scores(value: Any, *, count: int) -> list[float]:
    data = _to_plain_data(value)
    if data in (None, []):
        return [1.0] * count
    if not isinstance(data, list):
        data = [data]
    scores = [float(item) for item in data]
    if len(scores) != count:
        raise ValueError("Prediction `scores` must match the number of boxes.")
    return scores


def _normalize_labels(value: Any, *, count: int) -> list[Any]:
    data = _to_plain_data(value)
    if data in (None, []):
        return [0] * count
    if not isinstance(data, list):
        data = [data]
    labels = [_coerce_label(item) for item in data]
    if len(labels) != count:
        raise ValueError("Prediction `labels` must match the number of boxes.")
    return labels


def _coerce_label(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_float.is_integer():
        return int(as_float)
    return as_float


def _class_name_for_label(label: Any, class_names: Sequence[str] | None) -> str:
    if class_names is None:
        return str(label)
    try:
        label_index = int(label)
    except (TypeError, ValueError):
        return str(label)
    if label_index == 0:
        return "background"
    index = label_index - 1
    if 0 <= index < len(class_names):
        return str(class_names[index])
    return str(label)


def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _exportable_prediction(prediction: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(prediction, Mapping):
        raise TypeError("Predictions must be mapping payloads.")
    return {
        "boxes": _to_plain_data(prediction.get("boxes", [])),
        "scores": _to_plain_data(prediction.get("scores", [])),
        "labels": _to_plain_data(prediction.get("labels", [])),
        "class_names": _to_plain_data(prediction.get("class_names", [])),
        "metadata": _to_jsonable(prediction.get("metadata", {})),
    }


def _to_jsonable(value: Any) -> Any:
    value = _to_plain_data(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def detect(
    *,
    pipeline: Any | None = None,
    build: bool = True,
    **pipeline_kwargs: Any,
):
    """Run inference using a detector pipeline and return detection outputs."""
    from ._deps import require_detector_runtime

    require_detector_runtime("detect")
    if pipeline is None:
        if not pipeline_kwargs:
            raise TypeError(
                "Provide either `pipeline` or the constructor keyword arguments "
                "required by the native inference backend."
            )
        return _run_native_inference_from_kwargs(pipeline_kwargs)

    if build:
        pipeline.build()

    return pipeline.test()
