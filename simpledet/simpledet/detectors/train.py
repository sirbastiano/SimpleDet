"""Training entry points for the detector public API."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import json
from pathlib import Path
import inspect
from collections.abc import Mapping
from typing import Any

from ._deps import require_dependency
from .data import load_dataset


SUPPORTED_MODELS = {
    "faster_rcnn_resnet50_fpn": "Faster R-CNN ResNet-50 FPN",
    "ssd300_vgg16": "SSD300 VGG-16",
    "retinanet_resnet50_fpn": "RetinaNet ResNet-50 FPN",
}


@dataclass(frozen=True)
class _ResolvedTrainConfig:
    dataset_path: str
    output_dir: str
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    seed: int
    device: str
    format: str | None
    train_split: str
    num_workers: int
    num_classes: int | None


def _parse_config(config: Any) -> _ResolvedTrainConfig:
    if isinstance(config, Mapping):
        data = dict(config)
    elif hasattr(config, "__dict__"):
        data = dict(vars(config))
    else:
        raise TypeError(
            "`config` must be a mapping or an object with attributes."
        )

    def _get(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return default

    dataset_path = _get(
        "dataset",
        "dataset_path",
        "data",
        "data_path",
        "data_root",
        default=None,
    )
    if dataset_path is None:
        raise TypeError("`config` must define a dataset path (dataset/data_path).")

    model_name = str(_get("model_name", default="faster_rcnn_resnet50_fpn")).strip()
    if not model_name:
        model_name = "faster_rcnn_resnet50_fpn"

    output_dir = _get("output_dir", default="output")
    format_value = _get("format", default=None)
    if format_value is not None:
        format_value = str(format_value)

    return _ResolvedTrainConfig(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        model_name=model_name,
        epochs=int(_get("epochs", default=1)),
        batch_size=int(_get("batch_size", default=2)),
        learning_rate=float(_get("learning_rate", default=1e-3)),
        optimizer=str(_get("optimizer", default="sgd")).strip().lower(),
        seed=int(_get("seed", default=71)),
        device=str(_get("device", default="cpu")),
        format=format_value,
        train_split=str(_get("train_split", default="train")),
        num_workers=int(_get("num_workers", default=0)),
        num_classes=_get("num_classes", default=None),
    )


def _supported_model_names() -> list[str]:
    return sorted(SUPPORTED_MODELS)


def _validate_model_name(model_name: str) -> str:
    candidates = _supported_model_names()
    normalized = model_name.strip().lower()
    for candidate in candidates:
        if normalized == candidate:
            return candidate

    suggestion = get_close_matches(normalized, candidates, n=3, cutoff=0.6)
    if suggestion:
        suggestion_text = f" Did you mean: {', '.join(suggestion)}?"
    else:
        suggestion_text = ""
    raise ValueError(
        f"Unknown model '{model_name}'. Supported models: {', '.join(candidates)}."
        f"{suggestion_text}"
    )


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


def _coerce_in_channels(in_channels: Any, *, tif_channels_to_load: Any = None) -> int:
    if in_channels is None:
        if tif_channels_to_load is None:
            raise TypeError(
                "Provide `in_channels` or `tif_channels_to_load` for native training mode."
            )
        try:
            in_channels = len(tif_channels_to_load)
        except TypeError as exc:
            raise TypeError("`tif_channels_to_load` must be an iterable.") from exc
    value = int(in_channels)
    if value <= 0:
        raise TypeError("`in_channels` must be a positive integer.")
    return value


def _run_native_training_from_kwargs(pipeline_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    dataset_root = (
        pipeline_kwargs.get("data_root")
        or pipeline_kwargs.get("dataset_root")
        or pipeline_kwargs.get("data_folder")
    )
    if dataset_root is None:
        raise TypeError(
            "When using `pipeline_kwargs`, pass `data_root` (or legacy `data_folder`) "
            "for native training."
        )

    from simpledet.api import run_native_training

    resolved_kwargs = dict(pipeline_kwargs)
    resolved_kwargs["data_root"] = str(dataset_root)
    resolved_kwargs["categories"] = _coerce_categories(pipeline_kwargs.get("categories"))
    resolved_kwargs["in_channels"] = _coerce_in_channels(
        pipeline_kwargs.get("in_channels"),
        tif_channels_to_load=pipeline_kwargs.get("tif_channels_to_load"),
    )
    resolved_kwargs.pop("dataset_root", None)
    resolved_kwargs.pop("data_folder", None)

    return run_native_training(**resolved_kwargs)


def _build_optimizer(model: Any, learning_rate: float, optimizer: str, *, seed: int):
    import torch

    torch.manual_seed(seed)
    normalized = optimizer.strip().lower()
    if normalized == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if normalized == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)

    valid = ("sgd", "adam", "adamw")
    raise ValueError(f"Unsupported optimizer '{optimizer}'. Supported: {', '.join(valid)}.")


def _build_torchvision_model(model_name: str, num_classes: int):
    import torchvision

    # Keep imports lazy so import errors are surfaced only when the train path is used.
    require_dependency("torchvision", "train")

    model_factories = {
        "faster_rcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
        "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    }
    factory = model_factories[model_name]
    kwargs = {"num_classes": num_classes}
    signature = inspect.signature(factory)
    for key in ("weights", "weights_backbone", "weights_v2"):
        if key in signature.parameters:
            kwargs[key] = None

    model = factory(**kwargs)
    return model


def _build_dataset(
    dataset_payload: Mapping[str, Any],
    *,
    device: str,
    train_split: str = "train",
):
    import torch
    from torch.utils.data import Dataset

    target_split = (train_split or "train").strip().lower()
    samples = [
        sample
        for sample in list(dataset_payload.get("samples", []))
        if str(sample.get("split", target_split)).strip().lower() == target_split
    ]

    class _CocoDetectionDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            sample = samples[index]
            width = int(sample.get("width") or 1)
            height = int(sample.get("height") or 1)
            image = torch.zeros((3, max(height, 1), max(width, 1)), dtype=torch.float32)

            annotations = sample.get("annotations", [])
            boxes = []
            labels = []
            area = []
            iscrowd = []
            for annotation in annotations:
                bbox = annotation.get("bbox", {})
                boxes.append([
                    float(bbox.get("x_min", 0.0)),
                    float(bbox.get("y_min", 0.0)),
                    float(bbox.get("x_max", 0.0)),
                    float(bbox.get("y_max", 0.0)),
                ])
                labels.append(int(annotation.get("category_id", 0)))
                area.append(float(annotation.get("area", 0.0)))
                iscrowd.append(int(annotation.get("iscrowd", 0)))

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([int(sample.get("image_id", index))], dtype=torch.int64),
                "area": torch.as_tensor(area, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }
            if target["boxes"].numel() == 0:
                target["boxes"] = target["boxes"].view(0, 4)
            return image, target

    return _CocoDetectionDataset()


def _train_minimal_flow(config: _ResolvedTrainConfig) -> dict[str, Any]:
    import random

    import torch
    from torch.utils.data import DataLoader

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dataset_kwargs: dict[str, Any] = {
        "path": config.dataset_path,
    }
    if config.format:
        load_dataset_kwargs["format"] = config.format

    dataset_payload = load_dataset(**load_dataset_kwargs)

    categories = dataset_payload.get("categories", [])
    if not categories and config.num_classes is None:
        raise ValueError("Dataset does not define classes; pass num_classes explicitly.")

    num_classes = int(config.num_classes or len(categories) or 1) + 1
    if num_classes <= 1:
        raise ValueError("num_classes must be > 0.")

    requested_device = config.device.strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model_name = config.model_name
    model = _build_torchvision_model(model_name, num_classes=num_classes)
    model.to(device)

    dataset = _build_dataset(
        dataset_payload,
        device=device.type,
        train_split=config.train_split,
    )
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty after loading.")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    optimizer = _build_optimizer(model, config.learning_rate, config.optimizer, seed=config.seed)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        step_count = 0
        for images, targets in dataloader:
            images = list(images)
            target_list = []
            for target in targets:
                target_list.append(
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                        "image_id": target["image_id"].to(device),
                        "area": target["area"].to(device),
                        "iscrowd": target["iscrowd"].to(device),
                    }
                )

            losses = model(list(images), target_list)
            loss = sum(loss for loss in losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_loss += float(loss.item())
            step_count += 1

        epoch_loss = cumulative_loss / max(step_count, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})

    final_epoch = config.epochs
    checkpoint_path = output_dir / f"epoch_{final_epoch:03d}.pth"
    torch.save(
        {
            "epoch": final_epoch,
            "model_name": model_name,
            "num_classes": num_classes,
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )

    metrics_path = output_dir / "metrics.json"
    metrics = {
        "epochs": config.epochs,
        "history": history,
        "num_samples": len(dataset),
        "model_name": model_name,
        "output_dir": str(output_dir),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "output_dir": str(output_dir),
        "model_name": model_name,
    }


def train(
    *,
    config: Any | None = None,
    pipeline: Any | None = None,
    build: bool = True,
    **pipeline_kwargs: Any,
):
    """Run detector training.

    Parameters
    ----------
    config:
        Optional config mapping or object that defines one-liner training inputs.
    pipeline:
        Optional pre-built :class:`simpledet.api.ObjectDetectionPipeline`.
    build:
        Build the pipeline before training.
    pipeline_kwargs:
        Forwarded to native training when ``pipeline`` is not provided.
    """

    if config is not None:
        if pipeline is not None or pipeline_kwargs:
            raise TypeError(
                "Use either `config` for one-liner training or `pipeline`/"
                "`pipeline_kwargs` for legacy pipeline training."
            )
        resolved = _parse_config(config)
        _validate_model_name(resolved.model_name)
        require_dependency("torch", "train")
        return _train_minimal_flow(resolved)

    if pipeline is None:
        if not pipeline_kwargs:
            raise TypeError(
                "Provide either `config` or `pipeline`/`pipeline_kwargs`."
                " In this build, pipeline keyword arguments are routed to native training."
            )
        return _run_native_training_from_kwargs(pipeline_kwargs)

    if build:
        if not hasattr(pipeline, "build"):
            raise TypeError("`pipeline` does not expose build().")
        pipeline.build()
    if not hasattr(pipeline, "train"):
        raise TypeError("`pipeline` does not expose train().")

    pipeline.train()
    return pipeline
