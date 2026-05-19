"""PyTorch Lightning engine for the native SimpleDet backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .modeling import build_native_model


def _load_lightning():
    try:
        import lightning.pytorch as pl  # type: ignore
        from lightning.pytorch.callbacks import ModelCheckpoint  # type: ignore
        return pl, ModelCheckpoint
    except ModuleNotFoundError:
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
        return pl, ModelCheckpoint


@dataclass(slots=True)
class NativeEngineConfig:
    architecture: str = "retinanet"
    num_classes: int = 2
    in_channels: int = 3
    detector_spec: Any | None = None
    learning_rate: float = 1e-3
    optimizer: str = "sgd"
    scheduler: str | None = None
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1
    max_epochs: int = 1
    accelerator: str = "cpu"
    devices: int = 1
    output_dir: str = "runs/native"


class NativeDetectionLightningModule:
    """LightningModule wrapper around a torchvision detection model."""

    def __init__(self, config: NativeEngineConfig) -> None:
        pl, _ = _load_lightning()
        self._pl = pl
        self.config = config
        self.model = build_native_model(
            config.architecture,
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            detector_spec=getattr(config, "detector_spec", None),
        )
        self.latest_predictions: list[dict[str, Any]] = []
        self.loaded_checkpoint_metadata: dict[str, Any] | None = None

    def forward_loss(self, images: Any, targets: Any) -> Any:
        if hasattr(self.model, "forward_loss"):
            return self.model.forward_loss(images, targets)
        return self.model(images, targets)

    def predict(self, images: Any) -> Any:
        if hasattr(self.model, "predict"):
            return self.model.predict(images)
        return self.model(images)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "backend": "native_lightning",
            "format_version": 1,
            "model_class": type(self.model).__name__,
            "engine_config": _engine_config_metadata(self.config),
        }

    @classmethod
    def build(cls, config: NativeEngineConfig):
        pl, _ = _load_lightning()

        class _WrappedModule(pl.LightningModule):
            def __init__(self, payload: "NativeDetectionLightningModule") -> None:
                super().__init__()
                self.payload = payload
                self.model = payload.model

            def forward(self, images, targets=None):
                if targets is not None:
                    return self.forward_loss(images, targets)
                return self.predict(images)

            def forward_loss(self, images, targets):
                return self.payload.forward_loss(images, targets)

            def predict(self, images):
                return self.payload.predict(images)

            def training_step(self, batch, batch_idx):
                images, targets = _unpack_detection_batch(batch)
                losses = self.forward_loss(images, targets)
                loss = _select_training_loss(losses)
                _log_loss_metrics(self, "train", losses, loss, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                images, targets = _unpack_detection_batch(batch)
                losses = self.forward_loss(images, targets)
                loss = _select_training_loss(losses)
                outputs = self.predict(images)
                metric = _prediction_count(outputs)
                _log_loss_metrics(self, "val", losses, loss, prog_bar=False)
                _log_metric(self, "val_detection_count", metric, prog_bar=False)
                return {"loss": loss, "detection_count": metric, "predictions": outputs}

            def test_step(self, batch, batch_idx):
                images, targets = _unpack_detection_batch(batch)
                outputs = self.predict(images)
                predictions = _serialize_predictions(outputs, targets)
                _log_metric(self, "test_detection_count", _prediction_count(outputs), prog_bar=False)
                self.payload.latest_predictions.extend(predictions)
                return {"predictions": predictions}

            def configure_optimizers(self):
                import torch

                optimizer = _build_optimizer(torch, self.model, self.payload.config)
                scheduler = _build_scheduler(torch, optimizer, self.payload.config)
                if scheduler is None:
                    return optimizer
                return {"optimizer": optimizer, "lr_scheduler": scheduler}

            def on_save_checkpoint(self, checkpoint):
                checkpoint["simpledet"] = self.payload.checkpoint_metadata()

            def on_load_checkpoint(self, checkpoint):
                metadata = checkpoint.get("simpledet", {})
                self.payload.loaded_checkpoint_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

        payload = cls(config)
        return _WrappedModule(payload), payload


def build_native_trainer(config: NativeEngineConfig):
    pl, ModelCheckpoint = _load_lightning()
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        save_top_k=1,
        monitor=None,
        save_last=True,
    )
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=int(config.max_epochs),
        accelerator=config.accelerator,
        devices=config.devices,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint],
        enable_model_summary=False,
    )
    return trainer


def _select_training_loss(losses):
    if not isinstance(losses, Mapping):
        return losses
    if not losses:
        raise RuntimeError("Native detector returned no losses for training.")
    if "loss_total" in losses:
        return losses["loss_total"]
    total = None
    for value in losses.values():
        total = value if total is None else total + value
    return total


def _build_optimizer(torch: Any, model: Any, config: NativeEngineConfig) -> Any:
    params = [param for param in model.parameters() if getattr(param, "requires_grad", True)]
    if not params:
        raise RuntimeError(
            "NativeDetectionLightningModule cannot configure an optimizer because "
            "the detector has no trainable parameters."
        )

    normalized = config.optimizer.strip().lower()
    if normalized == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9)
    if normalized == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate)
    if normalized == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'. Supported: sgd, adam, adamw.")


def _build_scheduler(torch: Any, optimizer: Any, config: NativeEngineConfig) -> Any | None:
    normalized = _optional_name(config.scheduler)
    if normalized is None:
        return None
    schedulers = torch.optim.lr_scheduler
    if normalized in {"step", "steplr"}:
        return schedulers.StepLR(
            optimizer,
            step_size=max(1, int(config.scheduler_step_size)),
            gamma=float(config.scheduler_gamma),
        )
    if normalized in {"exponential", "exponentiallr"}:
        return schedulers.ExponentialLR(optimizer, gamma=float(config.scheduler_gamma))
    if normalized in {"cosine", "cosineannealing", "cosineannealinglr"}:
        return schedulers.CosineAnnealingLR(optimizer, T_max=max(1, int(config.max_epochs)))
    raise ValueError(
        f"Unsupported scheduler '{config.scheduler}'. Supported: step, exponential, cosine."
    )


def _optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "off", "false"}:
        return None
    return normalized


def _unpack_detection_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        return batch["images"], batch["targets"]
    try:
        images, targets = batch
    except (TypeError, ValueError) as exc:
        raise TypeError("Native detection batches must contain images and targets.") from exc
    return images, targets


def _log_loss_metrics(module: Any, prefix: str, losses: Any, selected_loss: Any, *, prog_bar: bool) -> None:
    _log_metric(module, f"{prefix}_loss", selected_loss, prog_bar=prog_bar)
    if not isinstance(losses, Mapping):
        return
    for name, value in losses.items():
        if name == "loss_total":
            continue
        metric_name = name if name.startswith(f"{prefix}_") else f"{prefix}_{name}"
        _log_metric(module, metric_name, value, prog_bar=False)


def _log_metric(module: Any, name: str, value: Any, *, prog_bar: bool) -> None:
    try:
        module.log(name, value, prog_bar=prog_bar, on_step=False, on_epoch=True)
    except TypeError:
        module.log(name, value, prog_bar=prog_bar)


def _prediction_count(outputs: Any) -> float:
    count = 0
    for output in _as_sequence(outputs):
        if isinstance(output, Mapping) and "boxes" in output:
            count += len(output.get("boxes", []))
        elif isinstance(output, Mapping) and "features" in output:
            count += len(output.get("features", []))
    return float(count)


def _serialize_predictions(outputs: Any, targets: Any) -> list[dict[str, Any]]:
    serialized = []
    for output, target in zip(_as_sequence(outputs), _as_sequence(targets)):
        image_id = _target_image_id(target)
        if isinstance(output, Mapping) and "boxes" in output:
            serialized.append(
                {
                    "image_id": image_id,
                    "boxes": _tensor_to_list(output.get("boxes", [])),
                    "scores": _tensor_to_list(output.get("scores", [])),
                    "labels": _tensor_to_list(output.get("labels", [])),
                }
            )
        else:
            output_mapping = output if isinstance(output, Mapping) else {}
            serialized.append(
                {
                    "image_id": image_id,
                    "feature_levels": len(output_mapping.get("features", [])),
                    "head_keys": sorted(output_mapping.get("head_outputs", {}).keys()),
                }
            )
    return serialized


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return (value,)


def _target_image_id(target: Any) -> int:
    value = target.get("image_id", 0) if isinstance(target, Mapping) else 0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = value[0] if value else 0
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    elif hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value = value[0] if value else 0
    return int(value)


def _tensor_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_tensor_to_list(item) for item in value]
    if isinstance(value, list):
        return [_tensor_to_list(item) for item in value]
    return value


def _engine_config_metadata(config: NativeEngineConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name == "detector_spec":
            metadata[field.name] = _detector_spec_metadata(value)
        else:
            metadata[field.name] = _metadata_value(value)
    return metadata


def _detector_spec_metadata(detector_spec: Any) -> Any:
    if detector_spec is None:
        return None
    return {
        "architecture": _metadata_value(getattr(detector_spec, "architecture", None)),
        "name": _metadata_value(getattr(detector_spec, "name", None)),
        "family": _metadata_value(getattr(detector_spec, "family", None)),
    }


def _metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_metadata_value(item) for item in value]
    if isinstance(value, list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return repr(value)
