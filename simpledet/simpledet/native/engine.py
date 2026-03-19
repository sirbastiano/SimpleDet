"""PyTorch Lightning engine for the native SimpleDet backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..detectors._deps import require_dependency
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

    @classmethod
    def build(cls, config: NativeEngineConfig):
        pl, _ = _load_lightning()

        class _WrappedModule(pl.LightningModule):
            def __init__(self, payload: "NativeDetectionLightningModule") -> None:
                super().__init__()
                self.payload = payload
                self.model = payload.model

            def forward(self, images):
                return self.model(images)

            def training_step(self, batch, batch_idx):
                images, targets = batch
                losses = self.model(images, targets)
                loss = _select_training_loss(losses)
                self.log("train_loss", loss, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                images, targets = batch
                outputs = self.model(images)
                count = 0.0
                for output in outputs:
                    if "boxes" in output:
                        count += float(len(output.get("boxes", [])))
                    elif "features" in output:
                        count += float(len(output.get("features", [])))
                metric = float(count)
                self.log("val_detection_count", metric, prog_bar=False)
                return metric

            def test_step(self, batch, batch_idx):
                images, targets = batch
                outputs = self.model(images)
                predictions = []
                for output, target in zip(outputs, targets):
                    if "boxes" in output:
                        predictions.append(
                            {
                                "image_id": int(target["image_id"][0]),
                                "boxes": output["boxes"].detach().cpu().tolist(),
                                "scores": output["scores"].detach().cpu().tolist(),
                                "labels": output["labels"].detach().cpu().tolist(),
                            }
                        )
                    else:
                        predictions.append(
                            {
                                "image_id": int(target["image_id"][0]),
                                "feature_levels": len(output.get("features", [])),
                                "head_keys": sorted(output.get("head_outputs", {}).keys()),
                            }
                        )
                self.payload.latest_predictions.extend(predictions)
                return {"predictions": predictions}

            def configure_optimizers(self):
                import torch

                normalized = self.payload.config.optimizer.strip().lower()
                params = self.model.parameters()
                if normalized == "sgd":
                    return torch.optim.SGD(params, lr=self.payload.config.learning_rate, momentum=0.9)
                if normalized == "adam":
                    return torch.optim.Adam(params, lr=self.payload.config.learning_rate)
                if normalized == "adamw":
                    return torch.optim.AdamW(params, lr=self.payload.config.learning_rate)
                raise ValueError(
                    f"Unsupported optimizer '{self.payload.config.optimizer}'."
                )

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
    if "loss_total" in losses:
        return losses["loss_total"]
    return sum(losses.values())
