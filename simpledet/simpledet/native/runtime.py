"""Runtime entry points for the native Lightning backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..metrics import evaluate_coco_bbox_metrics
from ..suite import DetectorSpec
from .data import NativeDataConfig, NativeDetectionDataModule
from .engine import NativeDetectionLightningModule, NativeEngineConfig, build_native_trainer


@dataclass(slots=True)
class NativeProjectConfig:
    dataset_root: str
    categories: tuple[str, ...]
    detector_spec: DetectorSpec
    output_dir: str
    checkpoint_path: str | None = None
    in_channels: int = 3
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 1e-3
    optimizer: str = "sgd"
    scheduler: str | None = None
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1
    max_epochs: int = 1
    accelerator: str = "cpu"
    devices: int = 1


def _default_checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir).expanduser() / "checkpoints" / "last.ckpt"


def _resolved_checkpoint_path(config: NativeProjectConfig) -> Path:
    checkpoint_path = config.checkpoint_path
    resolved = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path
        else _default_checkpoint_path(config.output_dir)
    )
    if resolved.exists():
        return resolved
    if checkpoint_path:
        raise FileNotFoundError(f"Missing native checkpoint: {resolved}")
    raise FileNotFoundError(
        "Missing native checkpoint. Train first so "
        f"'{resolved}' exists, or pass `checkpoint_path` explicitly."
    )


def _trainer_checkpoint_path(trainer: Any, output_dir: str) -> Path:
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is not None:
        for candidate in (
            getattr(checkpoint_callback, "last_model_path", None),
            getattr(checkpoint_callback, "best_model_path", None),
        ):
            if candidate:
                return Path(str(candidate)).expanduser()
    return _default_checkpoint_path(output_dir)


def _call_trainer_test(trainer: Any, module: Any, *, datamodule: Any, checkpoint_path: Path) -> Any:
    try:
        return trainer.test(module, datamodule=datamodule, ckpt_path=str(checkpoint_path))
    except TypeError as exc:
        if "ckpt_path" not in str(exc):
            raise
        return trainer.test(module, datamodule=datamodule)


def run_native_training(config: NativeProjectConfig) -> dict[str, Any]:
    data = NativeDetectionDataModule(
        NativeDataConfig(
            dataset_root=config.dataset_root,
            in_channels=config.in_channels,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    )
    data.setup("fit")
    module, payload = NativeDetectionLightningModule.build(
        NativeEngineConfig(
            architecture=config.detector_spec.architecture,
            num_classes=len(config.categories) + 1,
            in_channels=config.in_channels,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            scheduler_step_size=config.scheduler_step_size,
            scheduler_gamma=config.scheduler_gamma,
            max_epochs=config.max_epochs,
            accelerator=config.accelerator,
            devices=config.devices,
            output_dir=config.output_dir,
            detector_spec=config.detector_spec,
        )
    )
    trainer = build_native_trainer(payload.config)
    trainer.fit(module, datamodule=data)
    checkpoint_path = _trainer_checkpoint_path(trainer, config.output_dir)

    result = {
        "backend": "native_lightning",
        "stages": ["fit"],
        "architecture": config.detector_spec.architecture,
        "output_dir": str(Path(config.output_dir).expanduser()),
        "checkpoint_dir": str(Path(config.output_dir).expanduser() / "checkpoints"),
        "checkpoint_path": str(checkpoint_path),
    }
    _write_native_manifest(config.output_dir, result)
    return result


def run_native_evaluation(config: NativeProjectConfig) -> dict[str, Any]:
    data = NativeDetectionDataModule(
        NativeDataConfig(
            dataset_root=config.dataset_root,
            in_channels=config.in_channels,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    )
    data.setup("test")
    module, payload = NativeDetectionLightningModule.build(
        NativeEngineConfig(
            architecture=config.detector_spec.architecture,
            num_classes=len(config.categories) + 1,
            in_channels=config.in_channels,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            scheduler_step_size=config.scheduler_step_size,
            scheduler_gamma=config.scheduler_gamma,
            max_epochs=config.max_epochs,
            accelerator=config.accelerator,
            devices=config.devices,
            output_dir=config.output_dir,
            detector_spec=config.detector_spec,
        )
    )
    trainer = build_native_trainer(payload.config)
    checkpoint_path = _resolved_checkpoint_path(config)
    _call_trainer_test(trainer, module, datamodule=data, checkpoint_path=checkpoint_path)
    metrics = _evaluate_native_metrics(data, payload.latest_predictions)
    metrics_path = _write_native_metrics(config.output_dir, metrics)
    result = {
        "backend": "native_lightning",
        "stages": ["test"],
        "architecture": config.detector_spec.architecture,
        "predictions": payload.latest_predictions,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "output_dir": str(Path(config.output_dir).expanduser()),
        "checkpoint_path": str(checkpoint_path),
    }
    _write_native_manifest(config.output_dir, result)
    return result


def run_native_inference(config: NativeProjectConfig) -> dict[str, Any]:
    return run_native_evaluation(config)


def _write_native_manifest(output_dir: str, payload: dict[str, Any]) -> None:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    (root / "native-manifest.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _write_native_metrics(output_dir: str, metrics: dict[str, Any]) -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "native-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics_path


def _evaluate_native_metrics(
    data: NativeDetectionDataModule,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = data.test_dataset
    if dataset is None:
        raise RuntimeError("Native evaluation metrics require an initialized test dataset.")
    return evaluate_coco_bbox_metrics(dataset.annotation_path, predictions)
