"""Runtime entry points for the native Lightning backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..suite import DetectorSpec
from .data import NativeDataConfig, NativeDetectionDataModule
from .engine import NativeDetectionLightningModule, NativeEngineConfig, build_native_trainer


@dataclass(slots=True)
class NativeProjectConfig:
    dataset_root: str
    categories: tuple[str, ...]
    detector_spec: DetectorSpec
    output_dir: str
    in_channels: int = 3
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 1e-3
    optimizer: str = "sgd"
    max_epochs: int = 1
    accelerator: str = "cpu"
    devices: int = 1


def run_native_training(config: NativeProjectConfig) -> dict[str, Any]:
    data = NativeDetectionDataModule(
        NativeDataConfig(
            dataset_root=config.dataset_root,
            in_channels=config.in_channels,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    )
    module, payload = NativeDetectionLightningModule.build(
        NativeEngineConfig(
            architecture=config.detector_spec.architecture,
            num_classes=len(config.categories) + 1,
            in_channels=config.in_channels,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            max_epochs=config.max_epochs,
            accelerator=config.accelerator,
            devices=config.devices,
            output_dir=config.output_dir,
            detector_spec=config.detector_spec,
        )
    )
    trainer = build_native_trainer(payload.config)
    trainer.fit(module, datamodule=data)

    result = {
        "backend": "native_lightning",
        "stages": ["fit"],
        "architecture": config.detector_spec.architecture,
        "output_dir": str(Path(config.output_dir).expanduser()),
        "checkpoint_dir": str(Path(config.output_dir).expanduser() / "checkpoints"),
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
    module, payload = NativeDetectionLightningModule.build(
        NativeEngineConfig(
            architecture=config.detector_spec.architecture,
            num_classes=len(config.categories) + 1,
            in_channels=config.in_channels,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            max_epochs=config.max_epochs,
            accelerator=config.accelerator,
            devices=config.devices,
            output_dir=config.output_dir,
            detector_spec=config.detector_spec,
        )
    )
    trainer = build_native_trainer(payload.config)
    trainer.test(module, datamodule=data)
    result = {
        "backend": "native_lightning",
        "stages": ["test"],
        "architecture": config.detector_spec.architecture,
        "predictions": payload.latest_predictions,
        "output_dir": str(Path(config.output_dir).expanduser()),
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
