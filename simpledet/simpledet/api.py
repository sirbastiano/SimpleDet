"""Native-only public API for SimpleDet."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from ._legacy import legacy_py_config_error
from .suite.specs import DetectorSpec

DEFAULT_IMAGE_SUBDIR = "imgs"
DEFAULT_ANNOTATION_SUBDIR = "Annotations"
DEFAULT_SPLIT_FILENAMES = {
    "train": "train_annotations.json",
    "val": "val_annotations.json",
    "test": "test_annotations.json",
}
@dataclass(frozen=True)
class ProjectLayout:
    """Convention-based dataset layout for a native detector project."""

    dataset_root: str
    result_folder: Optional[str] = None
    image_subdir: str = DEFAULT_IMAGE_SUBDIR
    annotation_subdir: str = DEFAULT_ANNOTATION_SUBDIR
    train_filename: str = DEFAULT_SPLIT_FILENAMES["train"]
    val_filename: str = DEFAULT_SPLIT_FILENAMES["val"]
    test_filename: str = DEFAULT_SPLIT_FILENAMES["test"]

    @property
    def root_path(self) -> Path:
        return Path(self.dataset_root).expanduser()

    @property
    def images_path(self) -> Path:
        return self.root_path / self.image_subdir

    @property
    def annotations_path(self) -> Path:
        return self.root_path / self.annotation_subdir

    @property
    def train_annotations(self) -> Path:
        return self.annotations_path / self.train_filename

    @property
    def val_annotations(self) -> Path:
        return self.annotations_path / self.val_filename

    @property
    def test_annotations(self) -> Path:
        return self.annotations_path / self.test_filename

    @property
    def resolved_result_folder(self) -> Path:
        if self.result_folder is not None:
            return Path(self.result_folder).expanduser()
        return self.root_path / "runs" / "simpledet"

    def validation_report(self) -> dict[str, Any]:
        checks = {
            "dataset_root": self.root_path,
            "images": self.images_path,
            "annotations_dir": self.annotations_path,
            "train_annotations": self.train_annotations,
            "val_annotations": self.val_annotations,
            "test_annotations": self.test_annotations,
        }
        return {
            "paths": {name: str(path) for name, path in checks.items()},
            "exists": {name: path.exists() for name, path in checks.items()},
        }


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and label paths for native detector execution."""

    data_root: str
    annot_file_train: str
    annot_file_val: str
    annot_file_test: str
    data_prefix: str = f"{DEFAULT_IMAGE_SUBDIR}/"
    categories: Sequence[str] = ("wake",)
    in_channels: int = 3
    tif_channels_to_load: Optional[Sequence[int]] = None

    def normalized(self) -> "DatasetConfig":
        channels = list(self.tif_channels_to_load or _default_band_selection(self.in_channels))
        return DatasetConfig(
            data_root=self.data_root,
            annot_file_train=self.annot_file_train,
            annot_file_val=self.annot_file_val,
            annot_file_test=self.annot_file_test,
            data_prefix=self.data_prefix,
            categories=tuple(self.categories),
            in_channels=self.in_channels,
            tif_channels_to_load=channels,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime and output settings for a detector project."""

    result_folder: str
    seed: int = 71
    resize: int = 768
    batch_size: int = 2
    max_epochs: int = 1
    amp: bool = True
    val_interval: int = 1


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimization knobs for the native training loop."""

    learning_rate: float = 0.001
    optimizer_choice: str = "AdamW"
    scheduler_choice: Optional[str] = None


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level operational config for a SimpleDet project."""

    dataset: DatasetConfig
    runtime: RuntimeConfig
    optimization: OptimizationConfig
    detector_spec: Optional[DetectorSpec | dict[str, Any]] = None
    model_cfg: Optional[dict[str, Any]] = None

    @staticmethod
    def from_mapping(payload: dict[str, Any]) -> "ProjectConfig":
        dataset = DatasetConfig(**dict(payload.get("dataset") or {})).normalized()
        runtime = RuntimeConfig(**dict(payload.get("runtime") or {}))
        optimization = OptimizationConfig(**dict(payload.get("optimization") or {}))
        return ProjectConfig(
            dataset=dataset,
            runtime=runtime,
            optimization=optimization,
            detector_spec=payload.get("detector_spec"),
            model_cfg=payload.get("model_cfg"),
        )

    @staticmethod
    def from_file(path: str | os.PathLike[str]) -> "ProjectConfig":
        config_path = Path(path).expanduser()
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".py":
            raise legacy_py_config_error(config_path)
        else:
            raise ValueError(
                f"Unsupported project config format '{config_path.suffix}'. Use .json or .toml."
            )
        if not isinstance(payload, dict):
            raise TypeError("Project config payload must be a mapping.")
        return ProjectConfig.from_mapping(payload)


def _default_band_selection(in_channels: int) -> list[int]:
    if int(in_channels) < 1:
        raise ValueError("`in_channels` must be at least 1.")
    return list(range(1, int(in_channels) + 1))


def load_project_config(path: str | os.PathLike[str]) -> ProjectConfig:
    return ProjectConfig.from_file(path)


def project_config_template(format: str = "toml") -> str:
    payload = {
        "dataset": {
            "data_root": "/path/to/dataset",
            "annot_file_train": "/path/to/dataset/Annotations/train_annotations.json",
            "annot_file_val": "/path/to/dataset/Annotations/val_annotations.json",
            "annot_file_test": "/path/to/dataset/Annotations/test_annotations.json",
            "data_prefix": "imgs/",
            "categories": ["wake"],
            "in_channels": 3,
            "tif_channels_to_load": [1, 2, 3],
        },
        "runtime": {
            "result_folder": "/tmp/simpledet-runs",
            "resize": 768,
            "batch_size": 2,
            "max_epochs": 12,
            "seed": 71,
            "amp": True,
            "val_interval": 1,
        },
        "optimization": {
            "learning_rate": 0.001,
            "optimizer_choice": "AdamW",
            "scheduler_choice": None,
        },
        "detector_spec": {
            "architecture": "retinanet",
            "num_classes": 1,
            "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
        },
    }
    normalized = format.strip().lower()
    if normalized == "json":
        return json.dumps(payload, indent=2)
    if normalized != "toml":
        raise ValueError("Unsupported template format. Use 'toml' or 'json'.")
    return """[dataset]
data_root = "/path/to/dataset"
annot_file_train = "/path/to/dataset/Annotations/train_annotations.json"
annot_file_val = "/path/to/dataset/Annotations/val_annotations.json"
annot_file_test = "/path/to/dataset/Annotations/test_annotations.json"
data_prefix = "imgs/"
categories = ["wake"]
in_channels = 3
tif_channels_to_load = [1, 2, 3]

[runtime]
result_folder = "/tmp/simpledet-runs"
resize = 768
batch_size = 2
max_epochs = 12
seed = 71
amp = true
val_interval = 1

[optimization]
learning_rate = 0.001
optimizer_choice = "AdamW"

[detector_spec]
architecture = "retinanet"
num_classes = 1

[detector_spec.encoder]
name = "resnet18.a1_in1k"
source = "timm"
"""


def init_project_config(
    path: str | os.PathLike[str],
    *,
    format: str | None = None,
    overwrite: bool = False,
) -> str:
    config_path = Path(path).expanduser()
    if config_path.suffix.lower() == ".py":
        raise legacy_py_config_error(config_path)
    resolved_format = (format or config_path.suffix.lstrip(".") or "toml").lower()
    if resolved_format not in {"toml", "json"}:
        raise ValueError("Unsupported project config format. Use 'toml' or 'json'.")
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {config_path}.")
    if not config_path.suffix:
        config_path = config_path.with_suffix(f".{resolved_format}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(project_config_template(resolved_format), encoding="utf-8")
    return str(config_path)


def validate_project_config(
    config: ProjectConfig | dict[str, Any] | str | os.PathLike[str],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    project = _coerce_project_config(config)
    image_root = (
        Path(project.dataset.data_root).expanduser() / project.dataset.data_prefix
        if project.dataset.data_prefix
        else Path(project.dataset.data_root).expanduser() / DEFAULT_IMAGE_SUBDIR
    )
    checks = {
        "dataset_root": Path(project.dataset.data_root).expanduser(),
        "images": image_root,
        "annotations_dir": Path(project.dataset.annot_file_train).expanduser().parent,
        "train_annotations": Path(project.dataset.annot_file_train).expanduser(),
        "val_annotations": Path(project.dataset.annot_file_val).expanduser(),
        "test_annotations": Path(project.dataset.annot_file_test).expanduser(),
    }
    report = {
        "paths": {name: str(path) for name, path in checks.items()},
        "exists": {name: path.exists() for name, path in checks.items()},
    }
    report["missing"] = [name for name, exists in report["exists"].items() if not exists]
    if strict and report["missing"]:
        raise FileNotFoundError(
            "Project validation failed. Missing required input paths: " + ", ".join(report["missing"])
        )
    return report


def _coerce_project_config(config: ProjectConfig | dict[str, Any] | str | os.PathLike[str]) -> ProjectConfig:
    if isinstance(config, ProjectConfig):
        return config
    if isinstance(config, (str, os.PathLike)):
        return load_project_config(config)
    if isinstance(config, dict):
        return ProjectConfig.from_mapping(config)
    raise TypeError("`config` must be a ProjectConfig, mapping, or path.")


def _coerce_detector_spec(detector_spec: Optional[DetectorSpec | dict[str, Any]], *, model_cfg: Optional[dict[str, Any]] = None) -> DetectorSpec:
    if model_cfg is not None:
        raise TypeError("`model_cfg` is no longer supported. Use `detector_spec`.")
    if detector_spec is None:
        raise TypeError("`detector_spec` is required for native execution.")
    if isinstance(detector_spec, DetectorSpec):
        return detector_spec
    if not isinstance(detector_spec, dict):
        raise TypeError("`detector_spec` must be a DetectorSpec or mapping.")
    return DetectorSpec(**detector_spec)


def _build_native_project_config(
    *,
    dataset_root: str,
    categories: Sequence[str],
    detector_spec: DetectorSpec,
    in_channels: int,
    result_folder: Optional[str],
    kwargs: dict[str, Any],
):
    from .native.runtime import NativeProjectConfig

    return NativeProjectConfig(
        dataset_root=dataset_root,
        categories=tuple(categories),
        detector_spec=detector_spec,
        output_dir=result_folder or str(Path(dataset_root).expanduser() / "runs" / "simpledet"),
        checkpoint_path=kwargs.get("checkpoint_path") or kwargs.get("ckpt_path"),
        in_channels=int(in_channels),
        batch_size=int(kwargs.get("batch_size", 2)),
        num_workers=int(kwargs.get("num_workers", 0)),
        learning_rate=float(kwargs.get("learning_rate", 1e-3)),
        optimizer=str(kwargs.get("optimizer_choice", "adamw")).lower(),
        max_epochs=int(kwargs.get("max_epochs", 1)),
        accelerator=str(kwargs.get("accelerator", "cpu")),
        devices=int(kwargs.get("devices", 1)),
    )


def run_native_training(
    *,
    data_root: Optional[str] = None,
    dataset_root: Optional[str] = None,
    categories: Sequence[str],
    in_channels: int,
    detector_spec: DetectorSpec | dict[str, Any],
    model_cfg: Optional[dict[str, Any]] = None,
    result_folder: Optional[str] = None,
    validate: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_training(
        dataset_root=str(dataset_root or data_root),
        categories=categories,
        in_channels=in_channels,
        detector_spec=detector_spec,
        model_cfg=model_cfg,
        result_folder=result_folder,
        validate=validate,
        **kwargs,
    )


def run_native_inference(
    *,
    data_root: Optional[str] = None,
    dataset_root: Optional[str] = None,
    categories: Sequence[str],
    in_channels: int,
    detector_spec: DetectorSpec | dict[str, Any],
    model_cfg: Optional[dict[str, Any]] = None,
    result_folder: Optional[str] = None,
    validate: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_inference(
        dataset_root=str(dataset_root or data_root),
        categories=categories,
        in_channels=in_channels,
        detector_spec=detector_spec,
        model_cfg=model_cfg,
        result_folder=result_folder,
        validate=validate,
        **kwargs,
    )


def run_native_evaluation(**kwargs: Any) -> dict[str, Any]:
    return run_native_inference(**kwargs)


def run_project(
    config: ProjectConfig | dict[str, Any] | str | os.PathLike[str],
    *,
    stages: Sequence[str] = ("build", "train", "test"),
    validate: bool = True,
) -> dict[str, Any]:
    project = _coerce_project_config(config)
    detector_spec = _coerce_detector_spec(project.detector_spec, model_cfg=project.model_cfg)
    if validate:
        validate_project_config(project, strict=True)
    native_config = _build_native_project_config(
        dataset_root=project.dataset.data_root,
        categories=project.dataset.categories,
        detector_spec=detector_spec,
        in_channels=project.dataset.in_channels,
        result_folder=project.runtime.result_folder,
        kwargs={
            "batch_size": project.runtime.batch_size,
            "learning_rate": project.optimization.learning_rate,
            "optimizer_choice": project.optimization.optimizer_choice,
            "max_epochs": project.runtime.max_epochs,
        },
    )
    from .native.runtime import run_native_inference as _run_native_inference
    from .native.runtime import run_native_training as _run_native_training

    normalized = []
    aliases = {"fit": "train", "infer": "test", "inference": "test", "eval": "test", "evaluate": "test"}
    for stage in stages:
        resolved = aliases.get(str(stage).strip().lower(), str(stage).strip().lower())
        if resolved not in {"build", "train", "test"}:
            raise ValueError(f"Unsupported project stage: {stage}")
        normalized.append(resolved)
    run_order = list(dict.fromkeys(normalized or ["build", "train", "test"]))
    result: dict[str, Any] = {
        "backend": "native_lightning",
        "architecture": detector_spec.architecture,
        "output_dir": native_config.output_dir,
        "stages": run_order,
    }
    for stage in run_order:
        if stage == "build":
            continue
        if stage == "train":
            train_result = _run_native_training(native_config)
            result["train"] = train_result
            checkpoint_path = train_result.get("checkpoint_path")
            if checkpoint_path:
                native_config.checkpoint_path = str(checkpoint_path)
        else:
            result["test"] = _run_native_inference(native_config)
    return result


def run_training(
    *,
    dataset_root: str,
    categories: Sequence[str],
    in_channels: int,
    model_cfg: Optional[dict[str, Any]] = None,
    detector_spec: Optional[DetectorSpec | dict[str, Any]] = None,
    tif_channels_to_load: Optional[Sequence[int]] = None,
    result_folder: Optional[str] = None,
    validate: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    del tif_channels_to_load
    resolved_detector_spec = _coerce_detector_spec(detector_spec, model_cfg=model_cfg)
    if validate:
        layout = ProjectLayout(dataset_root=dataset_root, result_folder=result_folder)
        validate_project_config(
            {
                "dataset": {
                    "data_root": str(layout.root_path),
                    "annot_file_train": str(layout.train_annotations),
                    "annot_file_val": str(layout.val_annotations),
                    "annot_file_test": str(layout.test_annotations),
                    "data_prefix": f"{layout.image_subdir}/",
                    "categories": categories,
                    "in_channels": in_channels,
                },
                "runtime": {"result_folder": str(layout.resolved_result_folder)},
                "optimization": {},
                "detector_spec": resolved_detector_spec,
            },
            strict=True,
        )
    from .native.runtime import run_native_training as _run_native_training

    return _run_native_training(
        _build_native_project_config(
            dataset_root=dataset_root,
            categories=categories,
            detector_spec=resolved_detector_spec,
            in_channels=in_channels,
            result_folder=result_folder,
            kwargs=kwargs,
        )
    )


def run_inference(
    *,
    dataset_root: str,
    categories: Sequence[str],
    in_channels: int,
    model_cfg: Optional[dict[str, Any]] = None,
    detector_spec: Optional[DetectorSpec | dict[str, Any]] = None,
    tif_channels_to_load: Optional[Sequence[int]] = None,
    result_folder: Optional[str] = None,
    validate: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    del tif_channels_to_load
    resolved_detector_spec = _coerce_detector_spec(detector_spec, model_cfg=model_cfg)
    if validate:
        layout = ProjectLayout(dataset_root=dataset_root, result_folder=result_folder)
        validate_project_config(
            {
                "dataset": {
                    "data_root": str(layout.root_path),
                    "annot_file_train": str(layout.test_annotations),
                    "annot_file_val": str(layout.test_annotations),
                    "annot_file_test": str(layout.test_annotations),
                    "data_prefix": f"{layout.image_subdir}/",
                    "categories": categories,
                    "in_channels": in_channels,
                },
                "runtime": {"result_folder": str(layout.resolved_result_folder)},
                "optimization": {},
                "detector_spec": resolved_detector_spec,
            },
            strict=True,
        )
    from .native.runtime import run_native_inference as _run_native_inference

    return _run_native_inference(
        _build_native_project_config(
            dataset_root=dataset_root,
            categories=categories,
            detector_spec=resolved_detector_spec,
            in_channels=in_channels,
            result_folder=result_folder,
            kwargs=kwargs,
        )
    )


def run_evaluation(**kwargs: Any) -> dict[str, Any]:
    return run_inference(**kwargs)


def list_available_encoders(pattern: str | None = None) -> list[str]:
    try:
        import timm
    except ModuleNotFoundError:
        return []
    return sorted(timm.list_models(pattern))


def list_available_necks() -> list[str]:
    from .suite.catalog import list_native_neck_families

    return list_native_neck_families()


def list_available_heads() -> list[str]:
    from .suite.catalog import list_native_head_families

    return list_native_head_families()


def print_available_encoders(pattern: str | None = None) -> None:
    for name in list_available_encoders(pattern):
        print(name)


def print_available_necks() -> None:
    for name in list_available_necks():
        print(name)


def print_available_heads() -> None:
    for name in list_available_heads():
        print(name)
