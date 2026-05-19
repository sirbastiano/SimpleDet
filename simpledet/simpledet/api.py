"""Native-only public API for SimpleDet."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from ._legacy import legacy_py_config_error
from .errors import ConfigPathError, ConfigValidationError
from .suite.specs import DecoderSpec, DetectorSpec, EncoderSpec, HeadSpec, NeckSpec

DEFAULT_IMAGE_SUBDIR = "imgs"
DEFAULT_ANNOTATION_SUBDIR = "Annotations"
DEFAULT_SPLIT_FILENAMES = {
    "train": "train_annotations.json",
    "val": "val_annotations.json",
    "test": "test_annotations.json",
}
DEFAULT_PROJECT_STAGES = ("build", "train", "test")
PROJECT_MANIFEST_FILENAME = "run-manifest.json"


def _clean_mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("Project config sections must be mappings.")
    return dict(payload)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_sequence(value: Any, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_image_dir(value: str | None) -> str:
    text = (value or DEFAULT_IMAGE_SUBDIR).strip()
    if Path(text).expanduser().is_absolute():
        return str(Path(text).expanduser())
    return text.strip("/") or DEFAULT_IMAGE_SUBDIR


def _resolve_dataset_file(
    root: str,
    value: str | None,
    *,
    default_filename: str,
    annotation_subdir: str = DEFAULT_ANNOTATION_SUBDIR,
) -> str:
    if value:
        path = Path(value).expanduser()
        if path.is_absolute() or not root:
            return str(path)
        return str(Path(root).expanduser() / path)
    if not root:
        return str(Path(annotation_subdir) / default_filename)
    return str(Path(root).expanduser() / annotation_subdir / default_filename)


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

    data_root: str = ""
    annot_file_train: str = ""
    annot_file_val: str = ""
    annot_file_test: str = ""
    data_prefix: str = f"{DEFAULT_IMAGE_SUBDIR}/"
    categories: Optional[Sequence[str]] = None
    in_channels: int = 3
    tif_channels_to_load: Optional[Sequence[int]] = None
    format: str = "coco"
    root: Optional[str] = None
    train: Optional[str] = None
    val: Optional[str] = None
    test: Optional[str] = None
    classes: Optional[Sequence[str]] = None
    image_path: Optional[str] = None
    images_dir: Optional[str] = None
    annotation_path: Optional[str] = None
    splits: Optional[dict[str, Any]] = None
    transforms: Optional[dict[str, Any]] = None

    @staticmethod
    def from_mapping(payload: dict[str, Any] | None) -> "DatasetConfig":
        aliases = {
            "imagePath": "image_path",
            "annotationPath": "annotation_path",
            "imageDir": "images_dir",
            "imagesDir": "images_dir",
            "dataRoot": "data_root",
            "inChannels": "in_channels",
            "tifChannelsToLoad": "tif_channels_to_load",
        }
        data = {}
        field_names = set(DatasetConfig.__dataclass_fields__)
        for key, value in _clean_mapping(payload).items():
            normalized_key = aliases.get(str(key), str(key))
            if normalized_key in field_names:
                data[normalized_key] = value
        return DatasetConfig(**data).normalized()

    def normalized(self) -> "DatasetConfig":
        root = _string_or_none(self.data_root) or _string_or_none(self.root) or ""
        splits = dict(self.splits or {})
        train = self.annot_file_train or self.train or splits.get("train")
        val = self.annot_file_val or self.val or splits.get("val")
        test = self.annot_file_test or self.test or splits.get("test")
        annotation_subdir = self.annotation_path or DEFAULT_ANNOTATION_SUBDIR
        images_dir = _normalize_image_dir(self.images_dir or self.image_path or self.data_prefix)
        categories = tuple(self.categories or self.classes or ("wake",))
        channels = list(self.tif_channels_to_load or _default_band_selection(self.in_channels))
        return DatasetConfig(
            data_root=root,
            annot_file_train=_resolve_dataset_file(
                root,
                _string_or_none(train),
                default_filename=DEFAULT_SPLIT_FILENAMES["train"],
                annotation_subdir=annotation_subdir,
            ),
            annot_file_val=_resolve_dataset_file(
                root,
                _string_or_none(val),
                default_filename=DEFAULT_SPLIT_FILENAMES["val"],
                annotation_subdir=annotation_subdir,
            ),
            annot_file_test=_resolve_dataset_file(
                root,
                _string_or_none(test),
                default_filename=DEFAULT_SPLIT_FILENAMES["test"],
                annotation_subdir=annotation_subdir,
            ),
            data_prefix=f"{images_dir}/",
            categories=categories,
            in_channels=int(self.in_channels),
            tif_channels_to_load=channels,
            format=str(self.format or "coco").lower(),
            root=root,
            train=train,
            val=val,
            test=test,
            classes=tuple(self.classes or categories),
            image_path=images_dir,
            images_dir=images_dir,
            annotation_path=annotation_subdir,
            splits=splits or None,
            transforms=dict(self.transforms or {}) or None,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime and output settings for a detector project."""

    result_folder: Optional[str] = None
    seed: int = 71
    resize: int = 768
    batch_size: int = 2
    max_epochs: int = 1
    amp: bool = True
    val_interval: int = 1
    num_workers: int = 0
    accelerator: str = "cpu"
    devices: int = 1

    @staticmethod
    def from_mapping(
        payload: dict[str, Any] | None,
        *,
        result_folder: str | None = None,
        seed: int | None = None,
    ) -> "RuntimeConfig":
        data = _clean_mapping(payload)
        return RuntimeConfig(
            result_folder=result_folder or _string_or_none(data.get("result_folder")),
            seed=int(seed if seed is not None else data.get("seed", 71)),
            resize=int(data.get("resize", 768)),
            batch_size=int(data.get("batch_size", 2)),
            max_epochs=int(data.get("max_epochs", 1)),
            amp=bool(data.get("amp", True)),
            val_interval=int(data.get("val_interval", 1)),
            num_workers=int(data.get("num_workers", 0)),
            accelerator=str(data.get("accelerator", "cpu")),
            devices=int(data.get("devices", 1)),
        )


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimization knobs for the native training loop."""

    learning_rate: float = 0.001
    optimizer_choice: str = "AdamW"
    scheduler_choice: Optional[str] = None
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1

    @staticmethod
    def from_mapping(
        optimizer_payload: dict[str, Any] | None,
        scheduler_payload: dict[str, Any] | None = None,
    ) -> "OptimizationConfig":
        optimizer = _clean_mapping(optimizer_payload)
        scheduler = _clean_mapping(scheduler_payload)
        scheduler_choice = (
            scheduler.get("scheduler_choice")
            or scheduler.get("name")
            or scheduler.get("type")
            or optimizer.get("scheduler_choice")
        )
        return OptimizationConfig(
            learning_rate=float(optimizer.get("learning_rate", optimizer.get("lr", 0.001))),
            optimizer_choice=str(
                optimizer.get("optimizer_choice", optimizer.get("name", optimizer.get("type", "AdamW")))
            ),
            scheduler_choice=_string_or_none(scheduler_choice),
            scheduler_step_size=int(
                scheduler.get("scheduler_step_size", scheduler.get("step_size", optimizer.get("scheduler_step_size", 1)))
            ),
            scheduler_gamma=float(
                scheduler.get("scheduler_gamma", scheduler.get("gamma", optimizer.get("scheduler_gamma", 0.1)))
            ),
        )


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint settings for project execution."""

    path: Optional[str] = None
    resume: bool = False
    strict: bool = True

    @staticmethod
    def from_mapping(payload: dict[str, Any] | str | None) -> "CheckpointConfig":
        if isinstance(payload, str):
            return CheckpointConfig(path=payload)
        data = _clean_mapping(payload)
        return CheckpointConfig(
            path=_string_or_none(data.get("path") or data.get("checkpoint_path") or data.get("ckpt_path")),
            resume=bool(data.get("resume", False)),
            strict=bool(data.get("strict", True)),
        )


@dataclass(frozen=True)
class ExportConfig:
    """Export settings recorded in the project manifest."""

    enabled: bool = True
    formats: Sequence[str] = ()
    output_dir: Optional[str] = None
    predictions_path: Optional[str] = None

    @staticmethod
    def from_mapping(payload: dict[str, Any] | None) -> "ExportConfig":
        data = _clean_mapping(payload)
        formats = data.get("formats", data.get("format", ()))
        if isinstance(formats, str):
            formats = (formats,)
        return ExportConfig(
            enabled=bool(data.get("enabled", True)),
            formats=tuple(str(item) for item in formats),
            output_dir=_string_or_none(data.get("output_dir")),
            predictions_path=_string_or_none(data.get("predictions_path")),
        )


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level operational config for a SimpleDet project."""

    dataset: DatasetConfig
    runtime: RuntimeConfig
    optimization: OptimizationConfig
    detector_spec: Optional[DetectorSpec | dict[str, Any]] = None
    detector: Optional[dict[str, Any]] = None
    model_cfg: Optional[dict[str, Any]] = None
    workdir: Optional[str] = None
    seed: int = 71
    stages: Sequence[str] = DEFAULT_PROJECT_STAGES
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def optimizer(self) -> OptimizationConfig:
        """Alias for the new project config field name."""
        return self.optimization

    @staticmethod
    def from_mapping(payload: dict[str, Any]) -> "ProjectConfig":
        data = _clean_mapping(payload)
        runtime_payload = _clean_mapping(data.get("runtime"))
        workdir = (
            _string_or_none(data.get("workdir"))
            or _string_or_none(runtime_payload.get("workdir"))
            or _string_or_none(runtime_payload.get("result_folder"))
        )
        seed_value = data.get("seed", runtime_payload.get("seed", 71))
        dataset = DatasetConfig.from_mapping(data.get("dataset"))
        runtime = RuntimeConfig.from_mapping(runtime_payload, result_folder=workdir, seed=int(seed_value))
        optimization = OptimizationConfig.from_mapping(
            data.get("optimization", data.get("optimizer")),
            data.get("scheduler"),
        )
        return ProjectConfig(
            dataset=dataset,
            runtime=runtime,
            optimization=optimization,
            detector=data.get("detector"),
            detector_spec=payload.get("detector_spec"),
            model_cfg=payload.get("model_cfg"),
            workdir=workdir or runtime.result_folder,
            seed=runtime.seed,
            stages=_coerce_sequence(data.get("stages"), default=DEFAULT_PROJECT_STAGES),
            checkpoint=CheckpointConfig.from_mapping(data.get("checkpoint")),
            export=ExportConfig.from_mapping(data.get("export", data.get("export_settings"))),
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
        raise ConfigValidationError("`in_channels` must be at least 1.")
    return list(range(1, int(in_channels) + 1))


def load_project_config(path: str | os.PathLike[str]) -> ProjectConfig:
    return ProjectConfig.from_file(path)


def project_config_template(format: str = "toml") -> str:
    payload = {
        "stages": ["build", "train", "test"],
        "workdir": "/tmp/simpledet-runs",
        "seed": 71,
        "detector": {
            "name": "retinanet",
            "num_classes": 1,
            "backbone": "resnet18",
            "pretrained": False,
        },
        "dataset": {
            "format": "coco",
            "root": "/path/to/dataset",
            "train": "Annotations/train_annotations.json",
            "val": "Annotations/val_annotations.json",
            "test": "Annotations/test_annotations.json",
            "data_prefix": "imgs/",
            "classes": ["wake"],
            "in_channels": 3,
            "tif_channels_to_load": [1, 2, 3],
        },
        "runtime": {
            "resize": 768,
            "batch_size": 2,
            "max_epochs": 12,
            "amp": True,
            "val_interval": 1,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
        },
        "scheduler": {
            "name": None,
        },
        "checkpoint": {"path": None, "resume": False},
        "export": {"formats": ["json"]},
    }
    normalized = format.strip().lower()
    if normalized == "json":
        return json.dumps(payload, indent=2)
    if normalized != "toml":
        raise ValueError("Unsupported template format. Use 'toml' or 'json'.")
    return """stages = ["build", "train", "test"]
workdir = "/tmp/simpledet-runs"
seed = 71

[detector]
name = "retinanet"
num_classes = 1
backbone = "resnet18"
pretrained = false

[dataset]
format = "coco"
root = "/path/to/dataset"
train = "Annotations/train_annotations.json"
val = "Annotations/val_annotations.json"
test = "Annotations/test_annotations.json"
data_prefix = "imgs/"
classes = ["wake"]
in_channels = 3
tif_channels_to_load = [1, 2, 3]

[runtime]
resize = 768
batch_size = 2
max_epochs = 12
amp = true
val_interval = 1

[optimizer]
name = "AdamW"
learning_rate = 0.001

[scheduler]

[checkpoint]
resume = false

[export]
formats = ["json"]
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
    dataset_root_text = _string_or_none(project.dataset.data_root)
    dataset_root = Path(dataset_root_text).expanduser() if dataset_root_text else None
    image_root = (
        dataset_root / _normalize_image_dir(project.dataset.data_prefix)
        if dataset_root is not None and project.dataset.data_prefix
        else dataset_root / DEFAULT_IMAGE_SUBDIR
        if dataset_root is not None
        else Path(_normalize_image_dir(project.dataset.data_prefix))
    )
    checks: dict[str, Path | None] = {
        "dataset_root": dataset_root,
        "images": image_root,
        "annotations_dir": Path(project.dataset.annot_file_train).expanduser().parent
        if project.dataset.annot_file_train
        else None,
        "train_annotations": Path(project.dataset.annot_file_train).expanduser()
        if project.dataset.annot_file_train
        else None,
        "val_annotations": Path(project.dataset.annot_file_val).expanduser()
        if project.dataset.annot_file_val
        else None,
        "test_annotations": Path(project.dataset.annot_file_test).expanduser()
        if project.dataset.annot_file_test
        else None,
    }
    report = {
        "paths": {name: str(path) if path is not None else "" for name, path in checks.items()},
        "exists": {name: bool(path is not None and path.exists()) for name, path in checks.items()},
    }
    report["missing"] = [name for name, exists in report["exists"].items() if not exists]
    if strict and report["missing"]:
        raise ConfigPathError(
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


def _coerce_component_spec(value: Any, expected_type: type) -> Any:
    if value is None or isinstance(value, expected_type):
        return value
    if isinstance(value, dict):
        return expected_type(**dict(value))
    return value


def _coerce_full_detector_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    mapping = dict(payload)
    mapping["encoder"] = _coerce_component_spec(mapping.get("encoder"), EncoderSpec)
    mapping["neck"] = _coerce_component_spec(mapping.get("neck"), NeckSpec)
    mapping["head"] = _coerce_component_spec(mapping.get("head"), HeadSpec)
    mapping["decoder"] = _coerce_component_spec(mapping.get("decoder"), DecoderSpec)
    return mapping


def _coerce_detector_builder_mapping(payload: dict[str, Any], *, in_channels: int) -> DetectorSpec:
    from .suite import build_detector

    mapping = dict(payload)
    architecture = mapping.pop("architecture", None) or mapping.pop("name", None)
    if not architecture:
        raise TypeError("`detector.name` or `detector.architecture` is required.")
    num_classes = mapping.pop("num_classes", None)
    if num_classes is None and "classes" in mapping:
        num_classes = len(tuple(mapping.pop("classes") or ()))
    if num_classes is None:
        num_classes = 1
    detector_in_channels = int(mapping.pop("in_channels", in_channels))
    encoder = _coerce_component_spec(mapping.pop("encoder", None), EncoderSpec)
    backbone = _coerce_component_spec(mapping.pop("backbone", None), EncoderSpec)
    neck = _coerce_component_spec(mapping.pop("neck", None), NeckSpec)
    head = _coerce_component_spec(mapping.pop("head", None), HeadSpec)
    decoder = _coerce_component_spec(mapping.pop("decoder", None), DecoderSpec)
    return build_detector(
        architecture,
        num_classes=int(num_classes),
        encoder=encoder,
        backbone=backbone,
        neck=neck,
        head=head,
        decoder=decoder,
        in_channels=detector_in_channels,
        pretrained=bool(mapping.pop("pretrained", True)),
        strict_auto_adapt=bool(mapping.pop("strict_auto_adapt", True)),
        imports=tuple(mapping.pop("imports", ())),
        **mapping,
    )


def _coerce_detector_spec(
    detector_spec: Optional[DetectorSpec | dict[str, Any]],
    *,
    model_cfg: Optional[dict[str, Any]] = None,
    detector: Optional[dict[str, Any]] = None,
    in_channels: int = 3,
) -> DetectorSpec:
    if model_cfg is not None:
        raise TypeError("`model_cfg` is no longer supported. Use `detector_spec`.")
    resolved = detector_spec if detector_spec is not None else detector
    if resolved is None:
        raise TypeError("`detector` or `detector_spec` is required for native execution.")
    if isinstance(resolved, DetectorSpec):
        return resolved
    if not isinstance(resolved, dict):
        raise TypeError("`detector` must be a DetectorSpec or mapping.")
    mapping = _coerce_full_detector_mapping(resolved)
    if "architecture" in mapping and "family" in mapping:
        return DetectorSpec(**mapping)
    return _coerce_detector_builder_mapping(mapping, in_channels=in_channels)


def _project_workdir(project: ProjectConfig) -> str | None:
    return project.workdir or project.runtime.result_folder


def _normalize_project_stages(stages: Sequence[str] | None) -> list[str]:
    aliases = {
        "fit": "train",
        "inference": "infer",
        "predict": "infer",
        "eval": "test",
        "evaluate": "test",
    }
    normalized: list[str] = []
    for stage in _coerce_sequence(stages, default=DEFAULT_PROJECT_STAGES):
        resolved = aliases.get(str(stage).strip().lower(), str(stage).strip().lower())
        if resolved not in {"build", "train", "test", "infer"}:
            raise ValueError(f"Unsupported project stage: {stage}")
        normalized.append(resolved)
    return list(dict.fromkeys(normalized or list(DEFAULT_PROJECT_STAGES)))


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclass_is_instance(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def _project_manifest_payload(
    *,
    project: ProjectConfig,
    detector_spec: DetectorSpec,
    stages: Sequence[str],
    output_dir: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "backend": "native_lightning",
        "detector": _to_jsonable(detector_spec),
        "dataset": {
            "format": project.dataset.format,
            "root": project.dataset.data_root,
            "images_dir": project.dataset.images_dir,
            "train": project.dataset.annot_file_train,
            "val": project.dataset.annot_file_val,
            "test": project.dataset.annot_file_test,
            "classes": list(project.dataset.categories or ()),
            "in_channels": project.dataset.in_channels,
        },
        "workdir": output_dir,
        "optimizer": _to_jsonable(project.optimization),
        "scheduler": {
            "name": project.optimization.scheduler_choice,
            "step_size": project.optimization.scheduler_step_size,
            "gamma": project.optimization.scheduler_gamma,
        },
        "runtime": _to_jsonable(project.runtime),
        "seed": project.seed,
        "stages": list(stages),
        "checkpoint": _to_jsonable(project.checkpoint),
        "export": _to_jsonable(project.export),
        "results": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_project_manifest(
    output_dir: str,
    payload: dict[str, Any],
) -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / PROJECT_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return manifest_path


def _build_native_project_config(
    *,
    dataset_root: str,
    categories: Sequence[str],
    detector_spec: DetectorSpec,
    in_channels: int,
    result_folder: Optional[str],
    kwargs: dict[str, Any],
    dataset: DatasetConfig | None = None,
    seed: int = 71,
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
        scheduler=kwargs.get("scheduler_choice"),
        scheduler_step_size=int(kwargs.get("scheduler_step_size", 1)),
        scheduler_gamma=float(kwargs.get("scheduler_gamma", 0.1)),
        max_epochs=int(kwargs.get("max_epochs", 1)),
        accelerator=str(kwargs.get("accelerator", "cpu")),
        devices=int(kwargs.get("devices", 1)),
        seed=int(seed),
        dataset_format=(dataset.format if dataset is not None else "coco"),
        images_dir=(dataset.images_dir if dataset is not None and dataset.images_dir else "images"),
        train_annotation_file=(dataset.annot_file_train if dataset is not None else None),
        val_annotation_file=(dataset.annot_file_val if dataset is not None else None),
        test_annotation_file=(dataset.annot_file_test if dataset is not None else None),
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
    stages: Sequence[str] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    project = _coerce_project_config(config)
    detector_spec = _coerce_detector_spec(
        project.detector_spec,
        model_cfg=project.model_cfg,
        detector=project.detector,
        in_channels=project.dataset.in_channels,
    )
    run_order = _normalize_project_stages(project.stages if stages is None else stages)
    if validate:
        validate_project_config(project, strict=True)
    output_dir = _project_workdir(project) or str(
        Path(project.dataset.data_root).expanduser() / "runs" / "simpledet"
    )
    native_config: Any | None = None

    def _native_config():
        nonlocal native_config
        if native_config is None:
            native_config = _build_native_project_config(
                dataset_root=project.dataset.data_root,
                categories=project.dataset.categories,
                detector_spec=detector_spec,
                in_channels=project.dataset.in_channels,
                result_folder=output_dir,
                kwargs={
                    "batch_size": project.runtime.batch_size,
                    "num_workers": project.runtime.num_workers,
                    "learning_rate": project.optimization.learning_rate,
                    "optimizer_choice": project.optimization.optimizer_choice,
                    "scheduler_choice": project.optimization.scheduler_choice,
                    "scheduler_step_size": project.optimization.scheduler_step_size,
                    "scheduler_gamma": project.optimization.scheduler_gamma,
                    "max_epochs": project.runtime.max_epochs,
                    "accelerator": project.runtime.accelerator,
                    "devices": project.runtime.devices,
                    "checkpoint_path": project.checkpoint.path,
                },
                dataset=project.dataset,
                seed=project.seed,
            )
        return native_config

    from .suite import compile_native_detector_plan

    result: dict[str, Any] = {
        "backend": "native_lightning",
        "architecture": detector_spec.architecture,
        "output_dir": output_dir,
        "stages": run_order,
    }
    for stage in run_order:
        if stage == "build":
            result["build"] = {"detector_plan": compile_native_detector_plan(detector_spec).to_dict()}
            continue
        if stage == "train":
            from .native.runtime import run_native_training as _run_native_training

            config_for_stage = _native_config()
            train_result = _run_native_training(config_for_stage)
            result["train"] = train_result
            checkpoint_path = train_result.get("checkpoint_path")
            if checkpoint_path:
                config_for_stage.checkpoint_path = str(checkpoint_path)
        else:
            from .native.runtime import run_native_inference as _run_native_inference

            result[stage] = _run_native_inference(_native_config())
    manifest_payload = _project_manifest_payload(
        project=project,
        detector_spec=detector_spec,
        stages=run_order,
        output_dir=output_dir,
        result=result,
    )
    manifest_path = _write_project_manifest(output_dir, manifest_payload)
    result["manifest_path"] = str(manifest_path)
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
    resolved_detector_spec = _coerce_detector_spec(
        detector_spec,
        model_cfg=model_cfg,
        in_channels=in_channels,
    )
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
            seed=int(kwargs.get("seed", 71)),
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
    resolved_detector_spec = _coerce_detector_spec(
        detector_spec,
        model_cfg=model_cfg,
        in_channels=in_channels,
    )
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
            seed=int(kwargs.get("seed", 71)),
        )
    )


def run_evaluation(**kwargs: Any) -> dict[str, Any]:
    return run_inference(**kwargs)


def predict_image(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .detectors.infer import predict_image as _predict_image

    return _predict_image(*args, **kwargs)


def predict_batch(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    from .detectors.infer import predict_batch as _predict_batch

    return _predict_batch(*args, **kwargs)


def load_checkpoint_for_inference(*args: Any, **kwargs: Any) -> Any:
    from .detectors.infer import load_checkpoint_for_inference as _load_checkpoint_for_inference

    return _load_checkpoint_for_inference(*args, **kwargs)


def export_predictions(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .detectors.infer import export_predictions as _export_predictions

    return _export_predictions(*args, **kwargs)


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


def list_heads(kind: str | None = None, pattern: str | None = None) -> list[str]:
    from .suite.catalog import list_heads as _list_heads

    return _list_heads(kind=kind, pattern=pattern)


def list_detectors(family: str | None = None, pattern: str | None = None) -> list[str]:
    from .suite import list_detectors as _list_detectors

    return _list_detectors(family=family, pattern=pattern)


def list_backbones(pattern: str | None = None) -> list[str]:
    from .suite import list_backbones as _list_backbones

    return _list_backbones(pattern=pattern)


def build_backbone(*args: Any, **kwargs: Any) -> Any:
    from .suite import build_backbone as _build_backbone

    return _build_backbone(*args, **kwargs)


def build_neck(*args: Any, **kwargs: Any) -> Any:
    from .suite import build_neck as _build_neck

    return _build_neck(*args, **kwargs)


def build_head(*args: Any, **kwargs: Any) -> Any:
    from .suite import build_head as _build_head

    return _build_head(*args, **kwargs)


def build_detector(*args: Any, **kwargs: Any) -> Any:
    from .suite import build_detector as _build_detector

    return _build_detector(*args, **kwargs)


def compile_native_detector_plan(*args: Any, **kwargs: Any) -> Any:
    from .suite import compile_native_detector_plan as _compile_native_detector_plan

    return _compile_native_detector_plan(*args, **kwargs)


def print_available_encoders(pattern: str | None = None) -> None:
    for name in list_available_encoders(pattern):
        print(name)


def print_available_necks() -> None:
    for name in list_available_necks():
        print(name)


def print_available_heads() -> None:
    for name in list_available_heads():
        print(name)
