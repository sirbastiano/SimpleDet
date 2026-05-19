"""Core package metadata for SimpleDet."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


__all__ = [
    "__version__",
    "train",
    "detect",
    "evaluate",
    "ProjectLayout",
    "DatasetConfig",
    "RuntimeConfig",
    "OptimizationConfig",
    "ProjectConfig",
    "load_project_config",
    "project_config_template",
    "init_project_config",
    "validate_project_config",
    "run_project",
    "run_training",
    "run_inference",
    "run_evaluation",
    "predict_image",
    "predict_batch",
    "load_checkpoint_for_inference",
    "export_predictions",
    "build_detector",
    "build_backbone",
    "build_neck",
    "build_head",
    "list_detectors",
    "list_heads",
    "list_backbones",
    "compile_native_detector_plan",
    "suite",
    "native",
    "extensions",
    "detectors",
]

try:
    __version__ = version("simpledet")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str):
    if name == "detectors":
        import simpledet.detectors as detectors

        return detectors
    if name == "suite":
        import simpledet.suite as suite

        return suite
    if name == "extensions":
        import simpledet.extensions as extensions

        return extensions
    if name == "native":
        import simpledet.native as native

        return native
    if name == "train":
        from .detectors import train as train_module

        return train_module.train
    if name == "detect":
        from .detectors import infer as infer_module

        return infer_module.detect
    if name == "evaluate":
        from .detectors import evaluate as evaluate_module

        return evaluate_module.evaluate
    if name in {
        "predict_image",
        "predict_batch",
        "load_checkpoint_for_inference",
        "export_predictions",
    }:
        from .detectors import infer as infer_module

        return getattr(infer_module, name)

    api = import_module(".api", __name__)

    exported = {
        "ProjectLayout",
        "DatasetConfig",
        "RuntimeConfig",
        "OptimizationConfig",
        "ProjectConfig",
        "load_project_config",
        "project_config_template",
        "init_project_config",
        "validate_project_config",
        "run_project",
        "run_training",
        "run_inference",
        "run_evaluation",
        "build_detector",
        "build_backbone",
        "build_neck",
        "build_head",
        "list_detectors",
        "list_heads",
        "list_backbones",
        "compile_native_detector_plan",
    }
    if name in exported:
        return getattr(api, name)
    raise AttributeError(f"module 'simpledet' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + __all__))
