"""Core package metadata for SimpleDet."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


__all__ = [
    "__version__",
    "train",
    "detect",
    "evaluate",
    "Config",
    "detectors",
    "suite",
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

    if name == "train":
        from .detectors._deps import require_detector_runtime

        require_detector_runtime("train")
        from .detectors import train as train_module

        return train_module.train

    if name == "detect":
        from .detectors._deps import require_detector_runtime

        require_detector_runtime("detect")
        from .detectors import infer as infer_module

        return infer_module.detect

    if name == "evaluate":
        from .detectors._deps import require_detector_runtime

        require_detector_runtime("evaluate")
        from .detectors import evaluate as evaluate_module

        return evaluate_module.evaluate

    if name == "Config":
        from .detectors._deps import require_config_dependency

        require_config_dependency("Config")
        from .detectors import data as data_module

        return data_module.Config

    raise AttributeError(f"module 'simpledet' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + __all__))
