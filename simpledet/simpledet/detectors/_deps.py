"""Shared dependency helpers for the public detectors API."""

from __future__ import annotations

from importlib import import_module


_EXTRA_INSTALL_HINTS = {
    "timm": "timm",
}


def _missing_dependency(module_name: str, feature: str) -> str:
    extra = _EXTRA_INSTALL_HINTS.get(module_name)
    install_hint = (
        f" Install with `python -m pip install 'simpledet[{extra}]'`."
        if extra is not None
        else " Install this dependency before importing this symbol."
    )
    return (
        f"simpledet public API '{feature}' requires the optional dependency "
        f"'{module_name}'.{install_hint}"
    )


def require_dependency(module_name: str, feature: str) -> None:
    """Raise ImportError with a clear message when a runtime dependency is missing."""
    try:
        import_module(module_name)
    except ModuleNotFoundError as exc:
        dependency = exc.name or module_name
        raise ImportError(_missing_dependency(dependency, feature)) from exc


def require_detector_runtime(feature: str = "detectors") -> None:
    """Validate optional runtime dependencies used by training and inference APIs."""
    for dependency in ("torch", "torchvision"):
        require_dependency(dependency, feature)


def require_config_dependency(feature: str = "config") -> None:
    """Validate the dependency used by Config helpers."""
    require_dependency("numpy", feature)
