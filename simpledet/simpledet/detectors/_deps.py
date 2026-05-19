"""Shared dependency helpers for the public detectors API."""

from __future__ import annotations

from importlib import import_module

from ..errors import OptionalDependencyError


_EXTRA_INSTALL_HINTS = {
    "timm": "timm",
}


def _extra_install_command(extra: str) -> str:
    return f"python -m pip install 'simpledet[{extra}]'"


def _missing_dependency(module_name: str, feature: str) -> tuple[str, str | None]:
    extra = _EXTRA_INSTALL_HINTS.get(module_name)
    install_hint = (
        f" Install with `{_extra_install_command(extra)}`."
        if extra is not None
        else " Install this dependency before importing this symbol."
    )
    message = (
        f"simpledet public API '{feature}' requires the optional dependency "
        f"'{module_name}'.{install_hint}"
    )
    return message, (_extra_install_command(extra) if extra is not None else None)


def require_dependency(module_name: str, feature: str) -> None:
    """Raise OptionalDependencyError when a runtime dependency is missing."""
    try:
        import_module(module_name)
    except ModuleNotFoundError as exc:
        dependency = exc.name or module_name
        message, install_command = _missing_dependency(dependency, feature)
        raise OptionalDependencyError(
            message,
            module_name=dependency,
            feature=feature,
            install_command=install_command,
        ) from exc


def require_detector_runtime(feature: str = "detectors") -> None:
    """Validate optional runtime dependencies used by training and inference APIs."""
    for dependency in ("torch", "torchvision"):
        require_dependency(dependency, feature)


def require_config_dependency(feature: str = "config") -> None:
    """Validate the dependency used by Config helpers."""
    require_dependency("numpy", feature)
