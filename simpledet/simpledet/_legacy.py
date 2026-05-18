"""Legacy runtime boundary helpers."""

from __future__ import annotations

from os import PathLike

LEGACY_PY_CONFIG_ERROR = (
    "Legacy MMDetection .py config import/conversion is unsupported. "
    "Use a native SimpleDet .json or .toml config, or provide a specific "
    "converter for this legacy config before loading it."
)


def legacy_py_config_error(path: str | PathLike[str]) -> ValueError:
    return ValueError(f"{LEGACY_PY_CONFIG_ERROR} Requested path: {path}.")
