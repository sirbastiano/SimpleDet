"""Public exception taxonomy for SimpleDet."""

from __future__ import annotations


class SimpleDetError(Exception):
    """Base class for public SimpleDet exceptions."""


class OptionalDependencyError(SimpleDetError, ImportError):
    """Raised when an optional runtime dependency is required but unavailable."""

    def __init__(
        self,
        message: str,
        *,
        module_name: str | None = None,
        feature: str | None = None,
        install_command: str | None = None,
    ) -> None:
        self.module_name = module_name
        self.feature = feature
        self.install_command = install_command
        super().__init__(message)


class RegistryError(SimpleDetError, LookupError):
    """Base class for public registry failures."""


class RegistryLookupError(RegistryError):
    """Raised when a public registry cannot resolve a component name or alias."""


class ConfigValidationError(SimpleDetError, ValueError):
    """Raised when project or runtime configuration is invalid."""


class ConfigPathError(ConfigValidationError, FileNotFoundError):
    """Raised when config validation fails because required paths are missing."""


class DatasetError(SimpleDetError, ValueError):
    """Base class for dataset loading and validation failures."""


class DatasetPathError(DatasetError, FileNotFoundError):
    """Raised when a dataset path required by a public API is missing."""


class TensorContractError(SimpleDetError, ValueError):
    """Raised when tensors violate a declared SimpleDet runtime contract."""


class CheckpointError(SimpleDetError):
    """Base class for checkpoint loading and validation failures."""


class CheckpointPathError(CheckpointError, FileNotFoundError):
    """Raised when a checkpoint path is missing or unsupported."""


__all__ = [
    "SimpleDetError",
    "OptionalDependencyError",
    "RegistryError",
    "RegistryLookupError",
    "ConfigValidationError",
    "ConfigPathError",
    "DatasetError",
    "DatasetPathError",
    "TensorContractError",
    "CheckpointError",
    "CheckpointPathError",
]
