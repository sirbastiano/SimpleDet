"""Native SimpleDet extension registries."""

from __future__ import annotations

from .registry import (
    ASSIGNERS,
    DECODERS,
    DETECTORS,
    ENCODERS,
    HEADS,
    LOSSES,
    NECKS,
    POSTPROCESSORS,
    ComponentMetadata,
    DependencyRequirement,
    ExtensionRegistry,
    MissingComponentDependencyError,
    RegistryLookupError,
)

__all__ = [
    "ComponentMetadata",
    "DependencyRequirement",
    "ExtensionRegistry",
    "RegistryLookupError",
    "MissingComponentDependencyError",
    "ENCODERS",
    "NECKS",
    "HEADS",
    "DECODERS",
    "DETECTORS",
    "LOSSES",
    "ASSIGNERS",
    "POSTPROCESSORS",
]
