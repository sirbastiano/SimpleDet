"""Native SimpleDet extension registries."""

from __future__ import annotations

from .registry import DECODERS, DETECTORS, ENCODERS, HEADS, NECKS, ExtensionRegistry

__all__ = [
    "ExtensionRegistry",
    "ENCODERS",
    "NECKS",
    "HEADS",
    "DECODERS",
    "DETECTORS",
]
