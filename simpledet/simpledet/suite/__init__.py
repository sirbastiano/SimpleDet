"""Canonical detector suite builders and compilers."""

from __future__ import annotations

from .catalog import build_decoder, build_detector, build_encoder, build_head, build_neck
from .compiler import compile_detector_spec
from .specs import DecoderSpec, DetectorSpec, EncoderSpec, HeadSpec, NeckSpec

__all__ = [
    "DecoderSpec",
    "DetectorSpec",
    "EncoderSpec",
    "HeadSpec",
    "NeckSpec",
    "build_decoder",
    "build_detector",
    "build_encoder",
    "build_head",
    "build_neck",
    "compile_detector_spec",
]

