"""Evaluation entry points for the detector public API."""

from __future__ import annotations

from typing import Any

def evaluate(
    *,
    pipeline: Any | None = None,
    build: bool = True,
    **pipeline_kwargs: Any,
):
    """Evaluate detections and return evaluation outputs for a pipeline."""
    from .infer import detect

    return detect(pipeline=pipeline, build=build, **pipeline_kwargs)
