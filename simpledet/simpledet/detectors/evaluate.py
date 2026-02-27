"""Evaluation entry points for the detector public API."""

from __future__ import annotations

from typing import Any

from ._deps import require_detector_runtime


def evaluate(
    *,
    pipeline: Any | None = None,
    build: bool = True,
    **pipeline_kwargs: Any,
):
    """Evaluate detections and return evaluation outputs for a pipeline."""
    require_detector_runtime("evaluate")

    from simpledet.api import ObjectDetectionPipeline

    if pipeline is None:
        if not pipeline_kwargs:
            raise TypeError(
                "Provide either `pipeline` or the constructor keyword arguments "
                "required by ObjectDetectionPipeline."
            )
        pipeline = ObjectDetectionPipeline(**pipeline_kwargs)

    if build:
        pipeline.build()

    return pipeline.test()
