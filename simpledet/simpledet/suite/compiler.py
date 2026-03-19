"""Legacy compiler shim retained only for explicit migration failures."""

from __future__ import annotations

from .specs import DetectorSpec


def compile_detector_spec(spec: DetectorSpec) -> dict[str, object]:
    if not isinstance(spec, DetectorSpec):
        raise TypeError("`spec` must be an instance of DetectorSpec.")
    raise RuntimeError(
        "compile_detector_spec is a retired MMDet-era compiler. "
        "Use simpledet.suite.compile_native_detector_plan(...) and the native runtime instead."
    )
