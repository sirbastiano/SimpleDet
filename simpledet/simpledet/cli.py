"""Command-line interface for the simpledet package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Iterable

from . import __version__


@dataclass(frozen=True, slots=True)
class _DoctorDependency:
    module: str
    extra: str
    distribution: str | None = None


_DOCTOR_EXTRA_REQUIREMENTS: tuple[tuple[str, tuple[_DoctorDependency, ...]], ...] = (
    (
        "cpu",
        (
            _DoctorDependency("torch", "cpu"),
            _DoctorDependency("torchvision", "cpu"),
            _DoctorDependency("pytorch_lightning", "cpu", "pytorch-lightning"),
            _DoctorDependency("numpy", "cpu"),
            _DoctorDependency("scipy", "cpu"),
            _DoctorDependency("pycocotools", "cpu"),
            _DoctorDependency("terminaltables", "cpu"),
        ),
    ),
    ("timm", (_DoctorDependency("timm", "timm"),)),
    (
        "geo",
        (
            _DoctorDependency("pandas", "geo"),
            _DoctorDependency("rasterio", "geo"),
            _DoctorDependency("geopandas", "geo"),
            _DoctorDependency("shapely", "geo"),
        ),
    ),
    (
        "plots",
        (
            _DoctorDependency("scienceplots", "plots", "SciencePlots"),
            _DoctorDependency("matplotlib", "plots"),
        ),
    ),
    ("docs", ()),
)

_DOCTOR_DEPENDENCIES: tuple[_DoctorDependency, ...] = tuple(
    dependency
    for _extra, dependencies in _DOCTOR_EXTRA_REQUIREMENTS
    for dependency in dependencies
)

_DETECTOR_HELP = {
    "detr": {
        "summary": "Transformer detector with DETR-style set prediction and Hungarian matching behavior.",
        "family": "transformer",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "deformable_detr": {
        "summary": "Transformer detector variant with deformable attention-style defaults.",
        "family": "transformer",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "conditional_detr": {
        "summary": "Conditional DETR-style detector family routed to the native transformer path.",
        "family": "transformer",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "dab_detr": {
        "summary": "DAB-DETR detector with dynamic anchor query defaults on the native query path.",
        "family": "transformer",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "dino": {
        "summary": "DINO-style transformer detector family routed to the native transformer path.",
        "family": "transformer",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "cornernet": {
        "summary": "CornerNet heatmap detector with paired corner heatmaps, embeddings, and offsets.",
        "family": "dense",
        "recommended_encoders": ("resnet18", "resnet18.a1_in1k"),
    },
    "retinanet": {
        "summary": "Dense one-stage detector with FPN-style multiscale heads.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "fcos": {
        "summary": "Anchor-free dense detector with simpler head configuration.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "atss": {
        "summary": "Adaptive training sample selection detector built on the native dense runtime.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "gfl": {
        "summary": "Generalized focal loss detector on the native anchor-based dense runtime.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "vfnet": {
        "summary": "VFNet-style dense detector routed to native ATSS-style training and decoding.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "fovea": {
        "summary": "Fovea-family dense detector routed to compatible native FCOS-style primitives.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "foveabox": {
        "summary": "Foveabox-style dense detector using compatible FCOS-style native head behavior.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "reppoints": {
        "summary": "RepPoints-style dense detector with native ATSS-compatible head wiring.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolof": {
        "summary": "YOLOF-style dense detector with FCOS-compatible native defaults.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "centernet": {
        "summary": "CenterNet-style dense detector with compatible native FCOS defaults.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolo": {
        "summary": "YOLO-family one-stage detector routed to a shared native dense head.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolo3": {
        "summary": "YOLOv3-family detector using the native dense runtime head/decoder path.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolo_v3": {
        "summary": "YOLOv3-family dense alias.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolov3": {
        "summary": "YOLOv3-family dense alias.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolov5": {
        "summary": "YOLOv5-family dense alias.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolov6": {
        "summary": "YOLOv6-family dense alias.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolov7": {
        "summary": "YOLOv7-family dense alias.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolov8": {
        "summary": "YOLOv8-family dense alias.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "yolox": {
        "summary": "YOLOX dense detector with native objectness, class, and bbox branches.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "rtmdet": {
        "summary": "RTMDet dense detector using a native RTMDet-style head and task-aligned targets.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "tood": {
        "summary": "TOOD dense detector using native dense FCOS-style defaults.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "ssd": {
        "summary": "Single-shot dense detector routed to a native anchor-compatible path.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "efficientdet": {
        "summary": "EfficientDet dense detector with native anchor-based class and box heads.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "sabl": {
        "summary": "SABL dense detector routed to native anchor-compatible defaults.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "solov2": {
        "summary": "SOLOv2 dense detector routed to a native dense FCOS-style path.",
        "family": "dense",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "grid_rcnn": {
        "summary": "Grid R-CNN two-stage detector using native ROI construction.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "cascade_rcnn": {
        "summary": "Cascade R-CNN two-stage detector using native ROI construction.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "cascade_mask_rcnn": {
        "summary": "Cascade Mask R-CNN two-stage detector with cascade bbox and mask heads.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "double_head_rcnn": {
        "summary": "Double-Head R-CNN detector using separate native ROI classification and regression towers.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "dynamic_rcnn": {
        "summary": "Dynamic R-CNN detector using native dynamic ROI feature mixing.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "fast_rcnn": {
        "summary": "Fast R-CNN ROI detector that consumes external or target proposals without an RPN stage.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "faster_rcnn": {
        "summary": "Two-stage ROI detector for balanced accuracy and broad compatibility.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "libra_rcnn": {
        "summary": "Libra R-CNN detector using native balanced ROI sampling with two-stage heads.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "mask_rcnn": {
        "summary": "Two-stage ROI detector with instance-mask heads built on the native runtime.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "sparse_rcnn": {
        "summary": "Sparse R-CNN detector with learned proposal boxes/features and sparse ROI refinement.",
        "family": "roi",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
    "rpn": {
        "summary": "Standalone region proposal network detector that emits native proposal boxes and scores.",
        "family": "proposal",
        "recommended_encoders": ("resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simpledet",
        description="SimpleDet package bootstrap and diagnostics utility.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "list-detectors",
            "list-heads",
            "list-backbones",
            "list-necks",
            "list-datasets",
            "list-encoders",
            "doctor",
        ),
        help="Optional discovery command alias.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    parser.add_argument(
        "--check-runtime",
        dest="check_runtime",
        action="store_true",
        help=(
            "Validate optional runtime dependency resolution used by "
            "simpledet.api."
        ),
    )
    parser.add_argument(
        "--list-detectors",
        action="store_true",
        help="Print supported high-level detector architectures and exit.",
    )
    parser.add_argument(
        "--list-heads",
        action="store_true",
        help="Print supported native detection heads and exit.",
    )
    parser.add_argument(
        "--list-backbones",
        action="store_true",
        help="Print supported native backbone aliases and exit.",
    )
    parser.add_argument(
        "--list-necks",
        action="store_true",
        help="Print supported native necks and exit.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print supported dataset formats and exit.",
    )
    parser.add_argument(
        "--list-encoders",
        action="store_true",
        help="Backward-compatible alias for --list-backbones.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print optional extra dependency status and exit.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Make doctor return non-zero when any setup check fails.",
    )
    parser.add_argument(
        "--workdir",
        help="Work directory path for doctor writability checks. Default: current directory.",
    )
    parser.add_argument(
        "--family",
        help="Filter list-detectors by family: dense, proposal, roi, or transformer.",
    )
    parser.add_argument(
        "--kind",
        help="Filter list-heads by kind: dense, roi, or transformer.",
    )
    parser.add_argument(
        "--pattern",
        help="Case-insensitive substring filter for discovery commands.",
    )
    parser.add_argument(
        "--show-detector-help",
        metavar="NAME",
        help="Print a short explanation and recommended encoders for one detector architecture.",
    )
    parser.add_argument(
        "--init-project",
        metavar="PATH",
        help="Write a starter project config file (.toml or .json).",
    )
    parser.add_argument(
        "--project-format",
        choices=("toml", "json"),
        default=None,
        help="Explicit format for --init-project. Default: infer from suffix or use toml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting files for --init-project.",
    )
    parser.add_argument(
        "--project-validate",
        metavar="PATH",
        help="Validate a project config file (.json or .toml) and exit.",
    )
    parser.add_argument(
        "--project-run",
        metavar="PATH",
        help="Run a project config file (.json or .toml).",
    )
    parser.add_argument(
        "--train-root",
        metavar="PATH",
        help="Run training directly from a dataset root without a project config.",
    )
    parser.add_argument(
        "--infer-root",
        metavar="PATH",
        help="Run inference directly from a dataset root without a project config.",
    )
    parser.add_argument(
        "--eval-root",
        metavar="PATH",
        help="Run evaluation directly from a dataset root without a project config.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run with --project-run. Default: config stages or build train test",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Category names for direct train/infer/eval execution.",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        help="Input channel count for direct train/infer/eval execution.",
    )
    parser.add_argument(
        "--tif-channels-to-load",
        nargs="+",
        type=int,
        help="1-based TIFF band selection for direct train/infer/eval execution.",
    )
    parser.add_argument(
        "--result-folder",
        help="Override the output workdir for project or direct execution.",
    )
    parser.add_argument(
        "--detector",
        "--architecture",
        dest="detector",
        help="High-level detector architecture for direct execution.",
    )
    parser.add_argument(
        "--encoder",
        help="Encoder/backbone name used with --detector for direct execution.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Class count for --detector direct execution. Defaults to the number of categories.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        help="Resize used by direct train/infer/eval execution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size used by direct train/infer/eval execution.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Epoch count used by direct training execution.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate used by direct training execution.",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip pipeline path validation before execution.",
    )
    parser.set_defaults(validate=True)
    return parser


def _check_runtime() -> int:
    """Return non-zero when runtime dependencies are not installed."""
    try:
        from .detectors._deps import require_dependency

        for dependency in ("torch", "torchvision", "pytorch_lightning", "numpy"):
            require_dependency(dependency, "simpledet runtime")
    except ModuleNotFoundError as exc:
        dependency = getattr(exc, "name", "")
        if dependency:
            print(
                "simpledet runtime dependencies are not fully installed. "
                "Install with `simpledet[cpu]` before running this check."
            )
            print(f"Missing dependency: {dependency}")
        else:
            print(str(exc))
        return 1
    except ImportError as exc:
        print(str(exc))
        return 1

    print("SimpleDet native runtime stack is available.")
    return 0

def _list_detectors(family: str | None = None, pattern: str | None = None) -> int:
    from .discovery import detector_rows

    print("name\tfamily\tnative_validation")
    for row in detector_rows(family=family, pattern=pattern):
        print(f"{row.name}\t{row.kind}\t{row.validation_status}")
    return 0


def _print_component_rows(rows) -> None:
    print("name\tkind\tvalidation_status\trequired_extra")
    for row in rows:
        print(
            f"{row.name}\t{row.kind}\t{row.validation_status}\t"
            f"{row.required_extra_text}"
        )


def _list_heads(kind: str | None = None, pattern: str | None = None) -> int:
    from .discovery import head_rows

    _print_component_rows(head_rows(kind=kind, pattern=pattern))
    return 0


def _list_backbones(pattern: str | None = None) -> int:
    from .discovery import backbone_rows

    _print_component_rows(backbone_rows(pattern=pattern))
    return 0


def _list_necks(pattern: str | None = None) -> int:
    from .discovery import neck_rows

    _print_component_rows(neck_rows(pattern=pattern))
    return 0


def _list_datasets(pattern: str | None = None) -> int:
    from .discovery import dataset_rows

    _print_component_rows(dataset_rows(pattern=pattern))
    return 0


def _list_encoders(pattern: str | None = None) -> int:
    return _list_backbones(pattern=pattern)


def _doctor(strict: bool = False, workdir: str | None = None) -> int:
    python_status = _python_status()
    workdir_status, workdir_path, workdir_detail = _check_workdir(workdir)
    extra_rows = _doctor_extra_rows()
    dependency_rows = _doctor_dependency_rows()

    print("SimpleDet doctor")
    print(f"python\t{python_status}\t{sys.version.split()[0]}")
    print(f"simpledet\tinstalled\t{__version__}")
    print(f"workdir\t{workdir_status}\t{workdir_path}\t{workdir_detail}")
    print("")

    print("extras")
    print("extra\tstatus\tmissing\thint")
    for row in extra_rows:
        print("\t".join(row))
    print("")

    print("dependencies")
    print("dependency\tstatus\tversion\textra\thint")
    for row in dependency_rows:
        print("\t".join(row))

    failed = (
        python_status != "supported"
        or workdir_status != "writable"
        or any(row[1] != "installed" for row in extra_rows)
    )
    if failed:
        print("")
        print("Doctor result: failed" if strict else "Doctor result: warnings")
    else:
        print("")
        print("Doctor result: ok")
    return 1 if strict and failed else 0


def _python_status() -> str:
    return "supported" if (3, 10) <= sys.version_info[:2] < (3, 13) else "unsupported"


def _doctor_extra_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for extra, dependencies in _DOCTOR_EXTRA_REQUIREMENTS:
        missing = tuple(
            dependency.module
            for dependency in dependencies
            if not _dependency_available(dependency.module)
        )
        status = "installed" if not missing else "missing"
        rows.append(
            (
                extra,
                status,
                ",".join(missing) if missing else "-",
                "-" if not missing else _install_hint(extra),
            )
        )
    return rows


def _doctor_dependency_rows() -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    seen: set[str] = set()
    for dependency in _DOCTOR_DEPENDENCIES:
        if dependency.module in seen:
            continue
        seen.add(dependency.module)
        available = _dependency_available(dependency.module)
        rows.append(
            (
                dependency.module,
                "available" if available else "unavailable",
                _dependency_version(dependency) if available else "-",
                dependency.extra,
                "-" if available else _install_hint(dependency.extra),
            )
        )
    return rows


def _dependency_available(module: str) -> bool:
    try:
        return importlib_util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _dependency_version(dependency: _DoctorDependency) -> str:
    distribution = dependency.distribution or dependency.module
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "-"


def _check_workdir(workdir: str | None) -> tuple[str, str, str]:
    raw_path = workdir or os.getcwd()
    path = Path(raw_path).expanduser()
    display_path = str(path.resolve(strict=False))
    if not path.exists():
        return "missing", display_path, "path does not exist"
    if not path.is_dir():
        return "unwritable", display_path, "not a directory"
    try:
        with tempfile.NamedTemporaryFile(prefix=".simpledet-doctor-", dir=path):
            pass
    except OSError as exc:
        detail = exc.strerror or exc.__class__.__name__
        return "unwritable", display_path, detail
    return "writable", display_path, "-"


def _install_hint(extra: str) -> str:
    return f"python -m pip install 'simpledet[{extra}]'"


def _active_discovery_command(args: argparse.Namespace) -> str | None:
    if args.command:
        return args.command
    flag_commands = (
        ("list_detectors", "list-detectors"),
        ("list_heads", "list-heads"),
        ("list_backbones", "list-backbones"),
        ("list_necks", "list-necks"),
        ("list_datasets", "list-datasets"),
        ("list_encoders", "list-encoders"),
        ("doctor", "doctor"),
    )
    for attribute, command in flag_commands:
        if getattr(args, attribute):
            return command
    return None


def _validate_discovery_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    command: str,
) -> None:
    if args.family and command != "list-detectors":
        parser.error("--family is only supported by list-detectors")
    if args.kind and command != "list-heads":
        parser.error("--kind is only supported by list-heads")
    if args.family:
        valid_families = {"dense", "proposal", "roi", "transformer"}
        if args.family.strip().lower() not in valid_families:
            parser.error("--family must be one of: dense, proposal, roi, transformer")
    if args.kind:
        valid_kinds = {"dense", "roi", "transformer"}
        if args.kind.strip().lower() not in valid_kinds:
            parser.error("--kind must be one of: dense, roi, transformer")
    if args.strict and command != "doctor":
        parser.error("--strict is only supported by doctor")
    if args.workdir and command != "doctor":
        parser.error("--workdir is only supported by doctor")


def _run_discovery_command(command: str, args: argparse.Namespace) -> int:
    if command == "list-detectors":
        return _list_detectors(family=args.family, pattern=args.pattern)
    if command == "list-heads":
        return _list_heads(kind=args.kind, pattern=args.pattern)
    if command == "list-backbones":
        return _list_backbones(pattern=args.pattern)
    if command == "list-encoders":
        return _list_encoders(pattern=args.pattern)
    if command == "list-necks":
        return _list_necks(pattern=args.pattern)
    if command == "list-datasets":
        return _list_datasets(pattern=args.pattern)
    if command == "doctor":
        return _doctor(strict=args.strict, workdir=args.workdir)
    raise ValueError(f"Unsupported discovery command '{command}'.")


def _show_detector_help(name: str) -> int:
    from .suite.catalog import ARCHITECTURE_FAMILIES, _unknown_architecture_message, resolve_architecture_name

    normalized = resolve_architecture_name(name)
    if normalized not in ARCHITECTURE_FAMILIES:
        raise ValueError(_unknown_architecture_message(name, normalized))

    payload = _DETECTOR_HELP.get(
        normalized,
        {
            "summary": "High-level detector supported by the suite catalog.",
            "family": ARCHITECTURE_FAMILIES[normalized],
            "recommended_encoders": ("resnet18.a1_in1k",),
        },
    )
    print(f"name: {normalized}")
    print(f"family: {payload['family']}")
    print(f"summary: {payload['summary']}")
    print("recommended_encoders:")
    for encoder in payload["recommended_encoders"]:
        print(f"- {encoder}")
    return 0


def _validate_project(path: str) -> int:
    from . import validate_project_config

    report = validate_project_config(path, strict=False)
    print(json.dumps(report, indent=2))
    return 0 if not report.get("missing") else 1


def _run_project(path: str, stages: Iterable[str] | None) -> int:
    from . import run_project

    result = run_project(path, stages=tuple(stages) if stages is not None else None)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _init_project(path: str, project_format: str | None, force: bool) -> int:
    from . import init_project_config

    created_path = init_project_config(path, format=project_format, overwrite=force)
    print(created_path)
    return 0


def _load_direct_detector_spec(args: argparse.Namespace):
    from .suite import build_detector

    if not args.detector:
        return None
    num_classes = args.num_classes if args.num_classes is not None else len(args.categories)
    return build_detector(
        args.detector,
        encoder=args.encoder,
        num_classes=num_classes,
        in_channels=args.in_channels,
    )


def _direct_runtime_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {}
    if args.tif_channels_to_load is not None:
        kwargs["tif_channels_to_load"] = args.tif_channels_to_load
    if args.resize is not None:
        kwargs["resize"] = args.resize
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        kwargs["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    return kwargs


def _run_direct_training(args: argparse.Namespace) -> int:
    from . import run_training

    result = run_training(
        dataset_root=args.train_root,
        categories=tuple(args.categories),
        in_channels=args.in_channels,
        detector_spec=_load_direct_detector_spec(args),
        result_folder=args.result_folder,
        validate=args.validate,
        **_direct_runtime_kwargs(args),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _run_direct_inference(args: argparse.Namespace) -> int:
    from . import run_inference

    result = run_inference(
        dataset_root=args.infer_root,
        categories=tuple(args.categories),
        in_channels=args.in_channels,
        detector_spec=_load_direct_detector_spec(args),
        result_folder=args.result_folder,
        validate=args.validate,
        **_direct_runtime_kwargs(args),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _run_direct_evaluation(args: argparse.Namespace) -> int:
    from . import run_evaluation

    result = run_evaluation(
        dataset_root=args.eval_root,
        categories=tuple(args.categories),
        in_channels=args.in_channels,
        detector_spec=_load_direct_detector_spec(args),
        result_folder=args.result_folder,
        validate=args.validate,
        **_direct_runtime_kwargs(args),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """Run CLI and return process exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(__version__)
        return 0

    if args.check_runtime:
        return _check_runtime()

    discovery_command = _active_discovery_command(args)
    if discovery_command:
        _validate_discovery_options(parser, args, discovery_command)
        return _run_discovery_command(discovery_command, args)

    if args.strict:
        parser.error("--strict is only supported by doctor")
    if args.workdir:
        parser.error("--workdir is only supported by doctor")

    if args.show_detector_help:
        return _show_detector_help(args.show_detector_help)

    if args.init_project:
        return _init_project(args.init_project, args.project_format, args.force)

    if args.project_validate:
        return _validate_project(args.project_validate)

    if args.project_run:
        return _run_project(args.project_run, args.stages)

    direct_root = args.train_root or args.infer_root or args.eval_root
    if direct_root:
        if not args.categories:
            parser.error("--categories is required for direct execution")
        if args.in_channels is None:
            parser.error("--in-channels is required for direct execution")
        if not args.detector:
            parser.error("--detector is required for direct execution")
        if args.train_root:
            return _run_direct_training(args)
        if args.infer_root:
            return _run_direct_inference(args)
        return _run_direct_evaluation(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
