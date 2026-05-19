"""Dataset and configuration helpers for the detector public API."""

from __future__ import annotations

import csv
import json
import math
import struct
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from .._legacy import legacy_py_config_error
from ..errors import DatasetError, DatasetPathError


class UnsupportedFormatError(DatasetError):
    """Raised when a requested dataset format is not registered."""

    def __init__(self, format_key: str, supported_formats: list[str]):
        self.format_key = format_key
        self.supported_formats = sorted(supported_formats)
        supported = ", ".join(self.supported_formats) or "<none>"
        super().__init__(
            f"Unsupported dataset format '{format_key}'. Supported formats: {supported}"
        )


class AmbiguousFormatError(DatasetError):
    """Raised when multiple dataset formats match the detected layout."""

    def __init__(self, path: Path, candidates: list[str]):
        self.path = path
        self.candidates = sorted(candidates)
        joined = ", ".join(self.candidates)
        super().__init__(
            f"Ambiguous dataset format for '{path}'. Detected possible formats: "
            f"{joined}. Pass explicit format='<format>' to disambiguate."
        )


class CocoSchemaError(DatasetError):
    """Raised when a COCO annotation file does not satisfy required schema."""

    def __init__(self, path: Path, missing: list[str], non_list: list[str] = ()):
        missing_text = (
            f"missing keys: {', '.join(sorted(missing))}"
            if missing
            else ""
        )
        non_list_text = (
            f"non-list keys: {', '.join(sorted(non_list))}"
            if non_list
            else ""
        )
        details = ", ".join(
            part for part in (missing_text, non_list_text) if part
        )
        super().__init__(f"Invalid COCO schema in '{path}': {details}")


DatasetPayload = dict[str, Any]


class DatasetAdapter(ABC):
    """Abstract dataset adapter interface.

    Parameters
    ----------
    format_key:
        Canonical format key used to look up this adapter.
    """

    format_key: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> DatasetPayload:
        """Load data from the provided path."""


_ADAPTERS: dict[str, type["DatasetAdapter"]] = {}

TAdapter = TypeVar("TAdapter", bound=type[DatasetAdapter])


def register_dataset_adapter(
    format_key: str,
    *,
    aliases: tuple[str, ...] | list[str] = (),
) -> Any:
    """Register a dataset adapter for one or more format keys.

    Registration is case-insensitive and guarantees deterministic format dispatch.
    """

    normalized_keys = {format_key.lower(), *(key.lower() for key in aliases)}

    def decorator(adapter_cls: TAdapter) -> TAdapter:
        if not issubclass(adapter_cls, DatasetAdapter):
            raise TypeError("Adapter classes must subclass DatasetAdapter.")

        for key in normalized_keys:
            if key in _ADAPTERS:
                raise ValueError(f"Dataset adapter for '{key}' is already registered.")
            _ADAPTERS[key] = adapter_cls

        return adapter_cls

    return decorator


def _get_adapter(format_key: str) -> type[DatasetAdapter]:
    adapter_key = format_key.lower().strip()
    adapter = _ADAPTERS.get(adapter_key)
    if adapter is None:
        raise UnsupportedFormatError(adapter_key, sorted(_ADAPTERS))
    return adapter


def list_formats() -> list[str]:
    """Return supported dataset format keys."""
    return sorted(_ADAPTERS)


def _is_json_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".json"


def _looks_like_coco_annotation(path: Path) -> bool:
    if not _is_json_file(path):
        return False

    if path.name.lower().startswith("instances_"):
        return True

    try:
        payload = _load_json_file(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and all(field in payload for field in ("images", "annotations", "categories"))
        and all(isinstance(payload[field], list) for field in ("images", "annotations", "categories"))
    )


def _has_any_files(dataset_root: Path, relative_dir: str, pattern: str) -> bool:
    directory = dataset_root / relative_dir
    if not directory.exists() or not directory.is_dir():
        return False
    return any(directory.rglob(pattern))


def _detect_dataset_format(dataset_root: Path, *, format_override: str | None = None) -> str:
    if format_override is not None:
        return format_override.strip()

    candidates: set[str] = set()

    if dataset_root.is_file():
        if _is_json_file(dataset_root):
            if _looks_like_coco_annotation(dataset_root):
                candidates.add("coco")
            else:
                candidates.add("json")
        elif dataset_root.suffix.lower() in {".jsonl", ".ndjson"}:
            candidates.add("json")
        elif dataset_root.suffix.lower() == ".csv":
            candidates.add("csv")
        if not candidates:
            raise DatasetError(f"Unable to auto-detect format for '{dataset_root}'.")
    else:
        if dataset_root.is_dir():
            if any(_is_json_file(path) and _looks_like_coco_annotation(path) for path in dataset_root.glob("instances_*.json")):
                candidates.add("coco")
            if any(_is_json_file(path) and _looks_like_coco_annotation(path) for path in (dataset_root / "annotations").glob("instances_*.json")):
                candidates.add("coco")
            if (
                _has_any_files(dataset_root, "labels", "*.txt")
                and _has_any_files(dataset_root, "images", "*")
            ):
                candidates.add("yolo")
            if (
                _has_any_files(dataset_root, "Annotations", "*.xml")
                and (dataset_root / "JPEGImages").is_dir()
            ):
                candidates.add("voc")
            if _has_any_files(dataset_root, ".", "*.csv"):
                candidates.add("csv")
            if any(path.suffix.lower() in {".jsonl", ".ndjson"} for path in dataset_root.glob("*")):
                candidates.add("json")
            if any(
                _is_json_file(path) and not path.name.lower().startswith("instances_")
                for path in dataset_root.glob("*.json")
            ):
                candidates.add("json")

            if not candidates:
                raise DatasetError(f"Unable to detect dataset format in '{dataset_root}'.")
        else:
            raise DatasetPathError(f"Dataset path not found: {dataset_root}")

    if len(candidates) != 1:
        raise AmbiguousFormatError(dataset_root, sorted(candidates))

    return next(iter(candidates))


def load_dataset(
    path: str,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> DatasetPayload:
    """Load a dataset using a registered adapter.

    Parameters
    ----------
    path:
        Path to the dataset root or format annotation file.
    format:
        Optional registry key for the format adapter (for example "coco") or omitted for
        auto-detection.
    """
    dataset_root = Path(path).expanduser()
    explicit_format = None if format in {None, "", "auto"} else format
    adapter_key = _detect_dataset_format(dataset_root, format_override=explicit_format)
    adapter_cls = _get_adapter(adapter_key)
    return adapter_cls().load(path, **kwargs)


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"COCO annotation file must contain a JSON object: {path}")
    return data


def _require_coco_list_fields(
    payload: dict[str, Any],
    source: str,
) -> None:
    missing = []
    invalid = []
    for field in ("images", "annotations", "categories"):
        if field not in payload:
            missing.append(field)
            continue
        if not isinstance(payload[field], list):
            invalid.append(field)
    if not missing and not invalid:
        return
    raise CocoSchemaError(Path(source), missing, invalid)


def _require_list_field(payload: dict[str, Any], field: str, source: str) -> list[dict[str, Any]]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ValueError(
            f"{source} annotations must include list field '{field}'."
        )
    return value


_GENERIC_REQUIRED_COLUMNS = ("path", "xmin", "ymin", "xmax", "ymax", "label")
_GENERIC_OPTIONAL_COLUMNS = ("split",)
_GENERIC_COLUMN_MAP = {
    "path": "path",
    "xmin": "xmin",
    "ymin": "ymin",
    "xmax": "xmax",
    "ymax": "ymax",
    "label": "label",
    "split": "split",
}


def _normalize_generic_columns(
    columns: dict[str, str] | None,
    *,
    source: str,
) -> dict[str, str]:
    mapping = dict(_GENERIC_COLUMN_MAP)
    if columns is None:
        return mapping

    if not isinstance(columns, dict):
        raise ValueError(f"{source} mapping must be a dictionary.")

    allowed = set(_GENERIC_REQUIRED_COLUMNS) | set(_GENERIC_OPTIONAL_COLUMNS)
    for key, value in columns.items():
        if key not in allowed:
            raise ValueError(f"{source} mapping has unknown key '{key}'.")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{source} mapping for '{key}' must be a non-empty column name."
            )
        mapping[key] = value.strip()

    missing = [key for key in _GENERIC_REQUIRED_COLUMNS if not mapping.get(key)]
    if missing:
        raise ValueError(
            f"{source} mapping missing required columns: {', '.join(missing)}."
        )
    return mapping


def _normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {
        (key.strip() if isinstance(key, str) else str(key)): value
        for key, value in row.items()
    }


def _normalize_split_name(value: Any, *, default: str = "train") -> str:
    if value is None:
        split = default
    elif isinstance(value, str):
        split = value.strip()
    else:
        split = str(value).strip()
    if not split:
        split = default
    if not split:
        raise ValueError("Dataset split must be a non-empty string.")
    return split


def _infer_split_from_name(path: Path, *, default: str = "train") -> str:
    name = path.stem.lower()
    for split in ("train", "val", "valid", "validation", "test"):
        if name == split or name.endswith(f"_{split}") or name.endswith(f"-{split}"):
            return "val" if split in {"valid", "validation"} else split
    return default


def _build_split_index(images: list[dict[str, Any]]) -> dict[str, list[int]]:
    splits: dict[str, list[int]] = defaultdict(list)
    for image in images:
        split = _normalize_split_name(image.get("split"), default="train")
        splits[split].append(int(image["image_id"]))
    return {name: ids for name, ids in sorted(splits.items())}


def _resolve_tabular_annotation_file(
    dataset_root: Path,
    annotation_file: str | Path,
    *,
    format_label: str,
    suffixes: set[str],
) -> Path:
    explicit_annotation = Path(annotation_file)
    if explicit_annotation.is_absolute():
        annotation_path = explicit_annotation
    elif dataset_root.is_file():
        candidate = dataset_root.parent / explicit_annotation
        if (
            dataset_root.name == explicit_annotation.name
            and dataset_root.suffix.lower() in suffixes
        ):
            annotation_path = dataset_root
        elif candidate.exists():
            annotation_path = candidate
        else:
            raise FileNotFoundError(
                f"Missing {format_label} annotation file: {candidate} (checked '{dataset_root}')"
            )
    else:
        annotation_path = dataset_root / explicit_annotation

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing {format_label} annotation file: {annotation_path}")
    if annotation_path.suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise ValueError(
            f"{format_label} annotation file must use one of {expected}: {annotation_path}"
        )
    return annotation_path


def _resolve_generic_json_annotation_file(
    dataset_root: Path,
    annotation_file: str | Path | None,
) -> Path:
    if annotation_file is not None:
        return _resolve_tabular_annotation_file(
            dataset_root,
            annotation_file,
            format_label="JSON",
            suffixes={".json", ".jsonl", ".ndjson"},
        )

    if dataset_root.is_file():
        if dataset_root.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
            raise ValueError(
                f"Unsupported JSON dataset path '{dataset_root}'. "
                "Expected '.json', '.jsonl', or '.ndjson'."
            )
        return dataset_root

    for name in ("annotations.jsonl", "annotations.ndjson", "annotations.json"):
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing JSON annotation file: {dataset_root / 'annotations.json'}")


def _validate_generic_columns_present(
    source_file: Path,
    row: dict[str, Any],
    columns: dict[str, str],
    *,
    row_no: int,
    source: str,
) -> None:
    missing = [
        canonical_col
        for canonical_col in _GENERIC_REQUIRED_COLUMNS
        if columns[canonical_col] not in row
    ]
    if missing:
        raise ValueError(
            f"{source} file '{source_file}' row {row_no} missing required columns: "
            f"{', '.join(sorted(columns[col] for col in missing))}."
        )


def _coerce_non_empty_str(
    value: Any,
    *,
    row_no: int,
    row_source: str,
    field: str,
) -> str:
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(
            f"{row_source} row {row_no} field '{field}' must be a non-empty string."
        )
    return text


def _coerce_float(
    value: Any,
    *,
    row_no: int,
    row_source: str,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{row_source} row {row_no} field '{field}' must be numeric, got boolean."
        )
    if isinstance(value, (int, float)):
        value_float = float(value)
    else:
        if not isinstance(value, str):
            raise ValueError(
                f"{row_source} row {row_no} field '{field}' must be numeric."
            )
        value = value.strip()
        if not value:
            raise ValueError(
                f"{row_source} row {row_no} field '{field}' must be numeric."
            )
        try:
            value_float = float(value)
        except ValueError as exc:
            raise ValueError(
                f"{row_source} row {row_no} field '{field}' is not a valid number: {value!r}."
            ) from exc

    if not math.isfinite(value_float):
        raise ValueError(
            f"{row_source} row {row_no} field '{field}' must be finite."
        )
    return value_float


def _coerce_split(
    split_value: Any,
    *,
    row_no: int,
    row_source: str,
    default_split: str,
) -> str:
    try:
        return _normalize_split_name(split_value, default=default_split)
    except ValueError as exc:
        raise ValueError(
            f"{row_source} row {row_no} split value is empty and no default split provided."
        ) from exc


def _read_generic_annotation_row(
    row: dict[str, Any],
    columns: dict[str, str],
    *,
    row_no: int,
    row_source: str,
    default_split: str,
) -> dict[str, Any]:
    path = _coerce_non_empty_str(
        row.get(columns["path"]),
        row_no=row_no,
        row_source=row_source,
        field="path",
    )
    x_min = _coerce_float(
        row.get(columns["xmin"]),
        row_no=row_no,
        row_source=row_source,
        field="xmin",
    )
    y_min = _coerce_float(
        row.get(columns["ymin"]),
        row_no=row_no,
        row_source=row_source,
        field="ymin",
    )
    x_max = _coerce_float(
        row.get(columns["xmax"]),
        row_no=row_no,
        row_source=row_source,
        field="xmax",
    )
    y_max = _coerce_float(
        row.get(columns["ymax"]),
        row_no=row_no,
        row_source=row_source,
        field="ymax",
    )
    if x_min >= x_max:
        raise ValueError(
            f"{row_source} row {row_no} has invalid bbox: xmin must be less than xmax."
        )
    if y_min >= y_max:
        raise ValueError(
            f"{row_source} row {row_no} has invalid bbox: ymin must be less than ymax."
        )

    label = _coerce_non_empty_str(
        row.get(columns["label"]),
        row_no=row_no,
        row_source=row_source,
        field="label",
    )
    split = _coerce_split(row.get(columns["split"]), row_no=row_no, row_source=row_source, default_split=default_split)

    return {
        "path": path,
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "label": label,
        "split": split,
    }


def _resolve_annotation_image_path(
    dataset_root: Path,
    images_root: Path,
    image_path: str,
    *,
    row_no: int,
    row_source: str,
) -> Path:
    path = Path(image_path)
    candidates = [path] if path.is_absolute() else [dataset_root / path, images_root / path]
    if not path.is_absolute() and path.name == path.as_posix():
        candidates.append(images_root / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{row_source} row {row_no} references missing image '{path}'."
    )


def _load_generic_annotations(
    annotation_rows: list[tuple[int, dict[str, Any]]],
    dataset_root: Path,
    *,
    columns: dict[str, str],
    default_split: str,
    images_dir: str | Path = "images",
    source_file: str,
    source: str,
) -> DatasetPayload:
    images_root = _resolve_dir(dataset_root, images_dir, name="images")
    parsed_images: list[dict[str, Any]] = []
    parsed_annotations: list[dict[str, Any]] = []
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_records: dict[tuple[str, Path], dict[str, Any]] = {}
    category_map: dict[str, int] = {}
    splits: dict[str, list[int]] = defaultdict(list)

    for row_no, row in annotation_rows:
        _validate_generic_columns_present(Path(source_file), row, columns, row_no=row_no, source=source)
        annotation = _read_generic_annotation_row(
            row,
            columns,
            row_no=row_no,
            row_source=source,
            default_split=default_split,
        )

        image_path = _resolve_annotation_image_path(
            dataset_root,
            images_root,
            annotation["path"],
            row_no=row_no,
            row_source=source,
        )
        width, height = _read_image_size(image_path)
        image_key = (annotation["split"], image_path)
        image_id = image_records.get(image_key, {}).get("image_id")
        if image_id is None:
            image_id = len(parsed_images) + 1
            image_record = {
                "image_id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "file_path": str(image_path),
                "split": annotation["split"],
            }
            image_records[image_key] = image_record
            parsed_images.append(image_record)
            splits[annotation["split"]].append(image_id)
        else:
            image_id = image_records[image_key]["image_id"]

        category_name = annotation["label"]
        if category_name not in category_map:
            category_map[category_name] = len(category_map)
        category_id = category_map[category_name]

        annotation_id = len(parsed_annotations) + 1
        x_min = annotation["bbox"]["x_min"]
        y_min = annotation["bbox"]["y_min"]
        x_max = annotation["bbox"]["x_max"]
        y_max = annotation["bbox"]["y_max"]
        parsed_annotation = {
            "id": annotation_id,
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "category_name": category_name,
            "bbox": annotation["bbox"],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0,
            "split": annotation["split"],
        }
        parsed_annotations.append(parsed_annotation)
        annotations_by_image[image_id].append(parsed_annotation)

    categories = [
        {"id": category_id, "name": category_name}
        for category_name, category_id in sorted(category_map.items(), key=lambda item: item[1])
    ]
    samples = [
        {
            "image_id": image_record["image_id"],
            "file_path": image_record["file_path"],
            "width": image_record["width"],
            "height": image_record["height"],
            "annotations": annotations_by_image.get(image_record["image_id"], []),
            "split": image_record["split"],
        }
        for image_record in parsed_images
    ]

    return {
        "images": parsed_images,
        "annotations": parsed_annotations,
        "samples": samples,
        "images_dir": str(images_root),
        "categories": categories,
        "category_map": {category_id: name for name, category_id in category_map.items()},
        "splits": {name: ids for name, ids in sorted(splits.items())},
        "meta": {
            "num_samples": len(samples),
            "num_images": len(parsed_images),
            "num_annotations": len(parsed_annotations),
            "num_categories": len(category_map),
        },
    }


@register_dataset_adapter("csv", aliases=("generic-csv",))
class GenericCsvDatasetAdapter(DatasetAdapter):
    """Adapter for generic CSV annotation files."""

    format_key = "csv"
    aliases = ("generic-csv",)

    def load(self, path: str, **kwargs: Any) -> DatasetPayload:
        dataset_root = Path(path).expanduser()
        annotation_file = kwargs.pop("annotation_file", None)
        images_dir = kwargs.pop("images_dir", "images")
        columns = _normalize_generic_columns(
            kwargs.pop("columns", None),
            source="CSV",
        )
        default_split = kwargs.pop("default_split", "train")

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

        if annotation_file is None and dataset_root.is_file():
            annotation_path = dataset_root
            if annotation_path.suffix.lower() != ".csv":
                raise ValueError(f"CSV annotation file must use '.csv' extension: {annotation_path}")
        else:
            annotation_path = _resolve_tabular_annotation_file(
                dataset_root,
                annotation_file or "annotations.csv",
                format_label="CSV",
                suffixes={".csv"},
            )
        annotation_root = dataset_root.parent if dataset_root.is_file() else dataset_root

        parsed_rows: list[tuple[int, dict[str, Any]]] = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            try:
                reader = csv.DictReader(handle)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Failed to parse CSV annotation file '{annotation_path}': {exc}"
                ) from exc

            if reader.fieldnames is None:
                raise ValueError(f"CSV annotation file '{annotation_path}' has no header row.")

            for row_no, row in enumerate(reader, start=2):
                parsed_rows.append((row_no, _normalize_row_keys(row)))

        dataset_payload = _load_generic_annotations(
            parsed_rows,
            annotation_root,
            columns=columns,
            default_split=default_split,
            images_dir=images_dir,
            source_file=str(annotation_path),
            source=f"CSV '{annotation_path}'",
        )
        dataset_payload.update(
            {
                "format": self.format_key,
                "path": str(dataset_root),
                "annotation_file": str(annotation_path),
                "source": str(annotation_path),
                "images_dir": str(_resolve_dir(annotation_root, images_dir, name="images")),
            }
        )
        return dataset_payload


@register_dataset_adapter("json", aliases=("jsonl", "generic-json", "ndjson"))
class GenericJsonlDatasetAdapter(DatasetAdapter):
    """Adapter for simple JSON, JSONL, and NDJSON annotation files."""

    format_key = "json"
    aliases = ("jsonl", "generic-json", "ndjson")

    def load(self, path: str, **kwargs: Any) -> DatasetPayload:
        dataset_root = Path(path).expanduser()
        annotation_file = kwargs.pop("annotation_file", None)
        images_dir = kwargs.pop("images_dir", "images")
        columns = _normalize_generic_columns(
            kwargs.pop("schema", None) or kwargs.pop("columns", None),
            source="JSON",
        )
        default_split = kwargs.pop("default_split", "train")

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

        annotation_path = _resolve_generic_json_annotation_file(dataset_root, annotation_file)
        annotation_root = dataset_root.parent if dataset_root.is_file() else dataset_root
        parsed_rows = self._read_rows(annotation_path)

        dataset_payload = _load_generic_annotations(
            parsed_rows,
            annotation_root,
            columns=columns,
            default_split=default_split,
            images_dir=images_dir,
            source_file=str(annotation_path),
            source=f"JSON '{annotation_path}'",
        )
        dataset_payload.update(
            {
                "format": self.format_key,
                "path": str(dataset_root),
                "annotation_file": str(annotation_path),
                "source": str(annotation_path),
                "images_dir": str(_resolve_dir(annotation_root, images_dir, name="images")),
            }
        )
        return dataset_payload

    @staticmethod
    def _read_rows(annotation_path: Path) -> list[tuple[int, dict[str, Any]]]:
        if annotation_path.suffix.lower() in {".jsonl", ".ndjson"}:
            return GenericJsonlDatasetAdapter._read_json_lines(annotation_path)

        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON annotation file '{annotation_path}' contains invalid JSON.") from exc

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = None
            for key in ("records", "annotations", "samples"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value
                    break
            if rows is None:
                raise ValueError(
                    f"JSON annotation file '{annotation_path}' must contain a list "
                    "or an object with 'records', 'annotations', or 'samples'."
                )
        else:
            raise ValueError(f"JSON annotation file '{annotation_path}' must contain a JSON array or object.")

        parsed_rows: list[tuple[int, dict[str, Any]]] = []
        for row_no, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSON annotation file '{annotation_path}' row {row_no} must contain a JSON object."
                )
            parsed_rows.append((row_no, _normalize_row_keys(row)))
        return parsed_rows

    @staticmethod
    def _read_json_lines(annotation_path: Path) -> list[tuple[int, dict[str, Any]]]:
        parsed_rows: list[tuple[int, dict[str, Any]]] = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for row_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON annotation file '{annotation_path}' row {row_no} "
                        "contains invalid JSON."
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSON annotation file '{annotation_path}' row {row_no} "
                        "must contain a JSON object."
                    )
                parsed_rows.append((row_no, _normalize_row_keys(row)))
        return parsed_rows


def _read_image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid PNG dimensions in '{path}'.")
            return width, height

        if not (len(header) >= 2 and header[:2] == b"\xff\xd8"):
            raise ValueError(f"Unsupported image format for '{path}'.")

        while True:
            marker_prefix = handle.read(1)
            if not marker_prefix:
                raise ValueError(f"Could not parse JPEG dimensions from '{path}'.")
            while marker_prefix == b"\xff":
                marker = handle.read(1)
                if not marker:
                    raise ValueError(f"Could not parse JPEG dimensions from '{path}'.")
                marker_prefix = marker
            marker_id = marker_prefix[0]
            if marker_id in {0xD9, 0xDA}:
                break

            segment_size = handle.read(2)
            if len(segment_size) != 2:
                raise ValueError(f"Could not parse JPEG dimensions from '{path}'.")
            segment_length = struct.unpack(">H", segment_size)[0]
            if segment_length < 2:
                raise ValueError(f"Malformed JPEG segment in '{path}'.")

            if marker_id in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                segment = handle.read(5)
                extra = handle.read(segment_length - 7)
                if len(extra) != segment_length - 7:
                    raise ValueError(f"Could not parse JPEG dimensions from '{path}'.")
                if len(segment) != 5:
                    raise ValueError(f"Malformed JPEG SOF segment in '{path}'.")
                _, height, width = struct.unpack(">BHH", segment)
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid JPEG dimensions in '{path}'.")
                return width, height


            handle.seek(segment_length - 2, 1)

        raise ValueError(f"Unsupported JPEG format in '{path}'.")


def _resolve_dir(dataset_root: Path, directory: str | Path, *, name: str) -> Path:
    explicit_dir = Path(directory)
    candidate = explicit_dir if explicit_dir.is_absolute() else dataset_root / explicit_dir
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {candidate}")
    return candidate


def _read_voc_xml(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid VOC XML in '{path}': {exc}") from exc


def _read_voc_str(node: ET.Element, tag: str, source: Path) -> str:
    text = node.findtext(tag)
    if text is None or not text.strip():
        raise ValueError(f"VOC annotation '{source}' missing required <{tag}> field.")
    return text.strip()


def _read_voc_int(node: ET.Element, tag: str, source: Path) -> int:
    text = _read_voc_str(node, tag, source)
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"VOC annotation '{source}' field <{tag}> must be an integer, got {text!r}."
        ) from exc
    if value <= 0:
        raise ValueError(
            f"VOC annotation '{source}' field <{tag}> must be positive, got {value!r}."
        )
    return value


def _find_image_for_voc_annotation(
    images_root: Path,
    annotation_path: Path,
) -> Path:
    root = _read_voc_xml(annotation_path)
    filename = _read_voc_str(root, "filename", annotation_path)
    candidate_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    parsed_filename = Path(filename)

    candidate_names: list[Path] = [images_root / parsed_filename]
    if parsed_filename != parsed_filename.name:
        candidate_names.append(images_root / parsed_filename.name)

    if not parsed_filename.suffix:
        for ext in candidate_exts:
            candidate_names.append(images_root / f"{parsed_filename.name}{ext}")

    for candidate in candidate_names:
        if candidate.exists():
            return candidate

    if parsed_filename.suffix:
        candidates = sorted(images_root.glob(f"**/{parsed_filename.name}"))
        if candidates:
            if len(candidates) > 1:
                raise ValueError(
                    f"VOC annotation '{annotation_path}' has ambiguous image candidates: "
                    f"{', '.join(str(match) for match in candidates)}"
                )
            return candidates[0]
        raise FileNotFoundError(
            f"Could not find image for VOC annotation '{annotation_path}'."
        )

    stem = annotation_path.stem
    for ext in candidate_exts:
        matches = sorted(images_root.glob(f"**/*{stem}{ext}"))
        if matches:
            if len(matches) > 1:
                raise ValueError(
                    f"VOC annotation '{annotation_path}' has ambiguous image candidates: "
                    f"{', '.join(str(match) for match in matches)}"
                )
            return matches[0]

    raise FileNotFoundError(f"Could not find image for VOC annotation '{annotation_path}'.")


def _parse_voc_bbox(
    bndbox: ET.Element,
    annotation_path: Path,
) -> tuple[float, float, float, float]:
    coords = {}
    for key in ("xmin", "ymin", "xmax", "ymax"):
        value_text = bndbox.findtext(key)
        if value_text is None:
            raise ValueError(
                f"VOC annotation '{annotation_path}' object is missing <{key}> inside <bndbox>."
            )
        try:
            value = float(value_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"VOC annotation '{annotation_path}' has non-numeric <{key}>: {value_text!r}."
            ) from exc
        if value < 0:
            raise ValueError(
                f"VOC annotation '{annotation_path}' has invalid <{key}> value: {value!r}."
            )
        coords[key] = value

    x_min, y_min, x_max, y_max = (
        coords["xmin"],
        coords["ymin"],
        coords["xmax"],
        coords["ymax"],
    )
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"VOC annotation '{annotation_path}' has invalid bbox: "
            f"({x_min}, {y_min}, {x_max}, {y_max})."
        )
    return x_min, y_min, x_max, y_max


def _load_voc_split_memberships(dataset_root: Path) -> dict[str, str]:
    split_dir = dataset_root / "ImageSets" / "Main"
    if not split_dir.exists():
        return {}

    memberships: dict[str, str] = {}
    for split_name, file_name in (
        ("train", "train.txt"),
        ("val", "val.txt"),
        ("test", "test.txt"),
    ):
        split_file = split_dir / file_name
        if not split_file.exists():
            continue
        for line_no, raw_line in enumerate(split_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            image_stem = line.split()[0]
            existing = memberships.get(image_stem)
            if existing is not None and existing != split_name:
                raise ValueError(
                    f"VOC image '{image_stem}' appears in both '{existing}' and "
                    f"'{split_name}' split files under '{split_dir}'."
                )
            memberships[image_stem] = split_name
    return memberships


def _parse_yolo_class_map(path: Path) -> dict[int, str]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Reading YOLO class map requires optional dependency 'pyyaml'. "
            "Install it to parse classes files."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YOLO classes file '{path}'. Expected YAML mapping.")

    names = payload.get("names")
    if names is None:
        names = payload.get("classes")
    if names is None:
        raise ValueError(
            "YOLO classes file must define 'names' (list or id->name mapping)."
        )

    class_map: dict[int, str] = {}
    if isinstance(names, list):
        for idx, name in enumerate(names):
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"Invalid YOLO classes entry at index {idx} in '{path}': expected class name."
                )
            class_map[idx] = name
        return class_map

    if isinstance(names, dict):
        for key, value in names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid YOLO class id '{key}' in '{path}'. Expected integer keys."
                ) from None
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Invalid YOLO class name for id {class_id} in '{path}'."
                )
            class_map[class_id] = value
        return class_map

    raise ValueError(
        f"Invalid 'names' field in '{path}'. Expected list or dictionary."
    )


def _find_image_for_label(
    images_root: Path,
    labels_root: Path,
    label_path: Path,
) -> Path:
    stem = label_path.stem
    relative_parent = label_path.parent.relative_to(labels_root)
    candidate_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    candidate_names: list[Path] = []
    for ext in candidate_exts:
        candidate_names.append(images_root / relative_parent / f"{stem}{ext}")
        candidate_names.append(images_root / f"{stem}{ext}")

    for candidate in candidate_names:
        if candidate.exists():
            return candidate

    for ext in candidate_exts:
        matches = sorted(images_root.glob(f"**/*{stem}{ext}"))
        if matches:
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous image match for label '{label_path}'. "
                    f"Found multiple matches: {', '.join(str(m) for m in matches)}"
                )
            return matches[0]

    raise FileNotFoundError(f"Could not find image for label file '{label_path}'.")


def _parse_yolo_bbox_line(
    line: str,
    *,
    label_path: Path,
    line_no: int,
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(
            f"Invalid YOLO label format in '{label_path}' line {line_no}: "
            "expected 'class x_center y_center width height'."
        )

    class_id_raw, x_center, y_center, width, height = parts
    try:
        class_value = float(class_id_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid YOLO class id '{class_id_raw}' in '{label_path}' line {line_no}."
        ) from exc
    if not class_value.is_integer():
        raise ValueError(
            f"YOLO class id must be an integer in '{label_path}' line {line_no}."
        )
    class_id = int(class_value)
    if class_id < 0:
        raise ValueError(
            f"YOLO class id must be non-negative in '{label_path}' line {line_no}."
        )

    coords = [x_center, y_center, width, height]
    numeric = []
    for value in coords:
        try:
            numeric_value = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric YOLO value '{value}' in '{label_path}' line {line_no}."
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"Non-finite YOLO value '{value}' in '{label_path}' line {line_no}.")
        numeric.append(numeric_value)

    x_c, y_c, norm_w, norm_h = numeric
    if (
        x_c < 0
        or x_c > 1
        or y_c < 0
        or y_c > 1
        or norm_w < 0
        or norm_h < 0
        or norm_w > 1
        or norm_h > 1
    ):
        raise ValueError(
            f"Out-of-range normalized values in '{label_path}' line {line_no}: "
            "coordinates must be in [0, 1]."
        )

    x_min_norm = x_c - (norm_w / 2)
    x_max_norm = x_c + (norm_w / 2)
    y_min_norm = y_c - (norm_h / 2)
    y_max_norm = y_c + (norm_h / 2)

    if (
        x_min_norm < 0
        or x_min_norm > 1
        or x_max_norm < 0
        or x_max_norm > 1
        or y_min_norm < 0
        or y_min_norm > 1
        or y_max_norm < 0
        or y_max_norm > 1
    ):
        raise ValueError(
            f"Out-of-range normalized values in '{label_path}' line {line_no}: "
            "bounding box extends outside image bounds."
        )

    x_min = x_min_norm * image_width
    x_max = x_max_norm * image_width
    y_min = y_min_norm * image_height
    y_max = y_max_norm * image_height
    if x_min > x_max or y_min > y_max:
        raise ValueError(
            f"YOLO label values in '{label_path}' line {line_no} produce invalid bbox."
        )

    return class_id, x_min, y_min, x_max, y_max


def _infer_yolo_split(label_path: Path, labels_root: Path, *, default: str = "train") -> str:
    relative = label_path.relative_to(labels_root)
    if len(relative.parts) > 1:
        return _normalize_split_name(relative.parts[0], default=default)
    return default


@register_dataset_adapter("yolo", aliases=("yolo-txt", "yolov5", "yolov8"))
class YoloDatasetAdapter(DatasetAdapter):
    """Adapter for YOLOv5/YOLOv8 style TXT annotations."""

    format_key = "yolo"
    aliases = ("yolo-txt", "yolov5", "yolov8")

    def load(self, path: str, **kwargs: Any) -> DatasetPayload:
        dataset_root = Path(path).expanduser()
        labels_dir = kwargs.pop("labels_dir", "labels")
        images_dir = kwargs.pop("images_dir", "images")
        classes_file = kwargs.pop("classes_file", None)
        default_split = kwargs.pop("default_split", "train")

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

        labels_root = _resolve_dir(dataset_root, labels_dir, name="labels")
        images_root = _resolve_dir(dataset_root, images_dir, name="images")
        class_map = self._load_class_map(dataset_root, classes_file)

        label_paths = sorted(labels_root.rglob("*.txt"))
        if not label_paths:
            raise FileNotFoundError(
                f"No YOLO label files found in: {labels_root}"
            )

        discovered_categories: set[int] = set()
        samples: list[dict[str, Any]] = []
        parsed_images: list[dict[str, Any]] = []
        parsed_annotations: list[dict[str, Any]] = []
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        image_records: dict[tuple[str, Path], dict[str, Any]] = {}

        for label_path in label_paths:
            split = _infer_yolo_split(label_path, labels_root, default=default_split)
            image_path = _find_image_for_label(images_root, labels_root, label_path)
            image_key = (split, image_path)
            image_record = image_records.get(image_key)
            if image_record is None:
                width, height = _read_image_size(image_path)
                image_id = len(parsed_images) + 1
                image_record = {
                    "image_id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "file_path": str(image_path),
                    "split": split,
                }
                parsed_images.append(image_record)
                image_records[image_key] = image_record

            image_id = image_record["image_id"]
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for line_no, raw_line in enumerate(lines, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                (
                    category_id,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ) = _parse_yolo_bbox_line(
                    line,
                    label_path=label_path,
                    line_no=line_no,
                    image_width=image_record["width"],
                    image_height=image_record["height"],
                )
                if class_map and category_id not in class_map:
                    raise ValueError(
                        f"YOLO annotation in '{label_path}' line {line_no} references "
                        f"unknown class id {category_id}."
                    )

                discovered_categories.add(category_id)
                category_name = class_map.get(category_id, str(category_id))
                annotation_id = len(parsed_annotations) + 1
                bbox = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
                width = x_max - x_min
                height = y_max - y_min
                annotation = {
                    "id": annotation_id,
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "category_name": category_name,
                    "bbox": bbox,
                    "area": width * height,
                    "iscrowd": 0,
                    "split": split,
                }
                parsed_annotations.append(annotation)
                annotations_by_image[image_id].append(annotation)

        samples = [
            {
                "image_id": image_record["image_id"],
                "file_path": image_record["file_path"],
                "width": image_record["width"],
                "height": image_record["height"],
                "annotations": annotations_by_image.get(image_record["image_id"], []),
                "split": image_record["split"],
            }
            for image_record in parsed_images
        ]

        if not class_map:
            class_map = {category_id: str(category_id) for category_id in sorted(discovered_categories)}

        categories = [
            {"id": category_id, "name": class_name}
            for category_id, class_name in sorted(class_map.items(), key=lambda item: item[0])
        ]

        return {
            "format": self.format_key,
            "path": str(dataset_root),
            "labels_dir": str(labels_root),
            "images_dir": str(images_root),
            "images": parsed_images,
            "annotations": parsed_annotations,
            "samples": samples,
            "categories": categories,
            "category_map": class_map,
            "splits": _build_split_index(parsed_images),
            "meta": {
                "num_samples": len(samples),
                "num_images": len(parsed_images),
                "num_annotations": len(parsed_annotations),
                "num_categories": len(class_map),
            },
        }

    @staticmethod
    def _load_class_map(dataset_root: Path, classes_file: str | None) -> dict[int, str]:
        if classes_file is None:
            return {}
        path = Path(classes_file)
        if not path.is_absolute():
            path = dataset_root / path
        if not path.exists():
            raise FileNotFoundError(f"YOLO classes file not found: {path}")
        return _parse_yolo_class_map(path)


@register_dataset_adapter("coco")
class CocoDatasetAdapter(DatasetAdapter):
    """Adapter for COCO-style annotation files."""

    format_key = "coco"
    aliases = ("coco-json", "coco_json")

    def load(self, path: str, **kwargs: Any) -> DatasetPayload:
        dataset_root = Path(path).expanduser()
        annotation_file = kwargs.pop("annotation_file", None)
        images_dir = kwargs.pop("images_dir", "images")
        annotation_path = self._resolve_annotation_file(dataset_root, annotation_file)
        split = _normalize_split_name(
            kwargs.pop("split", None),
            default=_infer_split_from_name(annotation_path),
        )
        image_dataset_root = self._resolve_dataset_root(dataset_root, annotation_path)

        payload = _load_json_file(annotation_path)
        _require_coco_list_fields(payload, str(annotation_path))
        images = _require_list_field(payload, "images", "COCO")
        annotations = _require_list_field(payload, "annotations", "COCO")
        categories = _require_list_field(payload, "categories", "COCO")

        category_map = self._parse_categories(categories)
        parsed_images = self._parse_images(
            images,
            image_dataset_root,
            images_dir,
            annotation_path=annotation_path,
            split=split,
        )
        image_ids = {image["image_id"] for image in parsed_images}
        parsed_annotations = self._parse_annotations(
            annotations,
            category_map,
            image_ids=image_ids,
            split=split,
        )
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for annotation in parsed_annotations:
            annotations_by_image[annotation["image_id"]].append(annotation)

        samples: list[dict[str, Any]] = []
        for image in parsed_images:
            image_id = image["image_id"]
            samples.append(
                {
                    "image_id": image_id,
                    "file_path": image["file_path"],
                    "width": image["width"],
                    "height": image["height"],
                    "annotations": annotations_by_image.get(image_id, []),
                    "split": image["split"],
                }
            )

        return {
            "format": self.format_key,
            "path": str(dataset_root),
            "annotation_file": str(annotation_path),
            "images": parsed_images,
            "annotations": parsed_annotations,
            "samples": samples,
            "categories": categories,
            "category_map": category_map,
            "images_dir": str(self._resolve_images_dir(image_dataset_root, annotation_path, images_dir)),
            "splits": _build_split_index(parsed_images),
            "meta": {
                "num_samples": len(samples),
                "num_images": len(parsed_images),
                "num_annotations": len(parsed_annotations),
                "num_categories": len(categories),
            },
        }

    def _parse_categories(self, categories: list[dict[str, Any]]) -> dict[int, str]:
        category_map: dict[int, str] = {}
        for category in categories:
            if not isinstance(category, dict):
                raise ValueError("COCO category entries must be dictionaries.")
            category_id = category.get("id")
            category_name = category.get("name")
            if not isinstance(category_id, int) or isinstance(category_id, bool):
                raise ValueError("Each COCO category must include integer 'id'.")
            if not isinstance(category_name, str) or not category_name.strip():
                raise ValueError(
                    "Each COCO category must include string 'name'."
                )
            category_map[category_id] = category_name.strip()

        return category_map

    def _resolve_images_dir(
        self,
        dataset_root: Path,
        annotation_path: Path,
        images_dir: str | Path,
    ) -> Path:
        explicit_dir = Path(images_dir)
        if explicit_dir.is_absolute():
            candidate = explicit_dir
        else:
            candidate = annotation_path.parent / explicit_dir
            if not candidate.exists() and (dataset_root / explicit_dir).exists():
                candidate = dataset_root / explicit_dir
        if not candidate.exists():
            raise FileNotFoundError(f"COCO image folder not found: {candidate}")
        return candidate

    @staticmethod
    def _safe_resolve_image_path(images_root: Path, candidate: Path) -> Path | None:
        try:
            resolved = candidate.resolve()
            return resolved if resolved.is_relative_to(images_root.resolve()) else None
        except OSError:
            return None

    def _find_image_file(
        self,
        root: Path,
        images_root: Path,
        file_name: str,
    ) -> Path:
        file_path = Path(file_name)
        candidates = [images_root / file_path, root / file_path]
        if file_path.name != file_path.as_posix():
            candidates.append(root / "images" / file_path)
            candidates.append(images_root / file_path.name)
        for candidate in candidates:
            safe_candidate = self._safe_resolve_image_path(images_root, candidate)
            if safe_candidate is not None and safe_candidate.exists():
                return safe_candidate
        raise FileNotFoundError(f"Missing COCO image file '{file_name}' in '{images_root}'")

    def _parse_images(
        self,
        images: list[dict[str, Any]],
        dataset_root: Path,
        images_dir: str | Path,
        *,
        annotation_path: Path,
        split: str,
    ) -> list[dict[str, Any]]:
        images_root = self._resolve_images_dir(
            dataset_root,
            annotation_path,
            images_dir,
        )
        parsed_images: list[dict[str, Any]] = []
        for image in images:
            if not isinstance(image, dict):
                raise ValueError("COCO image entries must be dictionaries.")
            image_id = image.get("id")
            file_name = image.get("file_name")
            if not isinstance(image_id, int) or isinstance(image_id, bool):
                raise ValueError("Each COCO image must include integer 'id'.")
            if not isinstance(file_name, str) or not file_name:
                raise ValueError("Each COCO image must include string 'file_name'.")
            file_path = self._find_image_file(dataset_root, images_root, file_name)
            width = image.get("width")
            height = image.get("height")
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or not isinstance(height, int)
                or isinstance(height, bool)
            ):
                raise ValueError(
                    f"COCO image '{file_name}' must include integer width/height."
                )
            if width <= 0 or height <= 0:
                raise ValueError(f"COCO image '{file_name}' width/height must be positive.")
            parsed_images.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "file_path": str(file_path),
                    "split": split,
                }
            )
        return parsed_images

    def _parse_annotations(
        self,
        annotations: list[dict[str, Any]],
        category_map: dict[int, str],
        *,
        image_ids: set[int],
        split: str,
    ) -> list[dict[str, Any]]:
        parsed_annotations: list[dict[str, Any]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotation entries must be dictionaries.")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if not isinstance(image_id, int) or isinstance(image_id, bool):
                raise ValueError(
                    "Each COCO annotation must include integer 'image_id'."
                )
            if image_id not in image_ids:
                raise ValueError(f"COCO annotation refers to unknown image_id {image_id}.")
            if not isinstance(category_id, int) or isinstance(category_id, bool):
                raise ValueError(
                    "Each COCO annotation must include integer 'category_id'."
                )
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(
                    f"COCO annotation for image_id {image_id} must include bbox [x,y,w,h]."
                )
            x_min, y_min, width, height = bbox
            if not all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in (x_min, y_min, width, height)
            ):
                raise ValueError(
                    f"COCO annotation for image_id {image_id} has non-numeric bbox values."
                )
            if not all(math.isfinite(float(value)) for value in (x_min, y_min, width, height)):
                raise ValueError(
                    f"COCO annotation for image_id {image_id} has non-finite bbox values."
                )
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"COCO annotation for image_id {image_id} has non-positive bbox width/height."
                )
            category_name = category_map.get(category_id)
            if category_name is None:
                raise ValueError(
                    f"COCO annotation refers to unknown category_id {category_id}."
                )
            area = annotation.get("area", width * height)
            parsed_annotations.append(
                {
                    "id": annotation.get("id"),
                    "annotation_id": annotation.get("id"),
                    "image_id": image_id,
                    "category_id": category_id,
                    "category_name": category_name,
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_min + width,
                        "y_max": y_min + height,
                    },
                    "area": area,
                    "iscrowd": annotation.get("iscrowd", 0),
                    "split": split,
                }
            )
        return parsed_annotations

    def _resolve_annotation_file(
        self,
        dataset_root: Path,
        annotation_file: str | None,
    ) -> Path:
        if annotation_file:
            explicit_path = Path(annotation_file)
            if explicit_path.is_absolute():
                annotation_path = explicit_path
            elif dataset_root.is_file():
                annotation_path = dataset_root.parent / explicit_path
            else:
                annotation_path = dataset_root / explicit_path
            if not annotation_path.exists():
                raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
            return annotation_path

        if dataset_root.is_file():
            if dataset_root.suffix.lower() != ".json":
                raise ValueError(
                    f"Unsupported COCO path '{dataset_root}'. "
                    "Expected a COCO json file or a directory containing one."
                )
            return dataset_root

        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

        default_annotation = dataset_root / "annotations" / "instances_train.json"
        candidates = [
            default_annotation,
            dataset_root / "instances_train.json",
            *sorted((dataset_root / "annotations").glob("instances_*.json")),
            *sorted(dataset_root.glob("instances_*.json")),
        ]
        candidates = [candidate for candidate in candidates if candidate.exists()]
        if not candidates:
            raise FileNotFoundError(f"Missing COCO annotation file: {default_annotation}")

        return candidates[0]

    @staticmethod
    def _resolve_dataset_root(dataset_root: Path, annotation_path: Path) -> Path:
        if dataset_root.is_dir():
            return dataset_root
        if annotation_path.parent.name.lower() == "annotations":
            return annotation_path.parent.parent
        return annotation_path.parent


@register_dataset_adapter("voc", aliases=("pascal-voc", "pascal_voc"))
class VocDatasetAdapter(DatasetAdapter):
    """Adapter for Pascal VOC XML annotations."""

    format_key = "voc"
    aliases = ("pascal-voc", "pascal_voc")

    def load(self, path: str, **kwargs: Any) -> DatasetPayload:
        dataset_root = Path(path).expanduser()
        annotations_dir = kwargs.pop("annotations_dir", "Annotations")
        images_dir = kwargs.pop("images_dir", "JPEGImages")

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

        annotations_root = _resolve_dir(
            dataset_root, annotations_dir, name="annotations"
        )
        images_root = _resolve_dir(dataset_root, images_dir, name="images")
        split_memberships = _load_voc_split_memberships(dataset_root)
        annotation_paths = sorted(annotations_root.rglob("*.xml"))
        if not annotation_paths:
            raise FileNotFoundError(
                f"No VOC XML files found in: {annotations_root}"
            )

        discovered_categories: set[str] = set()
        parsed_images: list[dict[str, Any]] = []
        parsed_annotations: list[dict[str, Any]] = []
        raw_annotations: list[dict[str, Any]] = []
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        image_records: dict[tuple[str, Path], dict[str, Any]] = {}

        for annotation_path in annotation_paths:
            root = _read_voc_xml(annotation_path)
            _, width, height = self._read_image_metadata(root, annotation_path)
            image_path = _find_image_for_voc_annotation(images_root, annotation_path)
            split = split_memberships.get(annotation_path.stem, "train")

            image_key = (split, image_path)
            image_record = image_records.get(image_key)
            if image_record is None:
                image_id = len(parsed_images) + 1
                image_record = {
                    "image_id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "file_path": str(image_path),
                    "split": split,
                }
                parsed_images.append(image_record)
                image_records[image_key] = image_record

            objects = root.findall("object")
            if not objects:
                raise ValueError(
                    f"VOC annotation '{annotation_path}' must include at least one <object> element."
                )

            image_id = image_record["image_id"]
            for obj in objects:
                category_name = obj.findtext("name")
                if category_name is None or not category_name.strip():
                    raise ValueError(
                        f"VOC annotation '{annotation_path}' has an object missing <name>."
                    )
                category_name = category_name.strip()
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    raise ValueError(
                        f"VOC annotation '{annotation_path}' object '{category_name}' "
                        "is missing <bndbox>."
                    )

                x_min, y_min, x_max, y_max = _parse_voc_bbox(bndbox, annotation_path)
                pose = obj.findtext("pose") or "Unspecified"
                difficult_raw = obj.findtext("difficult")
                try:
                    difficult = int(difficult_raw or 0)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"VOC annotation '{annotation_path}' object '{category_name}' "
                        f"has invalid <difficult>: {difficult_raw!r}."
                    )
                if difficult not in (0, 1):
                    raise ValueError(
                        f"VOC annotation '{annotation_path}' object '{category_name}' "
                        f"has invalid <difficult>: {difficult_raw!r}."
                    )

                discovered_categories.add(category_name)
                raw_annotations.append(
                    {
                        "category_name": category_name,
                        "image_id": image_id,
                        "bbox": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                        },
                        "pose": pose,
                        "difficult": difficult,
                        "split": split,
                    }
                )

        ordered_categories = sorted(discovered_categories)
        category_map = {
            category_id: category_name
            for category_id, category_name in enumerate(ordered_categories)
        }
        category_id_map = {
            category_name: category_id
            for category_id, category_name in category_map.items()
        }
        if not category_map:
            raise ValueError(
                f"VOC dataset has no valid object categories in: {annotations_root}"
            )

        categories = [
            {"id": category_id, "name": category_name}
            for category_id, category_name in sorted(category_map.items())
        ]

        for raw_annotation in raw_annotations:
            category_id = category_id_map[raw_annotation["category_name"]]
            annotation_id = len(parsed_annotations) + 1
            width = raw_annotation["bbox"]["x_max"] - raw_annotation["bbox"]["x_min"]
            height = raw_annotation["bbox"]["y_max"] - raw_annotation["bbox"]["y_min"]
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"VOC annotation for image {raw_annotation['image_id']} has invalid bbox."
                )
            annotation = {
                "id": annotation_id,
                "annotation_id": annotation_id,
                "image_id": raw_annotation["image_id"],
                "category_id": category_id,
                "category_name": raw_annotation["category_name"],
                "bbox": raw_annotation["bbox"],
                "area": width * height,
                "iscrowd": raw_annotation["difficult"],
                "pose": raw_annotation["pose"],
                "difficult": raw_annotation["difficult"],
                "split": raw_annotation["split"],
            }
            parsed_annotations.append(annotation)
            annotations_by_image[annotation["image_id"]].append(annotation)

        samples = [
            {
                "image_id": image_record["image_id"],
                "file_path": image_record["file_path"],
                "width": image_record["width"],
                "height": image_record["height"],
                "annotations": annotations_by_image.get(image_record["image_id"], []),
                "split": image_record["split"],
            }
            for image_record in parsed_images
        ]

        return {
            "format": self.format_key,
            "path": str(dataset_root),
            "annotations_dir": str(annotations_root),
            "images_dir": str(images_root),
            "images": parsed_images,
            "annotations": parsed_annotations,
            "samples": samples,
            "categories": categories,
            "category_map": category_map,
            "splits": _build_split_index(parsed_images),
            "meta": {
                "num_samples": len(samples),
                "num_images": len(parsed_images),
                "num_annotations": len(parsed_annotations),
                "num_categories": len(category_map),
            },
        }

    @staticmethod
    def _read_image_metadata(root: ET.Element, annotation_path: Path) -> tuple[str, int, int]:
        filename = _read_voc_str(root, "filename", annotation_path)
        size = root.find("size")
        if size is None:
            raise ValueError(f"VOC annotation '{annotation_path}' missing <size> element.")
        width = _read_voc_int(size, "width", annotation_path)
        height = _read_voc_int(size, "height", annotation_path)
        return filename, width, height


def create_config(config_path: str, **overrides: Any):
    """Load and merge a lightweight JSON or TOML configuration file."""

    path = Path(config_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "TOML config loading requires Python 3.11+ or the optional 'tomli' package."
                ) from exc
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    elif suffix == ".py":
        raise legacy_py_config_error(path)
    else:
        raise ValueError(
            f"Unsupported config format '{path.suffix or '<none>'}'. Expected '.json' or '.toml'."
        )

    if not isinstance(payload, dict):
        raise ValueError(f"Config file '{path}' must contain a top-level mapping.")

    if not overrides:
        return payload

    merged = dict(payload)
    for key, value in overrides.items():
        cursor = merged
        segments = [segment for segment in str(key).split(".") if segment]
        if not segments:
            continue
        for segment in segments[:-1]:
            next_value = cursor.get(segment)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[segment] = next_value
            cursor = next_value
        cursor[segments[-1]] = value
    return merged
