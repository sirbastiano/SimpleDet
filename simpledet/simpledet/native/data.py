"""Native dataset and datamodule support for the Lightning backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..detectors.data import load_dataset
from ..errors import DatasetError


TransformFn = Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]]


class NativeDataValidationError(DatasetError):
    """Raised when native datamodule inputs are unusable for a stage."""


def _resolve_annotation_path(dataset_root: str, split: str) -> str:
    root = Path(dataset_root).expanduser()
    split_name = _normalize_split(split).lower()
    file_stem = "val" if split_name in {"valid", "validation"} else split_name
    names = {
        "train": "train_annotations.json",
        "val": "val_annotations.json",
        "test": "test_annotations.json",
    }
    if file_stem not in names:
        raise ValueError(f"Unsupported dataset split '{split}'.")

    candidates = (
        root / "Annotations" / names[file_stem],
        root / "annotations" / f"instances_{file_stem}.json",
        root / f"instances_{file_stem}.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


class NativeDetectionDataset:
    """Detection dataset backed by the repo's normalized dataset adapters."""

    def __init__(
        self,
        annotation_path: str,
        *,
        split: str,
        in_channels: int,
        dataset_root: str | None = None,
        format: str | None = "coco",
        images_dir: str = "images",
        transforms: TransformFn | None = None,
    ) -> None:
        annotation_file = Path(annotation_path).expanduser()
        root = Path(dataset_root).expanduser() if dataset_root is not None else _infer_dataset_root(annotation_file)
        self.split = _normalize_split(split)
        self.in_channels = max(int(in_channels), 1)
        self.transforms = transforms
        self.annotation_path = annotation_file
        self.dataset_root = root
        self.payload = load_dataset(
            str(root),
            format=format,
            annotation_file=str(annotation_file),
            images_dir=images_dir,
            split=self.split,
        )
        self.categories = list(self.payload.get("categories", []))
        self.samples = _select_split_samples(self.payload, self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _build_image_tensor(sample, in_channels=self.in_channels)
        target = _build_target(sample, index=index, split=self.split)
        if self.transforms is not None:
            transformed = self.transforms(image, target)
            if not isinstance(transformed, tuple) or len(transformed) != 2:
                raise TypeError("Native detection transforms must return (image, target).")
            image, target = transformed
        return image, target


def _infer_dataset_root(annotation_path: Path) -> Path:
    if annotation_path.parent.name.lower() in {"annotations", "annotation"}:
        return annotation_path.parent.parent
    return annotation_path.parent


def _normalize_split(split: str) -> str:
    normalized = str(split).strip()
    if not normalized:
        raise ValueError("Native dataset split must be a non-empty string.")
    return "val" if normalized.lower() in {"valid", "validation"} else normalized


def _select_split_samples(payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    target_split = _normalize_split(split).lower()
    samples = []
    for sample in payload.get("samples", []):
        sample_split = _normalize_split(str(sample.get("split", split))).lower()
        if sample_split == target_split:
            samples.append(sample)
    return samples


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "int64":
        return getattr(torch, "int64", getattr(torch, "long", None))
    return getattr(torch, name, None)


def _build_image_tensor(sample: dict[str, Any], *, in_channels: int):
    import torch

    file_path = sample.get("file_path")
    if file_path:
        try:
            from torchvision.io import read_image  # type: ignore

            image = read_image(str(file_path)).to(dtype=_torch_dtype(torch, "float32"))
            return _match_image_channels(image, in_channels)
        except ImportError:
            pass

    width = int(sample.get("width") or 1)
    height = int(sample.get("height") or 1)
    return torch.zeros(
        (max(int(in_channels), 1), max(height, 1), max(width, 1)),
        dtype=_torch_dtype(torch, "float32"),
    )


def _match_image_channels(image: Any, in_channels: int):
    import torch

    channels = max(int(in_channels), 1)
    current = int(image.shape[0])
    if current == channels:
        return image
    if current > channels:
        return image[:channels]
    padding = torch.zeros(
        (channels - current, int(image.shape[1]), int(image.shape[2])),
        dtype=getattr(image, "dtype", _torch_dtype(torch, "float32")),
    )
    return torch.cat([image, padding], dim=0)


def _build_target(sample: dict[str, Any], *, index: int, split: str) -> dict[str, Any]:
    import torch

    annotations = list(sample.get("annotations", []))
    boxes = []
    labels = []
    area = []
    iscrowd = []
    for annotation in annotations:
        bbox = annotation.get("bbox", {})
        boxes.append(
            [
                float(bbox.get("x_min", 0.0)),
                float(bbox.get("y_min", 0.0)),
                float(bbox.get("x_max", 0.0)),
                float(bbox.get("y_max", 0.0)),
            ]
        )
        labels.append(int(annotation.get("category_id", 0)))
        area.append(float(annotation.get("area", 0.0)))
        iscrowd.append(int(annotation.get("iscrowd", 0)))

    metadata = {
        "image_id": int(sample.get("image_id", index)),
        "file_name": sample.get("file_name"),
        "file_path": sample.get("file_path"),
        "width": int(sample.get("width") or 1),
        "height": int(sample.get("height") or 1),
        "split": _normalize_split(str(sample.get("split", split))),
        "annotation_count": len(annotations),
    }
    target = {
        "boxes": torch.as_tensor(boxes, dtype=_torch_dtype(torch, "float32")),
        "labels": torch.as_tensor(labels, dtype=_torch_dtype(torch, "int64")),
        "image_id": torch.tensor([metadata["image_id"]], dtype=_torch_dtype(torch, "int64")),
        "area": torch.as_tensor(area, dtype=_torch_dtype(torch, "float32")),
        "iscrowd": torch.as_tensor(iscrowd, dtype=_torch_dtype(torch, "int64")),
        "metadata": metadata,
    }
    if _tensor_numel(target["boxes"]) == 0 and hasattr(target["boxes"], "view"):
        target["boxes"] = target["boxes"].view(0, 4)
    return target


def _tensor_numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    shape = getattr(value, "shape", ())
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return int(count)


def detection_collate_fn(batch: Iterable[tuple[Any, Any]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass(slots=True)
class NativeDataConfig:
    dataset_root: str
    in_channels: int = 3
    batch_size: int = 2
    num_workers: int = 0
    format: str | None = "coco"
    images_dir: str = "images"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    train_annotation_file: str | None = None
    val_annotation_file: str | None = None
    test_annotation_file: str | None = None
    transforms: TransformFn | None = None
    train_transforms: TransformFn | None = None
    val_transforms: TransformFn | None = None
    test_transforms: TransformFn | None = None
    seed: int = 71
    shuffle_train: bool = True


class NativeDetectionDataModule:
    """Lightning-compatible datamodule without hard dependency at import time."""

    def __init__(self, config: NativeDataConfig) -> None:
        self.config = config
        self.train_dataset: NativeDetectionDataset | None = None
        self.val_dataset: NativeDetectionDataset | None = None
        self.test_dataset: NativeDetectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        normalized_stage = _normalize_stage(stage)
        if normalized_stage in {None, "fit"}:
            if self.train_dataset is None:
                self.train_dataset = self._build_dataset(
                    self.config.train_split,
                    annotation_file=self.config.train_annotation_file,
                    transforms=self.config.train_transforms or self.config.transforms,
                )
                _validate_non_empty(self.train_dataset, split="train")
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset(
                    self.config.val_split,
                    annotation_file=self.config.val_annotation_file,
                    transforms=self.config.val_transforms or self.config.transforms,
                )
                _validate_non_empty(self.val_dataset, split="val")
        if normalized_stage in {None, "test", "predict"}:
            if self.test_dataset is None:
                self.test_dataset = self._build_dataset(
                    self.config.test_split,
                    annotation_file=self.config.test_annotation_file,
                    transforms=self.config.test_transforms or self.config.transforms,
                )
                _validate_non_empty(self.test_dataset, split="test")

    def _build_dataset(
        self,
        split: str,
        *,
        annotation_file: str | None,
        transforms: TransformFn | None,
    ) -> NativeDetectionDataset:
        return NativeDetectionDataset(
            _resolve_config_annotation_file(self.config.dataset_root, annotation_file, split),
            split=split,
            in_channels=self.config.in_channels,
            dataset_root=self.config.dataset_root,
            format=self.config.format,
            images_dir=self.config.images_dir,
            transforms=transforms,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")
        return self._make_dataloader(self.train_dataset, shuffle=bool(self.config.shuffle_train))

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup("fit")
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup("test")
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def _make_dataloader(self, dataset: NativeDetectionDataset | None, *, shuffle: bool):
        from torch.utils.data import DataLoader

        if dataset is None:
            raise NativeDataValidationError("Native detection dataset was not initialized.")
        kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "collate_fn": detection_collate_fn,
        }
        generator = _build_torch_generator(self.config.seed) if shuffle else None
        if generator is not None:
            kwargs["generator"] = generator
        return DataLoader(dataset, **kwargs)


def _resolve_config_annotation_file(dataset_root: str, annotation_file: str | None, split: str) -> str:
    if annotation_file is None:
        return _resolve_annotation_path(dataset_root, split)
    path = Path(annotation_file).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(dataset_root).expanduser() / path)


def _normalize_stage(stage: str | None) -> str | None:
    if stage is None:
        return None
    normalized = str(stage).strip().lower()
    if normalized in {"", "all"}:
        return None
    if normalized in {"validate", "validation"}:
        return "fit"
    return normalized


def _validate_non_empty(dataset: NativeDetectionDataset, *, split: str) -> None:
    if len(dataset) == 0:
        raise NativeDataValidationError(
            f"Native detection {split} split is empty; "
            f"check dataset_root='{dataset.dataset_root}' and annotation_file='{dataset.annotation_path}'."
        )


def _build_torch_generator(seed: int):
    try:
        import torch
    except ModuleNotFoundError:
        return None
    generator_factory = getattr(torch, "Generator", None)
    if generator_factory is None:
        return None
    generator = generator_factory()
    manual_seed = getattr(generator, "manual_seed", None)
    if callable(manual_seed):
        return manual_seed(int(seed))
    return generator
