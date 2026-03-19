"""Native dataset and datamodule support for the Lightning backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _resolve_annotation_path(dataset_root: str, split: str) -> str:
    root = Path(dataset_root).expanduser()
    annotations_dir = root / "Annotations"
    split_name = str(split).strip().lower()
    mapping = {
        "train": "train_annotations.json",
        "val": "val_annotations.json",
        "test": "test_annotations.json",
    }
    try:
        return str(annotations_dir / mapping[split_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset split '{split}'.") from exc


class NativeDetectionDataset:
    """Minimal detection dataset backed by the repo's COCO-style payload loader."""

    def __init__(
        self,
        annotation_path: str,
        *,
        split: str,
        in_channels: int,
    ) -> None:
        payload = json.loads(Path(annotation_path).expanduser().read_text(encoding="utf-8"))
        self.payload = _normalize_coco_payload(payload)
        self.categories = list(self.payload.get("categories", []))
        self.samples = list(self.payload.get("samples", []))
        self.split = split
        self.in_channels = int(in_channels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        width = int(sample.get("width") or 1)
        height = int(sample.get("height") or 1)
        channels = max(self.in_channels, 1)
        image = torch.zeros((channels, max(height, 1), max(width, 1)), dtype=torch.float32)

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

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(sample.get("image_id", index))], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        if target["boxes"].numel() == 0:
            target["boxes"] = target["boxes"].view(0, 4)
        return image, target


def _normalize_coco_payload(payload: dict[str, Any]) -> dict[str, Any]:
    images = {int(item["id"]): item for item in payload.get("images", [])}
    categories = list(payload.get("categories", []))
    grouped_annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        grouped_annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    samples = []
    for image_id, image in images.items():
        annotations = []
        for annotation in grouped_annotations.get(image_id, []):
            bbox = list(annotation.get("bbox", [0, 0, 0, 0]))
            x_min = float(bbox[0]) if len(bbox) > 0 else 0.0
            y_min = float(bbox[1]) if len(bbox) > 1 else 0.0
            width = float(bbox[2]) if len(bbox) > 2 else 0.0
            height = float(bbox[3]) if len(bbox) > 3 else 0.0
            annotations.append(
                {
                    "category_id": int(annotation.get("category_id", 0)),
                    "area": float(annotation.get("area", width * height)),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_min + width,
                        "y_max": y_min + height,
                    },
                }
            )
        samples.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name", f"{image_id}.png"),
                "width": int(image.get("width", 1)),
                "height": int(image.get("height", 1)),
                "annotations": annotations,
            }
        )

    return {
        "categories": categories,
        "samples": samples,
    }


def detection_collate_fn(batch: Iterable[tuple[Any, Any]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass(slots=True)
class NativeDataConfig:
    dataset_root: str
    in_channels: int = 3
    batch_size: int = 2
    num_workers: int = 0


class NativeDetectionDataModule:
    """Lightning-compatible datamodule without hard dependency at import time."""

    def __init__(self, config: NativeDataConfig) -> None:
        self.config = config
        self.train_dataset: NativeDetectionDataset | None = None
        self.val_dataset: NativeDetectionDataset | None = None
        self.test_dataset: NativeDetectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = NativeDetectionDataset(
                _resolve_annotation_path(self.config.dataset_root, "train"),
                split="train",
                in_channels=self.config.in_channels,
            )
            self.val_dataset = NativeDetectionDataset(
                _resolve_annotation_path(self.config.dataset_root, "val"),
                split="val",
                in_channels=self.config.in_channels,
            )
        if stage in {None, "test", "predict"}:
            self.test_dataset = NativeDetectionDataset(
                _resolve_annotation_path(self.config.dataset_root, "test"),
                split="test",
                in_channels=self.config.in_channels,
            )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=detection_collate_fn,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=detection_collate_fn,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=detection_collate_fn,
        )
