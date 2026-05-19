import json
import struct
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _write_dummy_png(path: Path, width: int = 10, height: int = 10) -> None:
    png_signature = b"\x89PNG\r\n\x1a\n"
    ihdr = b"IHDR"
    header = (
        png_signature
        + struct.pack(">I", 13)
        + ihdr
        + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        + struct.pack(">I", 0)
    )
    path.write_bytes(header)


def _write_coco_split(root: Path, annotation_path: Path, *, image_id: int, file_name: str) -> None:
    _write_dummy_png(root / "images" / file_name)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": image_id, "file_name": file_name, "width": 10, "height": 8}
                ],
                "annotations": [
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [1, 2, 3, 4],
                        "area": 12,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "wake"}],
            }
        ),
        encoding="utf-8",
    )


def _write_empty_coco(root: Path, annotation_path: Path) -> None:
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps({"images": [], "annotations": [], "categories": []}),
        encoding="utf-8",
    )


class _Tensor:
    def __init__(self, data=None, *, shape=None, dtype=None):
        self.data = data
        self._shape = tuple(shape) if shape is not None else self._infer_shape(data)
        self.dtype = dtype

    @property
    def shape(self):
        return self._shape

    def to(self, dtype=None, **_kwargs):
        return _Tensor(self.data, shape=self.shape, dtype=dtype or self.dtype)

    def view(self, *shape):
        return _Tensor(self.data, shape=shape, dtype=self.dtype)

    def numel(self):
        count = 1
        for dimension in self.shape:
            count *= int(dimension)
        return count

    def tolist(self):
        return self.data

    def __len__(self):
        return int(self.shape[0]) if self.shape else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            length = len(range(*index.indices(int(self.shape[0]))))
            return _Tensor(self.data, shape=(length, *self.shape[1:]), dtype=self.dtype)
        return _Tensor(self.data, shape=self.shape[1:], dtype=self.dtype)

    def _infer_shape(self, value):
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return (len(value), len(value[0]))
            return (len(value),)
        return ()


class _FakeDataLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.batch_size = int(kwargs["batch_size"])
        self.shuffle = bool(kwargs["shuffle"])
        self.num_workers = int(kwargs["num_workers"])
        self.collate_fn = kwargs["collate_fn"]
        self.generator = kwargs.get("generator")

    def __iter__(self):
        indices = list(range(len(self.dataset)))[: self.batch_size]
        yield self.collate_fn([self.dataset[index] for index in indices])


class _Generator:
    def __init__(self):
        self.seed = None

    def manual_seed(self, seed):
        self.seed = int(seed)
        return self


def _fake_runtime_modules():
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return []

        def __call__(self, *args, **kwargs):
            forward = getattr(self, "forward", None)
            if callable(forward):
                return forward(*args, **kwargs)
            raise TypeError("forward not implemented")

    fake_nn.Module = Module
    fake_nn.ModuleList = list
    fake_nn.Sequential = lambda *layers: Module()
    fake_nn.Conv2d = lambda *args, **kwargs: Module()
    fake_nn.ConvTranspose2d = lambda *args, **kwargs: Module()
    fake_nn.Identity = lambda *args, **kwargs: Module()
    fake_nn.ReLU = lambda *args, **kwargs: Module()
    fake_nn.Linear = lambda *args, **kwargs: Module()
    fake_torch.nn = fake_nn
    fake_torch.float32 = "float32"
    fake_torch.int64 = "int64"
    fake_torch.long = "int64"
    fake_torch.as_tensor = lambda value, dtype=None: _Tensor(value, dtype=dtype)
    fake_torch.tensor = lambda value, dtype=None: _Tensor(value, dtype=dtype)
    fake_torch.zeros = lambda shape, dtype=None: _Tensor(shape=shape, dtype=dtype)
    fake_torch.cat = lambda tensors, dim=0: _Tensor(
        shape=(
            sum(int(tensor.shape[dim]) for tensor in tensors),
            *tensors[0].shape[dim + 1 :],
        ),
        dtype=tensors[0].dtype,
    )
    fake_torch.Generator = _Generator

    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.DataLoader = _FakeDataLoader
    fake_utils.data = fake_data
    fake_torch.utils = fake_utils

    fake_torchvision = types.ModuleType("torchvision")
    fake_io = types.ModuleType("torchvision.io")
    fake_io.read_image = lambda _path: _Tensor(shape=(3, 8, 10), dtype="uint8")
    fake_torchvision.io = fake_io

    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_f,
        "torch.utils": fake_utils,
        "torch.utils.data": fake_data,
        "torchvision": fake_torchvision,
        "torchvision.io": fake_io,
    }


class NativeDataModuleTests(unittest.TestCase):
    def test_coco_fixture_produces_train_batch_with_metadata_and_transforms(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            _write_coco_split(
                root,
                root / "Annotations" / "train_annotations.json",
                image_id=11,
                file_name="train.png",
            )
            _write_coco_split(
                root,
                root / "Annotations" / "val_annotations.json",
                image_id=12,
                file_name="val.png",
            )

            def _mark_transform(image, target):
                target["metadata"]["transformed"] = True
                return image, target

            with patch.dict(sys.modules, _fake_runtime_modules()):
                from simpledet.native.data import NativeDataConfig, NativeDetectionDataModule

                module = NativeDetectionDataModule(
                    NativeDataConfig(
                        dataset_root=str(root),
                        batch_size=1,
                        seed=123,
                        train_transforms=_mark_transform,
                    )
                )
                loader = module.train_dataloader()
                images, targets = next(iter(loader))

            self.assertTrue(loader.shuffle)
            self.assertEqual(loader.generator.seed, 123)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].shape, (3, 8, 10))
            self.assertEqual(targets[0]["boxes"].shape, (1, 4))
            self.assertEqual(targets[0]["labels"].tolist(), [1])
            self.assertEqual(targets[0]["metadata"]["image_id"], 11)
            self.assertEqual(targets[0]["metadata"]["split"], "train")
            self.assertTrue(targets[0]["metadata"]["transformed"])

    def test_split_resolution_selects_stage_specific_coco_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            _write_coco_split(
                root,
                root / "annotations" / "instances_train.json",
                image_id=21,
                file_name="train.png",
            )
            _write_coco_split(
                root,
                root / "annotations" / "instances_val.json",
                image_id=22,
                file_name="val.png",
            )
            _write_coco_split(
                root,
                root / "annotations" / "instances_test.json",
                image_id=23,
                file_name="test.png",
            )

            with patch.dict(sys.modules, _fake_runtime_modules()):
                from simpledet.native.data import NativeDataConfig, NativeDetectionDataModule

                module = NativeDetectionDataModule(NativeDataConfig(dataset_root=str(root)))
                module.setup(None)

            self.assertEqual(module.train_dataset.samples[0]["image_id"], 21)
            self.assertEqual(module.val_dataset.samples[0]["image_id"], 22)
            self.assertEqual(module.test_dataset.samples[0]["image_id"], 23)

    def test_empty_dataset_fails_fast_with_data_validation_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            _write_empty_coco(root, root / "Annotations" / "train_annotations.json")

            with patch.dict(sys.modules, _fake_runtime_modules()):
                from simpledet.native.data import (
                    NativeDataConfig,
                    NativeDataValidationError,
                    NativeDetectionDataModule,
                )

                module = NativeDetectionDataModule(NativeDataConfig(dataset_root=str(root)))
                with self.assertRaisesRegex(NativeDataValidationError, "train split is empty"):
                    module.setup("fit")


if __name__ == "__main__":
    unittest.main()
