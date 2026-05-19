import json
import os
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch


class _SmokeTensor:
    def __init__(self, data, *, shape=None):
        self.data = data
        self.shape = tuple(shape) if shape is not None else self._infer_shape(data)
        self.dtype = None
        self.device = None

    def numel(self):
        total = 1
        for dimension in self.shape:
            total *= int(dimension)
        return total

    def view(self, *shape):
        return _SmokeTensor(self.data, shape=shape)

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        value = self.tolist()
        while isinstance(value, list):
            value = value[0] if value else 0
        return value

    def tolist(self):
        return self.data

    @staticmethod
    def _infer_shape(value):
        if isinstance(value, (list, tuple)):
            if not value:
                return (0,)
            return (len(value), *_SmokeTensor._infer_shape(value[0]))
        return ()


class _SmokeDataLoader:
    def __init__(self, dataset, *, batch_size=1, shuffle=False, collate_fn=None, **_kwargs):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.collate_fn = collate_fn

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = list(reversed(indices))
        for start in range(0, len(indices), self.batch_size):
            batch = [self.dataset[index] for index in indices[start : start + self.batch_size]]
            yield self.collate_fn(batch) if self.collate_fn is not None else batch


class _SmokeParameter:
    requires_grad = True


class _SmokeDetector:
    def __init__(self):
        self.loss_batches = 0
        self.prediction_batches = 0

    def parameters(self):
        return iter((_SmokeParameter(),))

    def forward_loss(self, images, targets):
        self.loss_batches += 1
        return {"loss_total": 1.0}

    def predict(self, images):
        self.prediction_batches += 1
        return [
            {
                "boxes": [[1.0, 1.0, 5.0, 5.0]],
                "scores": [0.95],
                "labels": [1],
            }
            for _image in images
        ]


def _fake_runtime_modules():
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_functional = types.ModuleType("torch.nn.functional")
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return []

    class Generator:
        def manual_seed(self, _seed):
            return self

    fake_nn.Module = Module
    fake_nn.ModuleList = list
    fake_nn.Sequential = lambda *layers: Module()
    fake_nn.Conv2d = lambda *args, **kwargs: Module()
    fake_nn.ConvTranspose2d = lambda *args, **kwargs: Module()
    fake_nn.Identity = lambda *args, **kwargs: Module()
    fake_nn.ReLU = lambda *args, **kwargs: Module()
    fake_nn.Linear = lambda *args, **kwargs: Module()
    fake_nn.Embedding = lambda *args, **kwargs: Module()

    fake_torch.nn = fake_nn
    fake_torch.float32 = "float32"
    fake_torch.int64 = "int64"
    fake_torch.long = "int64"
    fake_torch.Generator = Generator
    fake_torch.zeros = lambda shape, **_kwargs: _SmokeTensor(0.0, shape=tuple(shape))
    fake_torch.as_tensor = lambda value, **_kwargs: _SmokeTensor(value)
    fake_torch.tensor = lambda value, **_kwargs: _SmokeTensor(value)
    fake_data.DataLoader = _SmokeDataLoader
    fake_utils.data = fake_data
    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_functional,
        "torch.utils": fake_utils,
        "torch.utils.data": fake_data,
    }


def _fake_lightning_modules(events, checkpoint_path: Path):
    class LightningModule:
        def __init__(self, *args, **kwargs):
            self.logged = []

        def log(self, name, value, **kwargs):
            self.logged.append((name, value, kwargs))

    class ModelCheckpoint:
        def __init__(self, *, dirpath, **_kwargs):
            self.dirpath = str(dirpath)
            self.last_model_path = ""
            self.best_model_path = None

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.checkpoint_callback = kwargs["callbacks"][0]
            events.append(("trainer_init", kwargs["accelerator"], kwargs["devices"]))

        def fit(self, module, datamodule=None):
            events.append(("fit", None, None))
            datamodule.setup("fit")
            batch = next(iter(datamodule.train_dataloader()))
            module.training_step(batch, 0)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("smoke checkpoint", encoding="utf-8")
            self.checkpoint_callback.last_model_path = str(checkpoint_path)

        def test(self, module, datamodule=None, ckpt_path=None):
            events.append(("test", ckpt_path, None))
            datamodule.setup("test")
            batch = next(iter(datamodule.test_dataloader()))
            module.test_step(batch, 0)

    return types.SimpleNamespace(LightningModule=LightningModule, Trainer=Trainer), ModelCheckpoint


def _write_png(path: Path, *, width: int = 8, height: int = 8) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw_rows = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_rows))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _write_coco_split(
    path: Path,
    *,
    image_name: str,
    width: int = 8,
    height: int = 8,
    malformed: bool = False,
) -> None:
    if malformed:
        payload = {
            "images": {"id": 1, "file_name": image_name},
            "annotations": [],
            "categories": [{"id": 1, "name": "wake"}],
        }
    else:
        payload = {
            "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [1, 1, 4, 4],
                    "area": 16,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "wake"}],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fixture_dataset(
    root: Path,
    *,
    image_size: int = 8,
    malformed_train: bool = False,
) -> None:
    image_name = "smoke.png"
    (root / "imgs").mkdir(parents=True, exist_ok=True)
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    _write_png(root / "imgs" / image_name, width=image_size, height=image_size)
    _write_coco_split(
        root / "Annotations" / "train_annotations.json",
        image_name=image_name,
        width=image_size,
        height=image_size,
        malformed=malformed_train,
    )
    for split in ("val", "test"):
        _write_coco_split(
            root / "Annotations" / f"{split}_annotations.json",
            image_name=image_name,
            width=image_size,
            height=image_size,
        )


def _smoke_project_config(dataset_root: Path, workdir: Path) -> dict:
    return {
        "stages": ["build", "train", "test", "infer"],
        "workdir": str(workdir),
        "detector": {
            "name": "retinanet",
            "num_classes": 1,
            "backbone": "resnet18",
            "pretrained": False,
        },
        "dataset": {
            "format": "coco",
            "root": str(dataset_root),
            "train": "Annotations/train_annotations.json",
            "val": "Annotations/val_annotations.json",
            "test": "Annotations/test_annotations.json",
            "data_prefix": "imgs/",
            "classes": ["wake"],
            "in_channels": 3,
        },
        "runtime": {
            "accelerator": "cpu",
            "devices": 1,
            "batch_size": 1,
            "max_epochs": 1,
            "num_workers": 0,
        },
        "optimizer": {"name": "SGD", "learning_rate": 0.001},
        "export": {"formats": ["json"]},
    }


def _clear_native_runtime_modules():
    module_names = (
        ("simpledet.native.runtime", "runtime"),
        ("simpledet.native.engine", "engine"),
        ("simpledet.native.modeling", "modeling"),
        ("simpledet.native.roi", "roi"),
        ("simpledet.native.assemblers", "assemblers"),
        ("simpledet.native.assignment", "assignment"),
        ("simpledet.native.backbones", "backbones"),
        ("simpledet.native.dense_ops", "dense_ops"),
        ("simpledet.native.heads", "heads"),
        ("simpledet.native.losses", "losses"),
        ("simpledet.native.necks", "necks"),
        ("simpledet.native.transformer_ops", "transformer_ops"),
    )
    native_package = sys.modules.get("simpledet.native")
    for module_name, attribute in module_names:
        sys.modules.pop(module_name, None)
        if native_package is not None:
            vars(native_package).pop(attribute, None)


class CpuSmokeWorkflowTests(unittest.TestCase):
    def tearDown(self):
        _clear_native_runtime_modules()

    def test_project_config_cpu_smoke_runs_build_train_test_and_infer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            workdir = Path(tmpdir) / "runs"
            checkpoint_path = workdir / "checkpoints" / "last.ckpt"
            _write_fixture_dataset(root)
            events = []
            detector = _SmokeDetector()

            with patch.dict(sys.modules, _fake_runtime_modules()):
                _clear_native_runtime_modules()
                import simpledet.native.engine as engine
                from simpledet.api import run_project

                fake_lightning = _fake_lightning_modules(events, checkpoint_path)
                with patch.object(engine, "build_native_model", return_value=detector), patch.object(
                    engine,
                    "_load_lightning",
                    return_value=fake_lightning,
                ):
                    result = run_project(_smoke_project_config(root, workdir))

            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_exists = manifest_path.exists()
            metrics_exists = (workdir / "native-metrics.json").exists()

        self.assertEqual(result["stages"], ["build", "train", "test", "infer"])
        self.assertEqual(result["train"]["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(result["test"]["predictions"][0]["image_id"], 1)
        self.assertEqual(result["infer"]["predictions"][0]["boxes"], [[1.0, 1.0, 5.0, 5.0]])
        self.assertEqual(manifest["detector"]["architecture"], "retinanet")
        self.assertEqual(manifest["dataset"]["root"], str(root))
        self.assertTrue(manifest_exists)
        self.assertTrue(metrics_exists)
        self.assertIn(("trainer_init", "cpu", 1), events)
        self.assertEqual([event[0] for event in events], ["trainer_init", "fit", "trainer_init", "test", "trainer_init", "test"])
        self.assertEqual(detector.loss_batches, 1)
        self.assertEqual(detector.prediction_batches, 2)

    def test_malformed_fixture_fails_validation_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            workdir = Path(tmpdir) / "runs"
            checkpoint_path = workdir / "checkpoints" / "last.ckpt"
            _write_fixture_dataset(root, malformed_train=True)
            events = []

            with patch.dict(sys.modules, _fake_runtime_modules()):
                _clear_native_runtime_modules()
                import simpledet.native.engine as engine
                from simpledet.api import run_project

                fake_lightning = _fake_lightning_modules(events, checkpoint_path)
                with patch.object(engine, "build_native_model") as build_model, patch.object(
                    engine,
                    "_load_lightning",
                    return_value=fake_lightning,
                ):
                    with self.assertRaisesRegex(ValueError, "Invalid COCO schema"):
                        run_project(_smoke_project_config(root, workdir))

        build_model.assert_not_called()
        self.assertEqual(events, [])
        self.assertFalse(workdir.exists())


class RealCpuSmokeWorkflowTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SIMPLEDET_RUN_REAL_CPU_SMOKE") == "1",
        "set SIMPLEDET_RUN_REAL_CPU_SMOKE=1 to run the optional real CPU smoke workflow",
    )
    def test_real_cpu_project_config_runs_build_train_test_and_infer(self):
        missing = _missing_real_cpu_runtime()
        if missing:
            self.fail(
                "Optional CPU runtime dependency missing for make test-cpu-smoke: "
                + ", ".join(missing)
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            workdir = Path(tmpdir) / "runs"
            _write_fixture_dataset(root, image_size=32)

            from simpledet.api import run_project

            result = run_project(_smoke_project_config(root, workdir))
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

            self.assertEqual(result["stages"], ["build", "train", "test", "infer"])
            self.assertEqual(result["train"]["backend"], "native_lightning")
            self.assertTrue(Path(result["train"]["checkpoint_path"]).exists())
            self.assertTrue(result["test"]["predictions"])
            self.assertTrue(result["infer"]["predictions"])
            self.assertEqual(manifest["detector"]["architecture"], "retinanet")


def _missing_real_cpu_runtime() -> list[str]:
    import importlib.util

    missing = []
    for module_name in ("torch", "torchvision"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    has_lightning = (
        importlib.util.find_spec("lightning") is not None
        or importlib.util.find_spec("pytorch_lightning") is not None
    )
    if not has_lightning:
        missing.append("lightning or pytorch_lightning")
    return missing


if __name__ == "__main__":
    unittest.main()
