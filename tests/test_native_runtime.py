import json
import math
from collections import OrderedDict
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simpledet.suite import build_detector


def _write_dummy_png(path: Path, width: int = 8, height: int = 8) -> None:
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


class _LossScalar:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __add__(self, other):
        return _LossScalar(self.value + float(other))

    def __radd__(self, other):
        return _LossScalar(float(other) + self.value)

    def __float__(self):
        return self.value


class _FakeParameter:
    def __init__(self, *, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad


class _TinyRetinaNet:
    def __init__(self, *, params=None) -> None:
        self._params = list(params) if params is not None else [_FakeParameter()]
        self.loss_batches = []
        self.prediction_batches = []

    def parameters(self):
        return iter(self._params)

    def forward_loss(self, images, targets):
        self.loss_batches.append((images, targets))
        return {"loss_cls": _LossScalar(0.5), "loss_bbox": _LossScalar(0.25)}

    def predict(self, images):
        self.prediction_batches.append(images)
        return [
            {
                "boxes": [[0.0, 0.0, 4.0, 4.0]],
                "scores": [0.9],
                "labels": [1],
            }
            for _ in images
        ]


class NativeRuntimeTests(unittest.TestCase):
    def _fake_runtime_modules(self):
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

        class _Optimizer:
            def __init__(self, params, **kwargs):
                self.params = list(params)
                self.kwargs = kwargs

        class _Scheduler:
            def __init__(self, optimizer, **kwargs):
                self.optimizer = optimizer
                self.kwargs = kwargs

        fake_nn.Module = Module
        fake_nn.ModuleList = list
        fake_nn.Sequential = lambda *layers: Module()
        fake_nn.Conv2d = lambda *args, **kwargs: Module()
        fake_nn.ConvTranspose2d = lambda *args, **kwargs: Module()
        fake_nn.Identity = lambda *args, **kwargs: Module()
        fake_nn.ReLU = lambda *args, **kwargs: Module()
        fake_torch.nn = fake_nn
        fake_torch.tensor = lambda *args, **kwargs: 0
        fake_torch.zeros_like = lambda value: value
        fake_torch.zeros = lambda *args, **kwargs: 0
        fake_torch.finfo = lambda dtype: types.SimpleNamespace(eps=1e-12)
        fake_torch.optim = types.SimpleNamespace(
            SGD=_Optimizer,
            Adam=_Optimizer,
            AdamW=_Optimizer,
            lr_scheduler=types.SimpleNamespace(
                StepLR=_Scheduler,
                ExponentialLR=_Scheduler,
                CosineAnnealingLR=_Scheduler,
            ),
        )
        return {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_f,
        }

    def _tensor_runtime_modules(self):
        fake_torch = types.ModuleType("torch")
        fake_nn = types.ModuleType("torch.nn")
        fake_f = types.ModuleType("torch.nn.functional")

        class Tensor:
            def __init__(self, data, dtype=None):
                self._array = np.array(data, dtype=dtype)

            @property
            def shape(self):
                return self._array.shape

            @property
            def dtype(self):
                return self._array.dtype

            @property
            def device(self):
                return None

            def unsqueeze(self, dim):
                return Tensor(np.expand_dims(self._array, axis=dim))

            def mean(self, dim=None, axis=None, keepdim=False):
                axes = axis if axis is not None else dim
                return Tensor(self._array.mean(axis=axes, keepdims=keepdim))

            def clone(self):
                return Tensor(self._array.copy())

            def to(self, dtype=None, device=None):
                if dtype is None or dtype == self._array.dtype:
                    return Tensor(self._array.copy())
                return Tensor(self._array.astype(dtype))

            def reshape(self, *shape):
                return Tensor(self._array.reshape(*shape))

            def reshape_as(self, other):
                return Tensor(self._array.reshape(other.shape))

            def squeeze(self, axis=None):
                if axis is None:
                    return Tensor(np.squeeze(self._array))
                return Tensor(np.squeeze(self._array, axis=axis))

            def index_select(self, dim, index_tensor):
                indices = self._unwrap(index_tensor).astype(int)
                return Tensor(np.take(self._array, indices, axis=dim))

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self._array.tolist()

            def new_zeros(self, shape=()):
                return Tensor(np.zeros(shape, dtype=self._array.dtype))

            def __len__(self):
                return len(self._array)

            def __iter__(self):
                for value in self._array:
                    yield Tensor(value) if isinstance(value, np.ndarray) else value

            def __array__(self, dtype=None):
                return np.asarray(self._array, dtype=dtype)

            def __getitem__(self, index):
                index = self._unwrap_index(index)
                value = self._array[index]
                return Tensor(value) if isinstance(value, np.ndarray) else Tensor(np.array(value))

            def __setitem__(self, index, value):
                self._array[self._unwrap_index(index)] = self._unwrap(value)

            def __add__(self, other):
                return Tensor(self._array + self._unwrap(other))

            def __radd__(self, other):
                return Tensor(self._unwrap(other) + self._array)

            def __sub__(self, other):
                return Tensor(self._array - self._unwrap(other))

            def __rsub__(self, other):
                return Tensor(self._unwrap(other) - self._array)

            def __mul__(self, other):
                return Tensor(self._array * self._unwrap(other))

            def __rmul__(self, other):
                return Tensor(self._unwrap(other) * self._array)

            def _unwrap(self, value):
                if isinstance(value, Tensor):
                    return value._array
                return np.asarray(value)

            def _unwrap_index(self, index):
                if isinstance(index, tuple):
                    return tuple(self._unwrap_index(item) for item in index)
                if isinstance(index, Tensor):
                    return index._array.astype(int)
                return index

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

        class Sequential(Module):
            def __init__(self, *layers):
                super().__init__()
                self.layers = list(layers)

            def forward(self, x):
                value = x
                for layer in self.layers:
                    value = layer(value)
                return value

        class Linear(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = int(in_features)
                self.out_features = int(out_features)
                self.weight = Tensor(np.zeros((self.out_features, self.in_features), dtype=np.float32))
                self.bias = Tensor(np.zeros(self.out_features, dtype=np.float32))

            def forward(self, x):
                array = np.asarray(x)
                if array.ndim > 2:
                    array = array.reshape(array.shape[0], -1)
                output = array @ self.weight._array.T + self.bias._array
                return Tensor(output)

        class ReLU(Module):
            def forward(self, x):
                return Tensor(np.maximum(np.asarray(x), 0.0))

        fake_nn.Module = Module
        fake_nn.ModuleList = list
        fake_nn.Sequential = Sequential
        fake_nn.Linear = Linear
        fake_nn.ReLU = ReLU
        fake_torch.nn = fake_nn
        fake_torch.tensor = lambda value, **kwargs: Tensor(value, dtype=kwargs.get("dtype"))
        fake_torch.zeros = lambda *args, **kwargs: Tensor(np.zeros(*args, dtype=kwargs.get("dtype", np.float32)))
        fake_torch.zeros_like = lambda value: Tensor(np.zeros_like(np.asarray(value)))
        fake_torch.sigmoid = lambda value: Tensor(1.0 / (1.0 + np.exp(-np.asarray(value))))
        fake_torch.softmax = lambda value, dim=-1: Tensor(
            np.exp(np.asarray(value) - np.max(np.asarray(value), axis=dim, keepdims=True))
            / np.sum(np.exp(np.asarray(value) - np.max(np.asarray(value), axis=dim, keepdims=True)), axis=dim, keepdims=True)
        )
        fake_torch.tanh = lambda value: Tensor(np.tanh(np.asarray(value)))
        fake_torch.maximum = lambda left, right: Tensor(np.maximum(np.asarray(left), np.asarray(right)))
        fake_torch.float32 = np.float32
        fake_torch.long = np.int64
        fake_torch.finfo = lambda dtype: types.SimpleNamespace(eps=np.finfo(np.float32).eps)
        return {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_f,
        }

    def _fake_lightning_modules(self):
        class _FakeLightningModule:
            def __init__(self, *args, **kwargs):
                self.logged = []

            def log(self, name, value, **kwargs):
                self.logged.append((name, value, kwargs))

        return types.SimpleNamespace(LightningModule=_FakeLightningModule), object

    def _import_engine_with_fake_runtime(self):
        import importlib

        return importlib.reload(importlib.import_module("simpledet.native.engine"))

    def _build_tiny_lightning_module(self, *, params=None, optimizer="sgd", scheduler=None):
        engine = self._import_engine_with_fake_runtime()
        model = _TinyRetinaNet(params=params)
        fake_pl, fake_checkpoint = self._fake_lightning_modules()
        config = engine.NativeEngineConfig(
            architecture="retinanet",
            num_classes=2,
            learning_rate=0.01,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=3,
        )
        with patch.object(engine, "build_native_model", return_value=model), patch.object(
            engine,
            "_load_lightning",
            return_value=(fake_pl, fake_checkpoint),
        ):
            module, payload = engine.NativeDetectionLightningModule.build(config)
        return module, payload, model

    def test_lightning_training_step_runs_tiny_retinanet_fixture(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            module, payload, model = self._build_tiny_lightning_module()

            loss = module.training_step((["image"], [{"image_id": [7]}]), 0)

        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(float(loss), 0.75)
        self.assertEqual(model.loss_batches, [(["image"], [{"image_id": [7]}])])
        self.assertIs(payload.model, model)
        self.assertIn("train_loss", [name for name, _, _ in module.logged])
        self.assertIn("train_loss_cls", [name for name, _, _ in module.logged])

    def test_lightning_validation_and_test_steps_log_metrics_and_predictions(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            module, payload, _ = self._build_tiny_lightning_module()

            validation = module.validation_step((["image"], [{"image_id": [8]}]), 0)
            test = module.test_step((["image"], [{"image_id": [8]}]), 0)

        self.assertEqual(validation["detection_count"], 1.0)
        self.assertTrue(math.isfinite(float(validation["loss"])))
        self.assertEqual(test["predictions"][0]["image_id"], 8)
        self.assertEqual(test["predictions"][0]["boxes"], [[0.0, 0.0, 4.0, 4.0]])
        self.assertEqual(payload.latest_predictions, test["predictions"])
        self.assertIn("val_detection_count", [name for name, _, _ in module.logged])
        self.assertIn("test_detection_count", [name for name, _, _ in module.logged])

    def test_lightning_optimizer_uses_only_trainable_parameters_and_scheduler(self):
        trainable = _FakeParameter(requires_grad=True)
        frozen = _FakeParameter(requires_grad=False)
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            module, _, _ = self._build_tiny_lightning_module(
                params=[trainable, frozen],
                optimizer="adamw",
                scheduler="step",
            )

            configured = module.configure_optimizers()

        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]
        self.assertEqual(optimizer.params, [trainable])
        self.assertEqual(optimizer.kwargs["lr"], 0.01)
        self.assertIs(scheduler.optimizer, optimizer)
        self.assertEqual(scheduler.kwargs["step_size"], 1)

    def test_lightning_optimizer_rejects_model_with_no_trainable_parameters(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            module, _, _ = self._build_tiny_lightning_module(
                params=[_FakeParameter(requires_grad=False)]
            )

            with self.assertRaisesRegex(RuntimeError, "no trainable parameters"):
                module.configure_optimizers()

    def test_lightning_checkpoint_metadata_round_trips(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            module, payload, _ = self._build_tiny_lightning_module(scheduler="cosine")
            checkpoint = {}

            module.on_save_checkpoint(checkpoint)
            module.on_load_checkpoint(checkpoint)

        metadata = checkpoint["simpledet"]
        self.assertEqual(metadata["backend"], "native_lightning")
        self.assertEqual(metadata["format_version"], 1)
        self.assertEqual(metadata["engine_config"]["architecture"], "retinanet")
        self.assertEqual(metadata["engine_config"]["scheduler"], "cosine")
        self.assertEqual(payload.loaded_checkpoint_metadata, metadata)

    def _write_dataset(self, root: Path) -> None:
        (root / "Annotations").mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(parents=True, exist_ok=True)
        _write_dummy_png(root / "images" / "a.png")
        payload = {
            "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
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
        for name in ("train_annotations.json", "val_annotations.json", "test_annotations.json"):
            (root / "Annotations" / name).write_text(json.dumps(payload), encoding="utf-8")

    def _write_empty_train_dataset(self, root: Path) -> None:
        (root / "Annotations").mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(parents=True, exist_ok=True)
        payload = {"images": [], "annotations": [], "categories": []}
        (root / "Annotations" / "train_annotations.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_native_training_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "runs"
            self._write_dataset(root)
            spec = build_detector("retinanet", num_classes=1, encoder="resnet18.a1_in1k")

            with patch.dict(sys.modules, self._tensor_runtime_modules()):
                from simpledet.native.runtime import NativeProjectConfig, run_native_training

                with patch(
                "simpledet.native.engine.build_native_model",
                return_value=types.SimpleNamespace(parameters=lambda: []),
                ), patch("simpledet.native.engine._load_lightning") as load_lightning:
                    class _FakeModule:
                        def __init__(self, *args, **kwargs):
                            pass

                        def log(self, *args, **kwargs):
                            return None

                    class _FakeTrainer:
                        checkpoint_callback = types.SimpleNamespace(
                            last_model_path=str(output / "checkpoints" / "epoch_001.ckpt"),
                            best_model_path=None,
                        )

                        def fit(self, module, datamodule=None):
                            datamodule.setup("fit")
                            return None

                    def _build_fake_trainer(**kwargs):
                        return _FakeTrainer()

                    def _build_fake_checkpoint(**kwargs):
                        return object()

                    fake_pl = types.SimpleNamespace(LightningModule=_FakeModule, Trainer=_build_fake_trainer)
                    fake_checkpoint = _build_fake_checkpoint
                    load_lightning.return_value = (fake_pl, fake_checkpoint)

                    result = run_native_training(
                        NativeProjectConfig(
                            dataset_root=str(root),
                            categories=("wake",),
                            detector_spec=spec,
                            output_dir=str(output),
                        )
                    )

            self.assertEqual(result["backend"], "native_lightning")
            self.assertEqual(result["checkpoint_path"], str(output / "checkpoints" / "epoch_001.ckpt"))
            self.assertTrue((output / "native-manifest.json").exists())

    def test_native_training_empty_dataset_fails_before_trainer_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "runs"
            self._write_empty_train_dataset(root)
            spec = build_detector("retinanet", num_classes=1, encoder="resnet18.a1_in1k")

            with patch.dict(sys.modules, self._fake_runtime_modules()):
                from simpledet.native.data import NativeDataValidationError
                from simpledet.native.runtime import NativeProjectConfig, run_native_training

                build_module_patch = patch(
                    "simpledet.native.runtime.NativeDetectionLightningModule.build"
                )
                build_trainer_patch = patch("simpledet.native.runtime.build_native_trainer")
                build_module = build_module_patch.start()
                build_trainer = build_trainer_patch.start()
                self.addCleanup(build_module_patch.stop)
                self.addCleanup(build_trainer_patch.stop)

                with self.assertRaisesRegex(NativeDataValidationError, "train split is empty"):
                    run_native_training(
                        NativeProjectConfig(
                            dataset_root=str(root),
                            categories=("wake",),
                            detector_spec=spec,
                            output_dir=str(output),
                        )
                    )

            build_module.assert_not_called()
            build_trainer.assert_not_called()

    def test_native_evaluation_returns_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "runs"
            self._write_dataset(root)
            checkpoint_path = output / "checkpoints" / "epoch_002.ckpt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            spec = build_detector("retinanet", num_classes=1, encoder="resnet18.a1_in1k")

            with patch.dict(sys.modules, self._tensor_runtime_modules()):
                from simpledet.native.runtime import NativeProjectConfig, run_native_evaluation

                import importlib

                roi_module = importlib.reload(importlib.import_module("simpledet.native.roi"))

                class _TensorBackbone:
                    def __call__(self, images):
                        return roi_module.torch.tensor(np.zeros((1, 8, 2, 2), dtype=np.float32))

                class _TensorNeck:
                    def __call__(self, features):
                        return OrderedDict([("0", features)])

                class _TensorRoIPool:
                    def __call__(self, features, proposals, image_shapes):
                        channels = int(features["0"].shape[1])
                        count = int(proposals[0].shape[0])
                        return roi_module.torch.tensor(np.zeros((count, channels, 2, 2), dtype=np.float32))

                roi_model = roi_module.NativeRoIModel(
                    backbone=roi_module.NativeRoIBackbone(
                        backbone=_TensorBackbone(),
                        neck=_TensorNeck(),
                        core_spec=roi_module.RoICoreSpec.from_num_levels(1),
                    ),
                    box_roi_pool=_TensorRoIPool(),
                    num_classes=2,
                    in_channels=8,
                    core_spec=roi_module.RoICoreSpec.from_num_levels(1),
                )

                with patch(
                "simpledet.native.engine.build_native_model",
                return_value=roi_model,
                ), patch("simpledet.native.engine._load_lightning") as load_lightning:
                    class _FakeModule:
                        def __init__(self, *args, **kwargs):
                            pass

                        def log(self, *args, **kwargs):
                            return None

                    class _FakeTrainer:
                        def test(self, module, datamodule=None, ckpt_path=None):
                            datamodule.setup("test")
                            assert ckpt_path == str(output / "checkpoints" / "epoch_002.ckpt")
                            image = roi_module.torch.tensor(np.zeros((3, 8, 8), dtype=np.float32))
                            target = {"image_id": [1], "boxes": [], "labels": []}
                            module.test_step(([image], [target]), 0)
                            return None

                    def _build_fake_trainer(**kwargs):
                        return _FakeTrainer()

                    def _build_fake_checkpoint(**kwargs):
                        return object()

                    fake_pl = types.SimpleNamespace(LightningModule=_FakeModule, Trainer=_build_fake_trainer)
                    fake_checkpoint = _build_fake_checkpoint
                    load_lightning.return_value = (fake_pl, fake_checkpoint)

                    result = run_native_evaluation(
                        NativeProjectConfig(
                            dataset_root=str(root),
                            categories=("wake",),
                            detector_spec=spec,
                            output_dir=str(output),
                            checkpoint_path=str(checkpoint_path),
                        )
                    )

            self.assertEqual(result["backend"], "native_lightning")
            self.assertEqual(result["checkpoint_path"], str(checkpoint_path))
            self.assertTrue((output / "native-manifest.json").exists())
            self.assertTrue((output / "native-metrics.json").exists())
            self.assertEqual(result["metrics_path"], str(output / "native-metrics.json"))
            self.assertIn("summary", result["metrics"])
            self.assertIn("per_class", result["metrics"])
            self.assertIn("recall", result["metrics"])
            self.assertIn("prediction_export", result["metrics"])
            prediction = result["predictions"][0]
            self.assertEqual(prediction["image_id"], 1)
            self.assertIn("boxes", prediction)
            self.assertIn("scores", prediction)
            self.assertIn("labels", prediction)
            self.assertEqual(len(prediction["boxes"]), 4)
            self.assertEqual(len(prediction["scores"]), 4)
            self.assertEqual(len(prediction["labels"]), 4)
            manifest = json.loads((output / "native-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["metrics_path"], str(output / "native-metrics.json"))

    def test_native_evaluation_requires_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "runs"
            self._write_dataset(root)
            spec = build_detector("retinanet", num_classes=1, encoder="resnet18.a1_in1k")

            with patch.dict(sys.modules, self._fake_runtime_modules()):
                from simpledet.native.runtime import NativeProjectConfig, run_native_evaluation

                with patch(
                    "simpledet.native.engine.build_native_model",
                    return_value=types.SimpleNamespace(parameters=lambda: []),
                ), patch("simpledet.native.engine._load_lightning") as load_lightning:
                    class _FakeModule:
                        def __init__(self, *args, **kwargs):
                            pass

                        def log(self, *args, **kwargs):
                            return None

                    class _FakeTrainer:
                        def test(self, module, datamodule=None, ckpt_path=None):
                            raise AssertionError("trainer.test should not run when the checkpoint is missing")

                    def _build_fake_trainer(**kwargs):
                        return _FakeTrainer()

                    def _build_fake_checkpoint(**kwargs):
                        return object()

                    fake_pl = types.SimpleNamespace(
                        LightningModule=_FakeModule,
                        Trainer=_build_fake_trainer,
                    )
                    fake_checkpoint = _build_fake_checkpoint
                    load_lightning.return_value = (fake_pl, fake_checkpoint)

                    with self.assertRaises(FileNotFoundError):
                        run_native_evaluation(
                            NativeProjectConfig(
                                dataset_root=str(root),
                                categories=("wake",),
                                detector_spec=spec,
                                output_dir=str(output),
                                checkpoint_path=str(output / "checkpoints" / "missing.ckpt"),
                            )
                        )

    def test_native_runtime_passes_detector_spec_into_model_builder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "runs"
            self._write_dataset(root)
            spec = build_detector("retinanet", num_classes=1, encoder="resnet18.a1_in1k")

            with patch.dict(sys.modules, self._fake_runtime_modules()):
                from simpledet.native.runtime import NativeProjectConfig, run_native_training

                with patch(
                "simpledet.native.engine.build_native_model",
                return_value=types.SimpleNamespace(parameters=lambda: []),
                ) as build_model, patch("simpledet.native.engine._load_lightning") as load_lightning:
                    class _FakeModule:
                        def __init__(self, *args, **kwargs):
                            pass

                        def log(self, *args, **kwargs):
                            return None

                    class _FakeTrainer:
                        def fit(self, module, datamodule=None):
                            return None

                    def _build_fake_trainer(**kwargs):
                        return _FakeTrainer()

                    def _build_fake_checkpoint(**kwargs):
                        return object()

                    fake_pl = types.SimpleNamespace(LightningModule=_FakeModule, Trainer=_build_fake_trainer)
                    fake_checkpoint = _build_fake_checkpoint
                    load_lightning.return_value = (fake_pl, fake_checkpoint)

                    run_native_training(
                        NativeProjectConfig(
                            dataset_root=str(root),
                            categories=("wake",),
                            detector_spec=spec,
                            output_dir=str(output),
                        )
                    )

            self.assertEqual(build_model.call_args.kwargs["detector_spec"], spec)

    def test_native_roi_runtime_returns_multi_detection_tensors(self):
        with patch.dict(sys.modules, self._tensor_runtime_modules()):
            import importlib

            roi_module = importlib.reload(importlib.import_module("simpledet.native.roi"))

            class _TensorBackbone:
                def __call__(self, images):
                    return roi_module.torch.tensor(np.zeros((1, 8, 2, 2), dtype=np.float32))

            class _TensorNeck:
                def __call__(self, features):
                    return OrderedDict([("0", features)])

            class _TensorRoIPool:
                def __call__(self, features, proposals, image_shapes):
                    channels = int(features["0"].shape[1])
                    count = int(proposals[0].shape[0])
                    return roi_module.torch.tensor(np.zeros((count, channels, 2, 2), dtype=np.float32))

            backbone = roi_module.NativeRoIBackbone(
                backbone=_TensorBackbone(),
                neck=_TensorNeck(),
                core_spec=roi_module.RoICoreSpec.from_num_levels(1),
            )
            model = roi_module.NativeRoIModel(
                backbone=backbone,
                box_roi_pool=_TensorRoIPool(),
                num_classes=2,
                in_channels=8,
                core_spec=roi_module.RoICoreSpec.from_num_levels(1),
            )

            image = roi_module.torch.tensor(np.zeros((3, 8, 8), dtype=np.float32))
            detections = model([image])

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertIn("boxes", detection)
        self.assertIn("scores", detection)
        self.assertIn("labels", detection)
        self.assertEqual(len(detection["boxes"]), 4)
        self.assertEqual(len(detection["scores"]), 4)
        self.assertEqual(len(detection["labels"]), 4)
        self.assertEqual(detection["labels"].tolist(), [1, 1, 1, 1])
