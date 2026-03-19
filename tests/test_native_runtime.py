import json
from collections import OrderedDict
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simpledet.suite import build_detector


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

        fake_nn.Module = Module
        fake_nn.ModuleList = list
        fake_nn.Sequential = lambda *layers: Module()
        fake_nn.Conv2d = lambda *args, **kwargs: Module()
        fake_nn.ReLU = lambda *args, **kwargs: Module()
        fake_torch.nn = fake_nn
        fake_torch.tensor = lambda *args, **kwargs: 0
        fake_torch.zeros_like = lambda value: value
        fake_torch.zeros = lambda *args, **kwargs: 0
        fake_torch.finfo = lambda dtype: types.SimpleNamespace(eps=1e-12)
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

    def _write_dataset(self, root: Path) -> None:
        (root / "Annotations").mkdir(parents=True, exist_ok=True)
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
                        def fit(self, module, datamodule=None):
                            datamodule.setup("fit")
                            return None

                    fake_pl = types.SimpleNamespace(LightningModule=_FakeModule, Trainer=lambda **kwargs: _FakeTrainer())
                    fake_checkpoint = lambda **kwargs: object()
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
            self.assertTrue((output / "native-manifest.json").exists())

    def test_native_evaluation_returns_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            output = Path(tmpdir) / "runs"
            self._write_dataset(root)
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
                        def test(self, module, datamodule=None):
                            datamodule.setup("test")
                            image = roi_module.torch.tensor(np.zeros((3, 8, 8), dtype=np.float32))
                            target = {"image_id": [1], "boxes": [], "labels": []}
                            module.test_step(([image], [target]), 0)
                            return None

                    fake_pl = types.SimpleNamespace(LightningModule=_FakeModule, Trainer=lambda **kwargs: _FakeTrainer())
                    fake_checkpoint = lambda **kwargs: object()
                    load_lightning.return_value = (fake_pl, fake_checkpoint)

                    result = run_native_evaluation(
                        NativeProjectConfig(
                            dataset_root=str(root),
                            categories=("wake",),
                            detector_spec=spec,
                            output_dir=str(output),
                        )
                    )

            self.assertEqual(result["backend"], "native_lightning")
            self.assertTrue((output / "native-manifest.json").exists())
            prediction = result["predictions"][0]
            self.assertEqual(prediction["image_id"], 1)
            self.assertIn("boxes", prediction)
            self.assertIn("scores", prediction)
            self.assertIn("labels", prediction)
            self.assertEqual(len(prediction["boxes"]), 4)
            self.assertEqual(len(prediction["scores"]), 4)
            self.assertEqual(len(prediction["labels"]), 4)

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

                    fake_pl = types.SimpleNamespace(LightningModule=_FakeModule, Trainer=lambda **kwargs: _FakeTrainer())
                    fake_checkpoint = lambda **kwargs: object()
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
