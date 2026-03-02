import unittest
import simpledet.detectors.train as train
from pathlib import Path
from unittest.mock import patch
import tempfile
import types


class TestTrainConfig(unittest.TestCase):
    def test_parse_config_resolves_defaults(self):
        payload = train._parse_config({
            "dataset": "/tmp/ds",
            "model_name": "retinanet_resnet50_fpn",
            "output_dir": "results/test-run",
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "num_workers": 3,
            "train_split": "val",
        })

        self.assertEqual(payload.dataset_path, "/tmp/ds")
        self.assertEqual(payload.model_name, "retinanet_resnet50_fpn")
        self.assertEqual(payload.output_dir, "results/test-run")
        self.assertEqual(payload.optimizer, "sgd")
        self.assertEqual(payload.train_split, "val")
        self.assertIsNone(payload.format)

    def test_parse_config_requires_dataset(self):
        with self.assertRaises(TypeError) as context:
            train._parse_config({"model_name": "faster_rcnn_resnet50_fpn"})
        self.assertIn("`config` must define a dataset path", str(context.exception))

    def test_validate_model_name_rejects_unknown_model(self):
        with self.assertRaises(ValueError) as context:
            train._validate_model_name("unknown_model")
        self.assertIn("Unknown model", str(context.exception))

    def test_train_rejects_both_config_and_pipeline_arguments(self):
        with self.assertRaises(TypeError) as context:
            train.train(config={"dataset": "/tmp/ds"}, pipeline="unused", build=False)
        self.assertIn("Use either `config` for one-liner training", str(context.exception))

    def test_train_minimal_flow_uses_model_name_from_config(self):
        class _DummyTensor:
            def to(self, *_args, **_kwargs):
                return self

            def item(self):
                return 1.0

            def backward(self):
                return None

            def __radd__(self, other):
                if other == 0:
                    return self
                return self

            def __add__(self, other):
                return self

        class _DummyModel:
            def __init__(self):
                self.to_calls = []

            def to(self, device):
                self.to_calls.append(device)
                return self

            def train(self):
                return None

            def state_dict(self):
                return {"weights": "mock"}

            def __call__(self, _images, _targets):
                return {"loss": _DummyTensor()}

        class _FakeDataLoader:
            def __init__(self, dataset, *_args, **_kwargs):
                self._dataset = dataset

            def __iter__(self):
                yield (
                    ("image-bytes",),
                    (
                        {
                            "boxes": _DummyTensor(),
                            "labels": _DummyTensor(),
                            "image_id": _DummyTensor(),
                            "area": _DummyTensor(),
                            "iscrowd": _DummyTensor(),
                        },
                    ),
                )

        class _FakeOptimizer:
            def zero_grad(self):
                return None

            def step(self):
                return None

        class _FakeTorch(types.ModuleType):
            def __init__(self):
                super().__init__("torch")
                self.manual_seed = lambda _seed: None
                self.cuda = types.SimpleNamespace(is_available=lambda: False)
                self.device = lambda value: types.SimpleNamespace(type=str(value))
                self.save = lambda payload, path: Path(path).write_text("checkpoint")

                utils_module = types.ModuleType("torch.utils")
                data_module = types.ModuleType("torch.utils.data")
                data_module.DataLoader = _FakeDataLoader
                utils_module.data = data_module
                self.utils = utils_module

        class _FakeDataset:
            def __len__(self):
                return 1

        fake_torch = _FakeTorch()
        fake_dataset = _FakeDataset()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            config = train._ResolvedTrainConfig(
                dataset_path="/tmp/ds",
                output_dir=str(output_dir),
                model_name="retinanet_resnet50_fpn",
                epochs=1,
                batch_size=1,
                learning_rate=0.1,
                optimizer="sgd",
                seed=71,
                device="cpu",
                format=None,
                train_split="train",
                num_workers=0,
                num_classes=3,
            )

            with patch.dict(
                "sys.modules",
                {
                    "torch": fake_torch,
                    "torch.utils": fake_torch.utils,
                    "torch.utils.data": fake_torch.utils.data,
                },
            ), patch(
                "simpledet.detectors.train.load_dataset",
                lambda **_kwargs: {"categories": ["a", "b", "c"], "samples": []},
            ), patch(
                "simpledet.detectors.train._build_torchvision_model",
                lambda *_args, **_kwargs: _DummyModel(),
            ), patch(
                "simpledet.detectors.train._build_dataset",
                lambda *_args, **_kwargs: fake_dataset,
            ), patch(
                "simpledet.detectors.train._build_optimizer",
                lambda *_args, **_kwargs: _FakeOptimizer(),
            ):
                result = train._train_minimal_flow(config)

            self.assertEqual(result["model_name"], "retinanet_resnet50_fpn")
            self.assertTrue(Path(result["checkpoint_path"]).is_file())
