import sys
import types
import unittest
from unittest.mock import patch

from simpledet.extensions import ASSIGNERS, DECODERS, DETECTORS, ENCODERS, HEADS, LOSSES, NECKS, POSTPROCESSORS


def _clear_native_runtime_modules():
    for registry in (ASSIGNERS, DECODERS, DETECTORS, ENCODERS, HEADS, LOSSES, NECKS, POSTPROCESSORS):
        for name, factory in tuple(registry._items.items()):
            if str(getattr(factory, "__module__", "")).startswith("simpledet.native"):
                registry._items.pop(name, None)
                registry._metadata.pop(name, None)
    for module_name in tuple(sys.modules):
        if module_name == "simpledet.native" or module_name.startswith("simpledet.native."):
            sys.modules.pop(module_name, None)
    simpledet_package = sys.modules.get("simpledet")
    if simpledet_package is not None:
        vars(simpledet_package).pop("native", None)


class NativeApiRoutingTests(unittest.TestCase):
    def setUp(self):
        _clear_native_runtime_modules()

    def tearDown(self):
        _clear_native_runtime_modules()

    def _fake_runtime_modules(self):
        fake_torch = types.ModuleType("torch")
        fake_nn = types.ModuleType("torch.nn")
        fake_f = types.ModuleType("torch.nn.functional")
        fake_numpy = types.ModuleType("numpy")
        fake_timm = types.ModuleType("timm")
        fake_native = types.ModuleType("simpledet.native")
        fake_native.__path__ = []
        fake_runtime = types.ModuleType("simpledet.native.runtime")

        class NativeProjectConfig:
            def __init__(self, **kwargs):
                vars(self).update(kwargs)

        fake_runtime.NativeProjectConfig = NativeProjectConfig
        fake_runtime.run_native_training = lambda config: {"backend": "native_lightning"}
        fake_runtime.run_native_inference = lambda config: {"backend": "native_lightning"}
        fake_native.runtime = fake_runtime
        simpledet_package = sys.modules.get("simpledet")
        if simpledet_package is not None:
            vars(simpledet_package)["native"] = fake_native
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            manual_seed=lambda *_args, **_kwargs: None,
            manual_seed_all=lambda *_args, **_kwargs: None,
        )
        fake_torch.manual_seed = lambda *_args, **_kwargs: None
        fake_torch.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
        )
        fake_torch.__version__ = "0.0.0"

        class Module:
            def __init__(self, *args, **kwargs):
                pass

            def parameters(self):
                return []

        fake_nn.Module = Module
        fake_nn.ModuleList = list
        fake_nn.Sequential = lambda *layers: object()
        fake_nn.Conv2d = lambda *args, **kwargs: object()
        fake_nn.ConvTranspose2d = lambda *args, **kwargs: object()
        fake_nn.Identity = lambda *args, **kwargs: object()
        fake_nn.Linear = lambda *args, **kwargs: object()
        fake_nn.Embedding = lambda *args, **kwargs: object()
        fake_nn.ReLU = lambda *args, **kwargs: object()
        fake_torch.nn = fake_nn
        fake_numpy.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
        fake_numpy.ndarray = object
        fake_timm.list_models = lambda pattern=None: ["resnet18.a1_in1k"]
        return {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_f,
            "numpy": fake_numpy,
            "timm": fake_timm,
            "simpledet.native": fake_native,
            "simpledet.native.runtime": fake_runtime,
        }

    def _retina_spec(self):
        return {
            "architecture": "retinanet",
            "family": "dense",
            "num_classes": 1,
            "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
        }

    def test_run_training_prefers_native_backend_for_supported_retinanet(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec=self._retina_spec(),
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_inference_prefers_native_backend_for_supported_retinanet(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_inference

            with patch("simpledet.native.runtime.run_native_inference", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_inference(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec=self._retina_spec(),
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_prefers_native_backend_for_supported_faster_rcnn(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "faster_rcnn",
                        "family": "roi",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_prefers_native_backend_for_supported_fcos(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "fcos",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_prefers_native_backend_for_supported_atss(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "atss",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_prefers_native_backend_for_supported_gfl(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "gfl",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_prefers_native_backend_for_supported_mask_rcnn(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "mask_rcnn",
                        "family": "roi",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_rejects_model_cfg_input(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with self.assertRaises(TypeError) as context:
                run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec=self._retina_spec(),
                    model_cfg={"type": "RetinaNet"},
                    validate=False,
                )
            self.assertIn("model_cfg", str(context.exception))

    def test_run_training_prefers_native_backend_for_registered_custom_detector(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training
            from simpledet.suite import build_custom_detector

            custom_name = "routing_custom_detector"
            if custom_name not in DETECTORS.names():
                @DETECTORS.register(custom_name)
                def _custom_assembler(components, *, num_classes):
                    return {"num_classes": num_classes}

            spec = build_custom_detector(custom_name, family="dense", num_classes=1, encoder="resnet18.a1_in1k")

            with patch("simpledet.native.runtime.run_native_training", return_value={"backend": "native_lightning"}) as native_mock:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec=spec,
                    validate=False,
                )

        native_mock.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")
