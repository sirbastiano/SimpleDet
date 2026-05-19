import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class PublicApiTests(unittest.TestCase):
    def _fake_runtime_modules(self):
        fake_torch = types.ModuleType("torch")
        fake_nn = types.ModuleType("torch.nn")
        fake_f = types.ModuleType("torch.nn.functional")
        fake_numpy = types.ModuleType("numpy")
        fake_timm = types.ModuleType("timm")

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
        fake_nn.Module = type("Module", (), {})
        fake_nn.ModuleList = list
        fake_nn.Sequential = lambda *layers: object()
        fake_nn.Conv2d = lambda *args, **kwargs: object()
        fake_nn.ConvTranspose2d = lambda *args, **kwargs: object()
        fake_nn.Identity = lambda *args, **kwargs: object()
        fake_nn.ReLU = lambda *args, **kwargs: object()
        fake_nn.Embedding = lambda *args, **kwargs: object()
        fake_nn.Linear = lambda *args, **kwargs: object()
        fake_torch.nn = fake_nn
        fake_numpy.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
        fake_numpy.ndarray = object
        fake_timm.list_models = lambda pattern=None: ["resnet18.a1_in1k", "convnext_tiny.in12k_ft_in1k"]

        return {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_f,
            "numpy": fake_numpy,
            "timm": fake_timm,
        }

    def test_project_layout_validation_report_tracks_expected_paths(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import ProjectLayout

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "imgs").mkdir()
            (root / "Annotations").mkdir()
            (root / "Annotations" / "test_annotations.json").write_text("{}", encoding="utf-8")

            layout = ProjectLayout(dataset_root=tmpdir)
            report = layout.validation_report()

        self.assertTrue(report["exists"]["dataset_root"])
        self.assertTrue(report["exists"]["images"])
        self.assertTrue(report["exists"]["annotations_dir"])
        self.assertFalse(report["exists"]["train_annotations"])
        self.assertFalse(report["exists"]["val_annotations"])
        self.assertTrue(report["exists"]["test_annotations"])

    def test_project_config_template_supports_toml_and_json(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import project_config_template

            toml_template = project_config_template("toml")
            json_template = project_config_template("json")

        self.assertIn("[dataset]", toml_template)
        self.assertIn('"dataset"', json_template)

    def test_init_project_config_writes_template_file(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import init_project_config

            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "project.toml"
                written_path = init_project_config(config_path)
                self.assertTrue(Path(written_path).exists())
                written_text = Path(written_path).read_text(encoding="utf-8")

        self.assertIn("[runtime]", written_text)

    def test_legacy_python_project_config_paths_are_rejected(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import init_project_config, load_project_config

            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "legacy.py"
                config_path.write_text("model = dict(type='RetinaNet')\n", encoding="utf-8")

                with self.assertRaises(ValueError) as load_context:
                    load_project_config(config_path)
                with self.assertRaises(ValueError) as init_context:
                    init_project_config(config_path, overwrite=True)
                with self.assertRaises(ValueError) as explicit_format_context:
                    init_project_config(config_path, format="toml", overwrite=True)

        for message in (
            str(load_context.exception),
            str(init_context.exception),
            str(explicit_format_context.exception),
        ):
            self.assertIn("Legacy MMDetection .py config import/conversion is unsupported", message)
            self.assertIn("specific converter", message)

    def test_validate_project_config_reports_missing_paths(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import validate_project_config

            report = validate_project_config(
                {
                    "dataset": {
                        "data_root": "/missing",
                        "annot_file_train": "/missing/Annotations/train_annotations.json",
                        "annot_file_val": "/missing/Annotations/val_annotations.json",
                        "annot_file_test": "/missing/Annotations/test_annotations.json",
                        "categories": ["wake"],
                        "in_channels": 3,
                    },
                    "runtime": {"result_folder": "/tmp/results"},
                    "optimization": {},
                    "detector_spec": {
                        "architecture": "retinanet",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                }
            )

        self.assertIn("dataset_root", report["missing"])
        self.assertIn("train_annotations", report["missing"])

    def test_run_project_delegates_selected_stages_to_native_runtime(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_project

            train_result = {
                "backend": "native_lightning",
                "stages": ["fit"],
                "checkpoint_path": "/results/checkpoints/epoch_001.ckpt",
            }
            with patch(
                "simpledet.native.runtime.run_native_inference",
                return_value={"backend": "native_lightning", "stages": ["test"]},
            ) as infer_mock, patch(
                "simpledet.native.runtime.run_native_training",
                return_value=train_result,
            ) as train_mock:
                result = run_project(
                    {
                        "dataset": {
                            "data_root": "/dataset",
                            "annot_file_train": "/dataset/Annotations/train_annotations.json",
                            "annot_file_val": "/dataset/Annotations/val_annotations.json",
                            "annot_file_test": "/dataset/Annotations/test_annotations.json",
                            "categories": ["wake"],
                            "in_channels": 3,
                        },
                        "runtime": {"result_folder": "/results"},
                        "optimization": {"learning_rate": 0.001},
                        "detector_spec": {
                            "architecture": "retinanet",
                            "family": "dense",
                            "num_classes": 1,
                            "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                        },
                    },
                    stages=("build", "train", "test"),
                    validate=False,
                )

        train_mock.assert_called_once()
        infer_mock.assert_called_once()
        self.assertEqual(infer_mock.call_args.args[0].checkpoint_path, train_result["checkpoint_path"])
        self.assertEqual(result["stages"], ["build", "train", "test"])
        self.assertEqual(result["backend"], "native_lightning")
        self.assertEqual(result["train"]["checkpoint_path"], train_result["checkpoint_path"])

    def test_run_training_uses_native_runtime(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with patch(
                "simpledet.native.runtime.run_native_training",
                return_value={
                    "backend": "native_lightning",
                    "checkpoint_path": "/results/checkpoints/epoch_001.ckpt",
                },
            ) as patched:
                result = run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "retinanet",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    validate=False,
                )

        patched.assert_called_once()
        self.assertEqual(result["backend"], "native_lightning")
        self.assertEqual(result["checkpoint_path"], "/results/checkpoints/epoch_001.ckpt")

    def test_run_inference_uses_native_runtime(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_inference

            checkpoint_path = "/dataset/runs/simpledet/checkpoints/epoch_003.ckpt"
            with patch(
                "simpledet.native.runtime.run_native_inference",
                return_value={
                    "backend": "native_lightning",
                    "checkpoint_path": checkpoint_path,
                },
            ) as patched:
                result = run_inference(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "retinanet",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    checkpoint_path=checkpoint_path,
                    validate=False,
                )

        patched.assert_called_once()
        self.assertEqual(patched.call_args.args[0].checkpoint_path, checkpoint_path)
        self.assertEqual(result["backend"], "native_lightning")
        self.assertEqual(result["checkpoint_path"], checkpoint_path)

    def test_run_evaluation_delegates_to_inference(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_evaluation

            with patch(
                "simpledet.api.run_inference",
                return_value={"backend": "native_lightning", "stages": ["test"]},
            ) as patched:
                result = run_evaluation(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "retinanet",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    checkpoint_path="/dataset/runs/simpledet/checkpoints/epoch_003.ckpt",
                )

        patched.assert_called_once()
        self.assertEqual(patched.call_args.kwargs["checkpoint_path"], "/dataset/runs/simpledet/checkpoints/epoch_003.ckpt")
        self.assertEqual(result["backend"], "native_lightning")

    def test_run_training_rejects_model_cfg(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            from simpledet.api import run_training

            with self.assertRaises(TypeError):
                run_training(
                    dataset_root="/dataset",
                    categories=("wake",),
                    in_channels=3,
                    detector_spec={
                        "architecture": "retinanet",
                        "family": "dense",
                        "num_classes": 1,
                        "encoder": {"name": "resnet18.a1_in1k", "source": "timm"},
                    },
                    model_cfg={"type": "RetinaNet"},
                    validate=False,
                )

    def test_package_exports_native_helpers(self):
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            import simpledet
            from simpledet.api import (
                DatasetConfig,
                OptimizationConfig,
                ProjectConfig,
                ProjectLayout,
                RuntimeConfig,
                init_project_config,
                load_project_config,
                project_config_template,
                run_evaluation,
                run_inference,
                run_project,
                run_training,
                validate_project_config,
            )

        self.assertEqual(simpledet.ProjectLayout.__module__, ProjectLayout.__module__)
        self.assertEqual(simpledet.DatasetConfig.__module__, DatasetConfig.__module__)
        self.assertEqual(simpledet.RuntimeConfig.__module__, RuntimeConfig.__module__)
        self.assertEqual(simpledet.OptimizationConfig.__module__, OptimizationConfig.__module__)
        self.assertEqual(simpledet.ProjectConfig.__module__, ProjectConfig.__module__)
        self.assertEqual(simpledet.ProjectLayout.__name__, ProjectLayout.__name__)
        self.assertEqual(simpledet.DatasetConfig.__name__, DatasetConfig.__name__)
        self.assertEqual(simpledet.RuntimeConfig.__name__, RuntimeConfig.__name__)
        self.assertEqual(simpledet.OptimizationConfig.__name__, OptimizationConfig.__name__)
        self.assertEqual(simpledet.ProjectConfig.__name__, ProjectConfig.__name__)
        self.assertEqual(simpledet.load_project_config.__name__, load_project_config.__name__)
        self.assertEqual(simpledet.project_config_template.__name__, project_config_template.__name__)
        self.assertEqual(simpledet.init_project_config.__name__, init_project_config.__name__)
        self.assertEqual(simpledet.validate_project_config.__name__, validate_project_config.__name__)
        self.assertEqual(simpledet.run_project.__name__, run_project.__name__)
        self.assertEqual(simpledet.run_training.__name__, run_training.__name__)
        self.assertEqual(simpledet.run_inference.__name__, run_inference.__name__)
        self.assertEqual(simpledet.run_evaluation.__name__, run_evaluation.__name__)
