import io
import json
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from simpledet.cli import _run_direct_inference, _run_direct_training, main
from simpledet import __version__


class TestCli(unittest.TestCase):
    def _fake_runtime_modules(self):
        fake_torch = types.ModuleType("torch")
        fake_nn = types.ModuleType("torch.nn")
        fake_f = types.ModuleType("torch.nn.functional")
        fake_numpy = types.ModuleType("numpy")
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
        fake_torch.nn = fake_nn
        fake_numpy.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
        fake_numpy.ndarray = object
        return {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": fake_f,
            "numpy": fake_numpy,
        }

    def test_main_returns_version(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["--version"])

        self.assertEqual(exit_code, 0)
        self.assertIn(__version__, output.getvalue().strip())

    def test_main_prints_help_when_no_args(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("SimpleDet package bootstrap", output.getvalue())

    def test_main_forwards_check_runtime_to_checker(self):
        with patch("simpledet.cli._check_runtime", return_value=7) as patched:
            exit_code = main(["--check-runtime"])

        self.assertEqual(exit_code, 7)
        patched.assert_called_once()

    def test_main_lists_detectors(self):
        output = io.StringIO()
        with patch("simpledet.cli._list_detectors", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--list-detectors"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()

    def test_main_lists_encoders(self):
        output = io.StringIO()
        with patch("simpledet.cli._list_encoders", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--list-encoders"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()

    def test_main_shows_detector_help(self):
        output = io.StringIO()
        with patch("simpledet.cli._show_detector_help", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--show-detector-help", "retinanet"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with("retinanet")

    def test_main_initializes_project_config(self):
        output = io.StringIO()
        with patch("simpledet.cli._init_project", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--init-project", "project.toml", "--project-format", "toml", "--force"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with("project.toml", "toml", True)

    def test_main_validates_project_config(self):
        output = io.StringIO()
        with patch("simpledet.cli._validate_project", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--project-validate", "project.json"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with("project.json")

    def test_main_runs_project_config(self):
        output = io.StringIO()
        with patch("simpledet.cli._run_project", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--project-run", "project.json", "--stages", "build", "test"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with("project.json", ["build", "test"])

    def test_main_runs_direct_training(self):
        output = io.StringIO()
        with patch("simpledet.cli._run_direct_training", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--train-root",
                        "/dataset",
                        "--categories",
                        "wake",
                        "--in-channels",
                        "3",
                        "--detector",
                        "retinanet",
                        "--batch-size",
                        "2",
                        "--no-validate",
                    ]
                )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        args = patched.call_args.args[0]
        self.assertEqual(args.train_root, "/dataset")
        self.assertEqual(args.categories, ["wake"])
        self.assertEqual(args.in_channels, 3)
        self.assertEqual(args.batch_size, 2)
        self.assertFalse(args.validate)

    def test_main_runs_direct_inference(self):
        output = io.StringIO()
        with patch("simpledet.cli._run_direct_inference", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--infer-root",
                        "/dataset",
                        "--categories",
                        "wake",
                        "--in-channels",
                        "1",
                        "--detector",
                        "retinanet",
                    ]
                )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        args = patched.call_args.args[0]
        self.assertEqual(args.infer_root, "/dataset")
        self.assertTrue(args.validate)

    def test_main_runs_direct_training_with_detector_builder_args(self):
        output = io.StringIO()
        with patch("simpledet.cli._run_direct_training", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--train-root",
                        "/dataset",
                        "--categories",
                        "wake",
                        "--in-channels",
                        "3",
                        "--detector",
                        "retinanet",
                        "--encoder",
                        "resnet18.a1_in1k",
                    ]
                )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        args = patched.call_args.args[0]
        self.assertEqual(args.detector, "retinanet")
        self.assertEqual(args.encoder, "resnet18.a1_in1k")

    def test_main_runs_direct_evaluation(self):
        output = io.StringIO()
        with patch("simpledet.cli._run_direct_evaluation", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--eval-root",
                        "/dataset",
                        "--categories",
                        "wake",
                        "--in-channels",
                        "1",
                        "--detector",
                        "retinanet",
                    ]
                )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        args = patched.call_args.args[0]
        self.assertEqual(args.eval_root, "/dataset")

    def test_main_requires_categories_for_direct_execution(self):
        with self.assertRaises(SystemExit) as exc:
            main(
                [
                    "--train-root",
                    "/dataset",
                    "--in-channels",
                    "3",
                    "--detector",
                    "retinanet",
                ]
            )

        self.assertEqual(exc.exception.code, 2)

    def test_main_requires_detector_for_direct_execution(self):
        with self.assertRaises(SystemExit) as exc:
            main(
                [
                    "--infer-root",
                    "/dataset",
                    "--categories",
                    "wake",
                    "--in-channels",
                    "3",
                ]
            )

        self.assertEqual(exc.exception.code, 2)

    def test_run_direct_training_relays_native_backend_result(self):
        args = type(
            "Args",
            (),
            {
                "train_root": "/dataset",
                "categories": ["wake"],
                "in_channels": 3,
                "detector": "retinanet",
                "encoder": "resnet18.a1_in1k",
                "num_classes": None,
                "tif_channels_to_load": None,
                "result_folder": None,
                "validate": True,
                "resize": None,
                "batch_size": None,
                "max_epochs": None,
                "learning_rate": None,
            },
        )()

        output = io.StringIO()
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            with patch("simpledet.run_training", return_value={"backend": "native_lightning"}) as patched:
                with redirect_stdout(output):
                    exit_code = _run_direct_training(args)

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        self.assertEqual(json.loads(output.getvalue())["backend"], "native_lightning")

    def test_run_direct_inference_relays_native_backend_result(self):
        args = type(
            "Args",
            (),
            {
                "infer_root": "/dataset",
                "categories": ["wake"],
                "in_channels": 3,
                "detector": "retinanet",
                "encoder": "resnet18.a1_in1k",
                "num_classes": None,
                "tif_channels_to_load": None,
                "result_folder": None,
                "validate": True,
                "resize": None,
                "batch_size": None,
                "max_epochs": None,
                "learning_rate": None,
            },
        )()

        output = io.StringIO()
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            with patch("simpledet.run_inference", return_value={"backend": "native_lightning"}) as patched:
                with redirect_stdout(output):
                    exit_code = _run_direct_inference(args)

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        self.assertEqual(json.loads(output.getvalue())["backend"], "native_lightning")
