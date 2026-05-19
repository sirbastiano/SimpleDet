import io
import json
import subprocess
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
        fake_nn.Embedding = lambda *args, **kwargs: types.SimpleNamespace(weight=object())
        fake_nn.Identity = lambda *args, **kwargs: object()
        fake_nn.Linear = lambda *args, **kwargs: object()
        fake_nn.ReLU = lambda *args, **kwargs: object()
        fake_nn.TransformerEncoderLayer = lambda *args, **kwargs: object()
        fake_nn.TransformerDecoderLayer = lambda *args, **kwargs: object()
        fake_nn.TransformerEncoder = lambda *args, **kwargs: object()
        fake_nn.TransformerDecoder = lambda *args, **kwargs: object()
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
        patched.assert_called_once_with(family=None, pattern=None)

    def test_main_accepts_list_detectors_command_alias(self):
        output = io.StringIO()
        with patch("simpledet.cli._list_detectors", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["list-detectors"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with(family=None, pattern=None)

    def test_main_accepts_discovery_command_aliases(self):
        commands = {
            "list-heads": "_list_heads",
            "list-backbones": "_list_backbones",
            "list-necks": "_list_necks",
            "list-datasets": "_list_datasets",
            "doctor": "_doctor",
        }
        for command, function_name in commands.items():
            with self.subTest(command=command):
                with patch(f"simpledet.cli.{function_name}", return_value=0) as patched:
                    exit_code = main([command])
                self.assertEqual(exit_code, 0)
                patched.assert_called_once()

    def test_list_detectors_prints_native_validation_status(self):
        from simpledet.cli import _list_detectors

        output = io.StringIO()
        with patch.dict(sys.modules, self._fake_runtime_modules()):
            with redirect_stdout(output):
                exit_code = _list_detectors()

        self.assertEqual(exit_code, 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "name\tfamily\tnative_validation")
        cascade = [line for line in lines if line.startswith("cascade_rcnn\t")]
        cornernet = [line for line in lines if line.startswith("cornernet\t")]
        grid = [line for line in lines if line.startswith("grid_rcnn\t")]
        sparse = [line for line in lines if line.startswith("sparse_rcnn\t")]
        self.assertEqual(cascade, ["cascade_rcnn\troi\truntime_validated"])
        self.assertEqual(cornernet, ["cornernet\tdense\truntime_validated"])
        self.assertEqual(grid, ["grid_rcnn\troi\truntime_validated"])
        self.assertEqual(sparse, ["sparse_rcnn\troi\truntime_validated"])

    def test_main_lists_encoders(self):
        output = io.StringIO()
        with patch("simpledet.cli._list_encoders", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--list-encoders"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with(pattern=None)

    def test_main_lists_encoders_as_backbone_alias(self):
        with patch("simpledet.cli._list_backbones", return_value=0) as patched:
            exit_code = main(["--list-encoders"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with(pattern=None)

    def test_list_heads_prints_kind_validation_and_many_aliases(self):
        from simpledet.cli import _list_heads

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = _list_heads()

        self.assertEqual(exit_code, 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "name\tkind\tvalidation_status\trequired_extra")
        rows = [line.split("\t") for line in lines[1:]]
        self.assertGreaterEqual(len(rows), 31)
        by_name = {row[0]: row for row in rows}
        self.assertEqual(by_name["retina_head"][1:4], ["dense", "runtime_validated", "cpu"])
        self.assertEqual(by_name["CascadeBBoxHead"][1:3], ["roi", "runtime_validated"])
        self.assertEqual(by_name["detr_head"][1:3], ["transformer", "runtime_validated"])

    def test_discovery_without_timm_reports_required_extra(self):
        from simpledet.cli import _list_backbones

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = _list_backbones()

        self.assertEqual(exit_code, 0)
        lines = output.getvalue().splitlines()
        by_name = {line.split("\t")[0]: line.split("\t") for line in lines[1:]}
        self.assertIn("resnet50", by_name)
        self.assertIn("timm", by_name["resnet50"][3].split(","))

    def test_list_datasets_prints_registered_formats(self):
        from simpledet.cli import _list_datasets

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = _list_datasets()

        self.assertEqual(exit_code, 0)
        names = {line.split("\t")[0] for line in output.getvalue().splitlines()[1:]}
        self.assertTrue({"coco", "csv", "json", "voc", "yolo"}.issubset(names))

    def test_doctor_reports_optional_extras(self):
        from simpledet.cli import _doctor

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = _doctor()

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("extra\tstatus\tmissing", text)
        self.assertIn("timm\t", text)

    def test_python_module_invalid_discovery_option_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "simpledet", "list-heads", "--family", "dense"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--family is only supported by list-detectors", result.stderr)

    def test_main_shows_detector_help(self):
        output = io.StringIO()
        with patch("simpledet.cli._show_detector_help", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--show-detector-help", "retinanet"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with("retinanet")

    def test_show_detector_help_reports_planned_transformer_variants(self):
        from simpledet.cli import _show_detector_help

        with self.assertRaisesRegex(ValueError, "Unsupported transformer variant.*planned"):
            _show_detector_help("detr3d")

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

    def test_main_runs_project_config_with_config_stages(self):
        output = io.StringIO()
        with patch("simpledet.cli._run_project", return_value=0) as patched:
            with redirect_stdout(output):
                exit_code = main(["--project-run", "project.json"])

        self.assertEqual(exit_code, 0)
        patched.assert_called_once_with("project.json", None)

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
