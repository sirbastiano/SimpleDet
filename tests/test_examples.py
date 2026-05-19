import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
EXPECTED_EXAMPLES = {
    "quick_detector.py",
    "timm_retinanet.py",
    "coco_training_config.py",
    "yolo_dataset_config.py",
    "load_ckpt_for_inference.py",
    "registry_discovery.py",
}


def _example_paths() -> list[Path]:
    return sorted(EXAMPLES_ROOT.glob("*.py"))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    package_path = str(REPO_ROOT / "simpledet")
    env["PYTHONPATH"] = (
        package_path
        if not env.get("PYTHONPATH")
        else package_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


def _run_example(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES_ROOT / script_name), *args],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )


class ExampleGalleryTests(unittest.TestCase):
    def test_expected_example_gallery_scripts_exist(self):
        self.assertEqual({path.name for path in _example_paths()}, EXPECTED_EXAMPLES)

    def test_example_scripts_compile(self):
        for path in _example_paths():
            with self.subTest(path=path.name):
                py_compile.compile(str(path), doraise=True)

    def test_example_scripts_import_without_side_effects(self):
        for path in _example_paths():
            with self.subTest(path=path.name):
                spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(callable(module.main))

    def test_example_scripts_expose_help(self):
        for path in _example_paths():
            with self.subTest(path=path.name):
                result = _run_example(path.name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn("usage:", result.stdout)

    def test_metadata_examples_run_without_sample_data(self):
        for script_name in ("quick_detector.py", "timm_retinanet.py", "registry_discovery.py"):
            with self.subTest(script=script_name):
                result = _run_example(script_name)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                payload = json.loads(result.stdout)
                self.assertTrue(payload)

    def test_timm_retinanet_example_builds_required_spec(self):
        spec = importlib.util.spec_from_file_location(
            "example_timm_retinanet",
            EXAMPLES_ROOT / "timm_retinanet.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        detector_spec = module.build_timm_retinanet_spec(num_classes=4)

        self.assertEqual(detector_spec.architecture, "retinanet")
        self.assertEqual(detector_spec.encoder.source, "timm")
        self.assertEqual(detector_spec.encoder.name, "timm:resnet18")
        self.assertFalse(detector_spec.encoder.pretrained)
        self.assertEqual(detector_spec.num_classes, 4)

    def test_data_examples_fail_clearly_when_sample_data_is_absent(self):
        cases = {
            "coco_training_config.py": "Sample COCO dataset not found",
            "yolo_dataset_config.py": "Sample YOLO dataset not found",
            "load_ckpt_for_inference.py": "Sample checkpoint not found",
        }
        for script_name, expected_message in cases.items():
            with self.subTest(script=script_name):
                result = _run_example(script_name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_message, result.stderr)
                self.assertIn("Pass --", result.stderr)

    def test_examples_do_not_hard_code_private_local_paths(self):
        private_markers = (
            "/shared/home/",
            "/home/",
            "/Users/",
            "C:\\Users\\",
            "/Data_large/",
        )
        offenders = []
        for path in _example_paths():
            text = path.read_text(encoding="utf-8")
            for marker in private_markers:
                if marker in text:
                    offenders.append(f"{path.name}: {marker}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
