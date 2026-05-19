import unittest
from unittest.mock import patch

import simpledet.detectors._deps as deps
from simpledet.errors import OptionalDependencyError


class TestDeps(unittest.TestCase):
    def test_require_dependency_raises_with_optional_dependency_hint(self):
        with patch(
            "simpledet.detectors._deps.import_module",
            side_effect=ModuleNotFoundError("mod", name="torch"),
        ):
            with self.assertRaises(OptionalDependencyError) as context:
                deps.require_dependency("torch", "training")
            self.assertIn(
                "simpledet public API 'training' requires the optional dependency 'torch'",
                str(context.exception),
            )

    def test_require_detector_runtime_and_config_dependency_use_import_module(self):
        with patch("simpledet.detectors._deps.import_module", side_effect=[None, None, None, None]):
            deps.require_detector_runtime()

        with patch("simpledet.detectors._deps.import_module", side_effect=[None]):
            deps.require_config_dependency()

    def test_require_dependency_raises_missing_dependency_name(self):
        def fake_import(name):
            if name == "torch":
                raise ModuleNotFoundError("not found", name="torch")
            return None

        with patch("simpledet.detectors._deps.import_module", side_effect=fake_import):
            with self.assertRaises(OptionalDependencyError) as context:
                deps.require_dependency("torch", "train")
            self.assertIn("optional dependency 'torch'", str(context.exception))

    def test_require_dependency_includes_timm_extra_hint(self):
        with patch(
            "simpledet.detectors._deps.import_module",
            side_effect=ModuleNotFoundError("not found", name="timm"),
        ):
            with self.assertRaises(OptionalDependencyError) as context:
                deps.require_dependency("timm", "native backbones")

        self.assertIn("optional dependency 'timm'", str(context.exception))
        self.assertEqual(
            context.exception.install_command,
            "python -m pip install 'simpledet[timm]'",
        )
        self.assertIn("python -m pip install 'simpledet[timm]'", str(context.exception))
