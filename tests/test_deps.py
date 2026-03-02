import unittest
from unittest.mock import patch

import simpledet.detectors._deps as deps


class TestDeps(unittest.TestCase):
    def test_require_dependency_raises_with_optional_dependency_hint(self):
        with patch(
            "simpledet.detectors._deps.import_module",
            side_effect=ModuleNotFoundError("mod", name="torch"),
        ):
            with self.assertRaises(ImportError) as context:
                deps.require_dependency("torch", "training")
            self.assertIn(
                "simpledet public API 'training' requires the optional dependency 'torch'",
                str(context.exception),
            )

    def test_require_detector_runtime_and_config_dependency_use_import_module(self):
        with patch("simpledet.detectors._deps.import_module", side_effect=[None, None, None]):
            deps.require_detector_runtime()

        with patch("simpledet.detectors._deps.import_module", side_effect=[None]):
            deps.require_config_dependency()

    def test_require_dependency_raises_missing_dependency_name(self):
        def fake_import(name):
            if name == "torch":
                raise ModuleNotFoundError("not found", name="torch")
            return None

        with patch("simpledet.detectors._deps.import_module", side_effect=fake_import):
            with self.assertRaises(ImportError) as context:
                deps.require_dependency("torch", "train")
            self.assertIn("optional dependency 'torch'", str(context.exception))
