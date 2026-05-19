import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import simpledet
from simpledet.errors import (
    CheckpointError,
    ConfigPathError,
    DatasetError,
    OptionalDependencyError,
    RegistryLookupError,
    SimpleDetError,
    TensorContractError,
)


class PublicErrorTaxonomyTests(unittest.TestCase):
    def test_root_package_exports_error_taxonomy(self):
        for name in (
            "SimpleDetError",
            "OptionalDependencyError",
            "RegistryLookupError",
            "ConfigValidationError",
            "DatasetError",
            "TensorContractError",
            "CheckpointError",
            "CheckpointNotFoundError",
        ):
            self.assertTrue(issubclass(getattr(simpledet, name), SimpleDetError))

    def test_missing_timm_raises_optional_dependency_error_with_exact_command(self):
        from simpledet.detectors import _deps

        with patch(
            "simpledet.detectors._deps.import_module",
            side_effect=ModuleNotFoundError("missing", name="timm"),
        ):
            with self.assertRaises(OptionalDependencyError) as context:
                _deps.require_dependency("timm", "native backbones")

        self.assertEqual(
            context.exception.install_command,
            "python -m pip install 'simpledet[timm]'",
        )
        self.assertIn(
            "Install with `python -m pip install 'simpledet[timm]'`.",
            str(context.exception),
        )

    def test_unknown_registry_component_uses_public_lookup_error(self):
        from simpledet.extensions import ExtensionRegistry

        registry = ExtensionRegistry("detector")
        registry.register("retinanet", aliases=("RetinaNet",))(object)

        with self.assertRaises(RegistryLookupError) as context:
            registry.lookup("unknown_detector")

        self.assertNotIsInstance(context.exception, KeyError)
        self.assertIn("Unknown detector component 'unknown_detector'", str(context.exception))
        self.assertIn("Registered detector names: retinanet", str(context.exception))

    def test_config_strict_validation_raises_config_error(self):
        from simpledet.api import validate_project_config

        with self.assertRaises(ConfigPathError) as context:
            validate_project_config(
                {
                    "dataset": {
                        "data_root": "/missing/simpledet",
                        "classes": ["wake"],
                    },
                    "detector": {"name": "retinanet"},
                    "runtime": {"result_folder": "/tmp/simpledet-runs"},
                },
                strict=True,
            )

        self.assertIn("Project validation failed", str(context.exception))
        self.assertIn("dataset_root", str(context.exception))

    def test_dataset_registry_error_uses_dataset_error_base(self):
        from simpledet.detectors.data import load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(DatasetError) as context:
                load_dataset(tmpdir, format="unknown_format")

        self.assertIn("Unsupported dataset format 'unknown_format'", str(context.exception))

    def test_tensor_contract_error_is_public_value_error(self):
        error = TensorContractError("tensor contract mismatch")

        self.assertIsInstance(error, SimpleDetError)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "tensor contract mismatch")

    def test_missing_checkpoint_uses_checkpoint_error_base(self):
        from simpledet.detectors.infer import CheckpointNotFoundError, load_checkpoint_for_inference

        missing = Path("/tmp/simpledet-missing-checkpoint.ckpt")
        with self.assertRaises(CheckpointNotFoundError) as context:
            load_checkpoint_for_inference(missing)

        self.assertIsInstance(context.exception, CheckpointError)
        self.assertIn("Checkpoint not found or unsupported format", str(context.exception))
        self.assertIn(str(missing), str(context.exception))
