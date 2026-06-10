import os
import tempfile
import unittest
from pathlib import Path


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise unittest.SkipTest("PyTorch CPU runtime is not installed.") from exc
    return torch


class CustomComponentsTrainingExampleTests(unittest.TestCase):
    def test_custom_component_spec_compiles_to_registered_parts(self):
        require_torch()
        import examples.custom_components  # noqa: F401
        from examples.custom_components_training import build_custom_detector_spec
        from simpledet.extensions import ENCODERS, HEADS, NECKS
        from simpledet.suite import compile_native_detector_plan

        spec = build_custom_detector_spec(num_classes=1)
        plan = compile_native_detector_plan(spec)

        self.assertEqual(plan.encoder.type, "ExampleTinyBackbone")
        self.assertEqual(plan.neck.type, "ExampleTinyNeck")
        self.assertEqual(plan.head.type, "ExampleTinyDenseHead")
        self.assertEqual(ENCODERS.lookup("custom_tiny_backbone").validation_status, "runtime_validated")
        self.assertEqual(NECKS.lookup("custom_tiny_neck").validation_status, "runtime_validated")
        self.assertEqual(HEADS.lookup("custom_tiny_head").validation_status, "runtime_validated")

    @unittest.skipUnless(
        os.environ.get("SIMPLEDET_RUN_CUSTOM_COMPONENT_SMOKE") == "1",
        "set SIMPLEDET_RUN_CUSTOM_COMPONENT_SMOKE=1 to run the custom component training smoke",
    )
    def test_custom_components_train_one_epoch_on_coco_format_data(self):
        require_torch()
        from examples.custom_components_training import (
            create_tiny_coco_dataset,
            run_custom_coco_training,
            summarize_run_result,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = create_tiny_coco_dataset(root / "dataset")
            result = run_custom_coco_training(dataset_root, workdir=root / "runs")
            summary = summarize_run_result(result)

            self.assertEqual(summary["stages"], ["build", "train", "test", "infer"])
            self.assertEqual(
                summary["custom_components"],
                {
                    "backbone": "ExampleTinyBackbone",
                    "neck": "ExampleTinyNeck",
                    "head": "ExampleTinyDenseHead",
                },
            )
            self.assertTrue(Path(summary["checkpoint_path"]).exists())
            self.assertTrue(Path(summary["manifest_path"]).exists())
            self.assertGreaterEqual(summary["test_predictions"], 1)
            self.assertGreaterEqual(summary["infer_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
