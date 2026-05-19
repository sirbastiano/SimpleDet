from types import SimpleNamespace
import unittest
from unittest.mock import patch

from native_tensor_contracts import make_cpu_detector_smoke_batch, require_torch


_QUERY_DETECTOR_CASES = (
    ("detr", "DETRHead", {}),
    ("conditional_detr", "ConditionalDETRHead", {}),
    ("dab_detr", "DABDETRHead", {}),
    ("deformable_detr", "DeformableDETRHead", {"num_feature_levels": 2}),
    ("dino", "DINOHead", {"num_feature_levels": 2}),
)


class NativeQueryDetectorTests(unittest.TestCase):
    def _components(self, torch, *, architecture, head_name, head_extra=None, overrides=None):
        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.heads import build_native_head
        from simpledet.suite.native_plan import ComponentPlan

        class _Backbone(torch.nn.Module):
            def forward(self, images):
                base = images.mean(dim=1, keepdim=True)
                level0 = torch.nn.functional.adaptive_avg_pool2d(base, (4, 4))
                level0 = level0.expand(-1, 8, -1, -1).contiguous()
                level1 = torch.nn.functional.adaptive_avg_pool2d(level0, (2, 2))
                return (level0, level1)

        class _Neck(torch.nn.Module):
            def forward(self, features):
                return tuple(features)

        params = {
            "num_queries": 5,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
        }
        params.update(dict(head_extra or {}))
        head, head_spec = build_native_head(
            ComponentPlan(kind="head", type=head_name, params=params),
            out_channels=8,
            num_classes=3,
        )
        return NativeModelComponents(
            plan=SimpleNamespace(
                architecture=architecture,
                family="transformer",
                head=SimpleNamespace(type=head_name, params=params),
                overrides=dict(overrides or {}),
            ),
            backbone=_Backbone(),
            backbone_spec=SimpleNamespace(feature_channels=(8, 8)),
            neck=_Neck(),
            neck_spec=SimpleNamespace(out_channels=8, num_outs=2),
            head=head,
            head_spec=head_spec,
        )

    def test_query_detectors_run_small_cpu_loss_and_prediction_paths(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.native.modeling import QueryDetector, build_detector

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(16, 16),
            boxes_per_image=1,
            num_classes=3,
        )

        for architecture, head_name, head_extra in _QUERY_DETECTOR_CASES:
            with self.subTest(architecture=architecture):
                components = self._components(
                    torch,
                    architecture=architecture,
                    head_name=head_name,
                    head_extra=head_extra,
                )
                with patch(
                    "simpledet.native.assemblers.build_native_components",
                    return_value=components,
                ):
                    model = build_detector(
                        name=architecture,
                        num_classes=3,
                        num_queries=5,
                        pretrained=False,
                    )

                self.assertIsInstance(model, QueryDetector)
                self.assertEqual(model.head_spec.name, head_name)

                losses = model(batch.images, batch.targets)
                self.assertEqual(set(losses), {"loss_cls", "loss_bbox", "loss_total"})
                for loss in losses.values():
                    self.assertEqual(tuple(loss.shape), ())
                    self.assertTrue(bool(torch.isfinite(loss)))

                model.eval()
                predictions = model(batch.images)

                self.assertEqual(len(predictions), 1)
                prediction = predictions[0]
                self.assertEqual(tuple(prediction["pred_logits"].shape), (1, 5, 4))
                self.assertEqual(tuple(prediction["pred_boxes"].shape), (1, 5, 4))
                self.assertTrue(bool(prediction["pred_boxes"].ge(0.0).all()))
                self.assertTrue(bool(prediction["pred_boxes"].le(1.0).all()))
                self.assertIn("boxes", prediction)
                self.assertIn("scores", prediction)
                self.assertIn("labels", prediction)

    def test_query_detector_construction_rejects_missing_position_settings(self):
        torch = require_torch()

        from simpledet.native.modeling import build_detector

        components = self._components(
            torch,
            architecture="detr",
            head_name="DETRHead",
            overrides={"positional_encoding": {}},
        )
        with patch(
            "simpledet.native.assemblers.build_native_components",
            return_value=components,
        ):
            with self.assertRaisesRegex(ValueError, "positional_encoding.num_feats"):
                build_detector(name="detr", num_classes=3, pretrained=False)

    def test_transformer_build_plans_default_to_query_heads(self):
        from simpledet.suite import build_detector, compile_native_detector_plan

        expected_heads = {
            "detr": "DETRHead",
            "conditional_detr": "ConditionalDETRHead",
            "dab_detr": "DABDETRHead",
            "deformable_detr": "DeformableDETRHead",
            "dino": "DINOHead",
        }
        for architecture, expected_head in expected_heads.items():
            with self.subTest(architecture=architecture):
                spec = build_detector(
                    architecture,
                    num_classes=3,
                    encoder="resnet18.a1_in1k",
                    num_queries=20,
                    hidden_dim=16,
                    num_heads=4,
                    num_encoder_layers=1,
                    num_decoder_layers=1,
                    dim_feedforward=32,
                )
                plan = compile_native_detector_plan(spec)

                self.assertEqual(plan.family, "transformer")
                self.assertEqual(plan.head.type, expected_head)
                self.assertEqual(plan.head.params["num_classes"], 3)
                self.assertEqual(plan.head.params["num_queries"], 20)
                self.assertEqual(plan.head.params["hidden_dim"], 16)


if __name__ == "__main__":
    unittest.main()
