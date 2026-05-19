import unittest

from native_tensor_contracts import make_dummy_feature_maps, require_torch


_KEYPOINT_HEADS = (
    {
        "name": "CenterNetHead",
        "alias": "centernet_head",
        "required_keys": ("heatmap", "wh", "offset", "cls_logits", "bbox_regression", "centerness"),
        "channels": {
            "heatmap": 3,
            "wh": 2,
            "offset": 2,
            "cls_logits": 3,
            "bbox_regression": 4,
            "centerness": 1,
        },
    },
    {
        "name": "CornerNetHead",
        "alias": "cornernet_head",
        "required_keys": (
            "top_left_heatmap",
            "bottom_right_heatmap",
            "top_left_embedding",
            "bottom_right_embedding",
            "top_left_offset",
            "bottom_right_offset",
        ),
        "channels": {
            "top_left_heatmap": 3,
            "bottom_right_heatmap": 3,
            "top_left_embedding": 1,
            "bottom_right_embedding": 1,
            "top_left_offset": 2,
            "bottom_right_offset": 2,
        },
    },
)

_TRANSFORMER_HEADS = (
    ("DETRHead", "detr_head", {}),
    ("ConditionalDETRHead", "conditional_detr_head", {}),
    ("DABDETRHead", "dab_detr_head", {}),
    ("DeformableDETRHead", "deformable_detr_head", {"num_feature_levels": 2}),
    ("DINOHead", "dino_head", {"num_feature_levels": 2}),
)


class NativeKeypointTransformerHeadTests(unittest.TestCase):
    def test_keypoint_and_transformer_head_aliases_are_registered(self):
        require_torch()

        import simpledet.native.heads  # noqa: F401
        from simpledet.extensions import HEADS
        from simpledet.suite import list_heads

        dense_heads = set(list_heads(kind="dense"))
        transformer_heads = set(list_heads(kind="transformer"))

        for case in _KEYPOINT_HEADS:
            with self.subTest(alias=case["alias"]):
                self.assertIn(case["alias"], dense_heads)
                metadata = HEADS.lookup(case["alias"])
                self.assertEqual(metadata.name, case["name"])
                self.assertEqual(metadata.family, "dense")
                self.assertEqual(metadata.validation_status, "runtime_validated")
                self.assertIn("heatmap", " ".join(metadata.tensor_contracts))

        for expected_name, alias, _ in _TRANSFORMER_HEADS:
            with self.subTest(alias=alias):
                self.assertIn(alias, transformer_heads)
                metadata = HEADS.lookup(alias)
                self.assertEqual(metadata.name, expected_name)
                self.assertEqual(metadata.family, "transformer")
                self.assertEqual(metadata.validation_status, "runtime_validated")
                self.assertIn("class_box_predictions", metadata.tensor_contracts)

    def test_keypoint_heads_construct_and_forward_small_cpu_tensors(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.native.heads import build_native_head

        feature_maps = make_dummy_feature_maps(
            batch_size=1,
            channels=8,
            spatial_shapes=((2, 2), (1, 1)),
        )

        for case in _KEYPOINT_HEADS:
            with self.subTest(head=case["name"]):
                plan = type("HeadPlan", (), {"type": case["alias"], "params": {"num_convs": 1}})()
                head, head_spec = build_native_head(plan, out_channels=8, num_classes=3)
                head.eval()

                with torch.no_grad():
                    outputs = head(feature_maps)

                self.assertEqual(head_spec.name, case["name"])
                self.assertEqual(head_spec.num_classes, 3)
                self.assertEqual(head_spec.in_channels, 8)
                for key in case["required_keys"]:
                    self.assertIn(key, outputs)
                    self.assertEqual(len(outputs[key]), len(feature_maps))
                    for tensor, feature in zip(outputs[key], feature_maps):
                        self.assertEqual(tuple(tensor.shape[-2:]), tuple(feature.shape[-2:]))
                        self.assertEqual(int(tensor.shape[0]), 1)
                        self.assertEqual(int(tensor.shape[1]), case["channels"][key])

    def test_build_head_detr_head_returns_query_class_and_box_predictions(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.suite import build_head

        head = build_head(
            name="detr_head",
            num_classes=91,
            num_queries=100,
            hidden_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
        )
        feature_maps = make_dummy_feature_maps(batch_size=1, channels=16, spatial_shapes=((2, 2),))

        with torch.no_grad():
            outputs = head(feature_maps)

        self.assertEqual(head.native_head_spec.name, "DETRHead")
        self.assertEqual(tuple(outputs["pred_logits"].shape), (1, 100, 92))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (1, 100, 4))
        self.assertEqual(tuple(outputs["class_logits"].shape), (1, 100, 92))
        self.assertEqual(tuple(outputs["box_predictions"].shape), (1, 100, 4))

    def test_transformer_heads_construct_and_forward_small_cpu_tensors(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.suite import build_head

        feature_maps = make_dummy_feature_maps(
            batch_size=1,
            channels=8,
            spatial_shapes=((2, 2), (1, 1)),
        )

        for expected_name, alias, extra in _TRANSFORMER_HEADS:
            with self.subTest(head=expected_name):
                head = build_head(
                    name=alias,
                    num_classes=3,
                    in_channels=8,
                    num_queries=5,
                    hidden_dim=16,
                    num_heads=4,
                    num_encoder_layers=1,
                    num_decoder_layers=1,
                    dim_feedforward=32,
                    **extra,
                )
                head.eval()

                with torch.no_grad():
                    outputs = head(feature_maps)

                self.assertEqual(head.native_head_spec.name, expected_name)
                self.assertEqual(tuple(outputs["pred_logits"].shape), (1, 5, 4))
                self.assertEqual(tuple(outputs["pred_boxes"].shape), (1, 5, 4))
                self.assertTrue(bool(outputs["pred_boxes"].ge(0.0).all()))
                self.assertTrue(bool(outputs["pred_boxes"].le(1.0).all()))
                if expected_name in {"DABDETRHead", "DINOHead"}:
                    self.assertEqual(tuple(outputs.get("reference_points", outputs["pred_boxes"]).shape), (1, 5, 4))

    def test_transformer_heads_reject_incompatible_attention_config_before_forward(self):
        require_torch()

        from simpledet.suite import build_head

        for _, alias, extra in _TRANSFORMER_HEADS:
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ValueError, "hidden_dim.*divisible.*num_heads"):
                    build_head(
                        name=alias,
                        num_classes=3,
                        num_queries=5,
                        hidden_dim=10,
                        num_heads=4,
                        num_encoder_layers=1,
                        num_decoder_layers=1,
                        dim_feedforward=32,
                        **extra,
                    )


if __name__ == "__main__":
    unittest.main()
