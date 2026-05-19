from collections import OrderedDict
import unittest

from native_tensor_contracts import require_torch


_ROI_BBOX_HEAD_CASES = (
    {"name": "Shared2FCBBoxHead", "alias": "shared_2fc_bbox_head"},
    {"name": "ConvFCBBoxHead", "alias": "convfc_bbox_head"},
    {"name": "DoubleConvFCBBoxHead", "alias": "double_convfc_bbox_head"},
    {"name": "DynamicBBoxHead", "alias": "dynamic_bbox_head"},
    {"name": "CascadeBBoxHead", "alias": "cascade_bbox_head"},
    {"name": "SABLHead", "alias": "sabl_bbox_head"},
    {"name": "SparseRoIHead", "alias": "sparse_roi_head", "sparse": True},
)


class NativeRoIPrimitiveTests(unittest.TestCase):
    def test_list_heads_roi_includes_common_bbox_aliases(self):
        require_torch()

        from simpledet.suite import list_heads

        roi_heads = set(list_heads(kind="roi"))

        for case in _ROI_BBOX_HEAD_CASES:
            with self.subTest(alias=case["alias"]):
                self.assertIn(case["alias"], roi_heads)

    def test_sabl_detector_default_still_uses_dense_atss_head(self):
        from simpledet.suite import build_detector, compile_native_detector_plan

        spec = build_detector("sabl", num_classes=3, encoder="resnet18.a1_in1k")
        plan = compile_native_detector_plan(spec)

        self.assertEqual(plan.family, "dense")
        self.assertEqual(plan.head.type, "ATSSHead")

    def test_build_head_shared_2fc_bbox_returns_roi_scores_and_deltas(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.suite import build_head

        head = build_head(name="shared_2fc_bbox_head", num_classes=20, in_channels=256)
        head.eval()
        roi_features = torch.full((2, 256, 7, 7), 0.25, dtype=torch.float32)

        with torch.no_grad():
            outputs = head(roi_features)

        self.assertEqual(head.native_head_spec.name, "Shared2FCBBoxHead")
        self.assertEqual(tuple(outputs["cls_score"].shape), (2, 21))
        self.assertEqual(tuple(outputs["bbox_pred"].shape), (2, 84))

    def test_roi_bbox_heads_construct_and_match_forward_shapes(self):
        torch = require_torch()
        torch.manual_seed(0)

        for case in _ROI_BBOX_HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, head_spec = self._build_roi_bbox_head(case)
                head.eval()
                roi_features = self._roi_features_for_case(torch, case)

                with torch.no_grad():
                    outputs = head(roi_features)

                self.assertEqual(head_spec.name, case["name"])
                self.assertEqual(head_spec.num_classes, 3)
                self.assertEqual(tuple(outputs["cls_score"].shape), (4, 4))
                self.assertEqual(tuple(outputs["bbox_pred"].shape), (4, 16))
                if case["name"] == "SABLHead":
                    self.assertEqual(outputs["side_confidence"].shape[0], 4)

    def test_roi_bbox_head_targets_and_loss_smoke(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.suite import build_head

        head = build_head(
            name="shared_2fc_bbox_head",
            num_classes=3,
            in_channels=8,
            roi_feat_size=2,
            fc_out_channels=16,
        )
        roi_features = torch.full((2, 8, 2, 2), 0.25, dtype=torch.float32)
        proposals = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        targets = head.get_targets(
            proposals,
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            torch.tensor([2]),
            num_samples=None,
        )
        outputs = head(roi_features)

        losses = head.loss(outputs, targets)

        self.assertEqual(set(losses), {"loss_cls", "loss_bbox", "loss_total"})
        self.assertTrue(bool(torch.isfinite(losses["loss_total"])))

    def test_class_agnostic_roi_bbox_head_rejects_class_specific_target_shape(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.suite import build_head

        head = build_head(
            name="shared_2fc_bbox_head",
            num_classes=3,
            in_channels=8,
            roi_feat_size=2,
            fc_out_channels=16,
            reg_class_agnostic=True,
        )
        outputs = head(torch.full((2, 8, 2, 2), 0.25, dtype=torch.float32))
        bad_targets = {
            "labels": torch.tensor([2, 0], dtype=torch.long),
            "label_weights": torch.ones((2,), dtype=torch.float32),
            "bbox_targets": torch.zeros((2, 16), dtype=torch.float32),
            "bbox_weights": torch.ones((2, 16), dtype=torch.float32),
        }

        with self.assertRaisesRegex(ValueError, "class-agnostic regression expects bbox_targets"):
            head.loss(outputs, bad_targets)

    def test_batch_roi_proposals_preserves_empty_images(self):
        torch = require_torch()

        from simpledet.native.roi import batch_roi_proposals

        proposals = [
            torch.tensor([[0.0, 0.0, 8.0, 8.0], [4.0, 4.0, 12.0, 12.0]]),
            torch.empty((0, 4), dtype=torch.float32),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        ]

        batched = batch_roi_proposals(proposals, batch_size=3)

        self.assertEqual(batched.counts, (2, 0, 1))
        self.assertEqual(tuple(batched.proposals[1].shape), (0, 4))
        self.assertEqual(tuple(batched.rois.shape), (3, 5))
        torch.testing.assert_close(batched.rois[:, 0], torch.tensor([0.0, 0.0, 2.0]))
        torch.testing.assert_close(batched.image_indices, torch.tensor([0, 0, 2]))

    def test_roi_align_features_returns_empty_tensor_without_pool_call(self):
        torch = require_torch()

        from simpledet.native.roi import roi_align_features

        class _Pool:
            output_size = (3, 3)

            def __call__(self, features, proposals, image_shapes):
                raise AssertionError("empty proposals should not call the ROI pool")

        features = OrderedDict([("0", torch.zeros((1, 5, 4, 4), dtype=torch.float32))])
        pooled = roi_align_features(
            _Pool(),
            features,
            [torch.empty((0, 4), dtype=torch.float32)],
            [(16, 16)],
        )

        self.assertEqual(tuple(pooled.shape), (0, 5, 3, 3))

    def test_bbox_targets_cover_positive_background_and_empty_edges(self):
        torch = require_torch()

        from simpledet.native.roi import build_roi_bbox_targets

        proposals = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        targets = build_roi_bbox_targets(
            proposals,
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            torch.tensor([2]),
            num_samples=None,
        )

        self.assertEqual(targets.labels.tolist(), [2, 0])
        torch.testing.assert_close(targets.bbox_targets[0], torch.zeros(4))
        torch.testing.assert_close(targets.bbox_weights[0], torch.ones(4))
        torch.testing.assert_close(targets.bbox_weights[1], torch.zeros(4))
        self.assertEqual(targets.positive_indices.tolist(), [0])
        self.assertEqual(targets.negative_indices.tolist(), [1])

        empty = build_roi_bbox_targets(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
        self.assertEqual(tuple(empty.proposals.shape), (0, 4))
        self.assertEqual(tuple(empty.labels.shape), (0,))
        self.assertEqual(tuple(empty.bbox_targets.shape), (0, 4))

    def test_mask_cascade_and_grid_targets_handle_positive_and_empty_rois(self):
        torch = require_torch()

        from simpledet.native.roi import (
            build_roi_grid_targets,
            build_roi_mask_targets,
            refine_cascade_stage_proposals,
        )

        proposals = torch.tensor([[1.0, 1.0, 5.0, 5.0]])
        mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        mask[:, 1:5, 1:5] = 1.0

        masks = build_roi_mask_targets(proposals, mask, torch.tensor([1]), output_size=4)
        self.assertEqual(tuple(masks.mask_targets.shape), (1, 4, 4))
        self.assertGreater(float(masks.mask_targets.mean()), 0.9)

        refined = refine_cascade_stage_proposals(
            proposals,
            torch.zeros((1, 3, 4), dtype=torch.float32),
            torch.tensor([2]),
            image_shape=(6, 6),
        )
        torch.testing.assert_close(refined.proposals, proposals)

        grid = build_roi_grid_targets(proposals, proposals, grid_size=3)
        self.assertEqual(tuple(grid.points.shape), (1, 3, 3, 2))
        torch.testing.assert_close(grid.points[0, 0, 0], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(grid.points[0, -1, -1], torch.tensor([5.0, 5.0]))
        torch.testing.assert_close(grid.weights, torch.ones((1, 3, 3)))

        empty_grid = build_roi_grid_targets(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0, 4), dtype=torch.float32),
            grid_size=3,
        )
        self.assertEqual(tuple(empty_grid.points.shape), (0, 3, 3, 2))

    def _build_roi_bbox_head(self, case):
        from simpledet.native.heads import build_native_head

        params = {
            "roi_feat_size": 2,
            "fc_out_channels": 16,
            "conv_out_channels": 8,
        }
        plan = type("HeadPlan", (), {"type": case["alias"], "params": params})()
        return build_native_head(plan, out_channels=8, num_classes=3)

    def _roi_features_for_case(self, torch, case):
        if case.get("sparse"):
            return torch.full((4, 8), 0.25, dtype=torch.float32)
        return torch.full((4, 8, 2, 2), 0.25, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
