from collections import OrderedDict
import unittest

from native_tensor_contracts import require_torch


class NativeRoIPrimitiveTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
