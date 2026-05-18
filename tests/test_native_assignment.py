import unittest

from native_tensor_contracts import require_torch


class NativeAssignmentTests(unittest.TestCase):
    def test_atss_assignment_returns_positive_negative_and_ignored_labels(self):
        torch = require_torch()

        from simpledet.native.assignment import atss_assign
        from simpledet.native.geometry import build_feature_map_specs, generate_anchors

        specs = build_feature_map_specs(feature_sizes=((2, 2), (1, 1)), strides=(16, 32))
        anchors_per_level = generate_anchors(
            specs,
            base_sizes=(8, 16),
            aspect_ratios=(1.0,),
            scales=(1.0,),
            device="cpu",
            dtype=torch.float32,
        )
        anchors = torch.cat(anchors_per_level, dim=0)

        result = atss_assign(
            anchors,
            torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
            torch.tensor([2]),
            num_level_priors=tuple(level.shape[0] for level in anchors_per_level),
            topk=1,
            ignored_boxes=torch.tensor([[20.0, 4.0, 28.0, 12.0]]),
            ignore_iou_thr=0.5,
        )

        self.assertEqual(int(result.positive_mask.sum()), 1)
        self.assertEqual(int(result.negative_mask.sum()), 3)
        self.assertEqual(int(result.ignored_mask.sum()), 1)
        self.assertEqual(set(result.labels.tolist()), {-1, 0, 2})

    def test_no_ground_truth_assignments_are_all_background(self):
        torch = require_torch()

        from simpledet.native.assignment import atss_assign, max_iou_assign

        priors = torch.tensor(
            [
                [0.0, 0.0, 8.0, 8.0],
                [16.0, 16.0, 24.0, 24.0],
            ]
        )
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_labels = torch.empty((0,), dtype=torch.long)

        max_iou = max_iou_assign(priors, empty_boxes, empty_labels)
        atss = atss_assign(priors, empty_boxes, empty_labels, num_level_priors=(2,))

        for result in (max_iou, atss):
            self.assertFalse(bool(result.positive_mask.any()))
            self.assertFalse(bool(result.ignored_mask.any()))
            self.assertEqual(result.labels.tolist(), [0, 0])
            self.assertEqual(tuple(result.matched_boxes.shape), (2, 4))

    def test_atss_loss_accepts_images_without_ground_truth_boxes(self):
        torch = require_torch()

        from simpledet.native.dense_ops import DenseATSSLoss

        image = torch.zeros((3, 32, 32), dtype=torch.float32)
        feature_maps = [
            torch.zeros((1, 4, 2, 2), dtype=torch.float32),
            torch.zeros((1, 4, 1, 1), dtype=torch.float32),
        ]
        head_outputs = {
            "cls_logits": [
                torch.zeros((1, 18, 2, 2), dtype=torch.float32),
                torch.zeros((1, 18, 1, 1), dtype=torch.float32),
            ],
            "bbox_regression": [
                torch.zeros((1, 36, 2, 2), dtype=torch.float32),
                torch.zeros((1, 36, 1, 1), dtype=torch.float32),
            ],
            "centerness": [
                torch.zeros((1, 9, 2, 2), dtype=torch.float32),
                torch.zeros((1, 9, 1, 1), dtype=torch.float32),
            ],
        }
        target = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
        }

        losses = DenseATSSLoss()([image], [target], [feature_maps], [head_outputs])

        self.assertTrue(bool(torch.isfinite(losses["loss_total"])))
        self.assertEqual(float(losses["loss_bbox"]), 0.0)
        self.assertEqual(float(losses["loss_centerness"]), 0.0)

    def test_point_and_center_region_assignment_select_smallest_box(self):
        torch = require_torch()

        from simpledet.native.assignment import center_region_assign, point_assign

        points = torch.tensor([[5.0, 5.0], [9.0, 9.0], [15.0, 15.0], [30.0, 30.0]])
        gt_boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [8.0, 8.0, 12.0, 12.0]])
        gt_labels = torch.tensor([1, 2])

        point_result = point_assign(points, gt_boxes, gt_labels)
        center_result = center_region_assign(points, gt_boxes, gt_labels, center_radius=0.5)

        self.assertEqual(point_result.labels.tolist(), [1, 2, 1, 0])
        self.assertEqual(center_result.labels.tolist(), [1, 2, 1, 0])

    def test_task_aligned_and_sim_ota_assigners_match_scored_boxes(self):
        torch = require_torch()

        from simpledet.native.assignment import sim_ota_assign, task_aligned_assign

        priors = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        )
        pred_boxes = priors.clone()
        pred_scores = torch.tensor([[0.9, 0.1], [0.1, 0.8], [0.2, 0.2]])
        gt_boxes = priors[:2].clone()
        gt_labels = torch.tensor([1, 2])

        task_aligned = task_aligned_assign(priors, pred_scores, pred_boxes, gt_boxes, gt_labels, topk=1)
        sim_ota = sim_ota_assign(priors, pred_scores, pred_boxes, gt_boxes, gt_labels, candidate_topk=1)

        self.assertEqual(task_aligned.labels.tolist(), [1, 2, 0])
        self.assertEqual(sim_ota.labels.tolist(), [1, 2, 0])

    def test_hungarian_assignment_is_one_to_one(self):
        torch = require_torch()

        from simpledet.native.assignment import hungarian_assign

        pred_logits = torch.tensor(
            [
                [8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0],
                [0.0, 8.0, 0.0],
            ]
        )
        pred_boxes = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.8, 0.8, 0.9, 0.9],
                [0.5, 0.5, 0.6, 0.6],
            ]
        )
        gt_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.6, 0.6]])
        gt_labels = torch.tensor([1, 2])

        result = hungarian_assign(pred_logits, pred_boxes, gt_boxes, gt_labels)

        self.assertEqual(result.labels.tolist(), [1, 0, 2])
        self.assertEqual(int(result.positive_mask.sum()), 2)

    def test_balanced_sampler_skips_ignored_assignments(self):
        torch = require_torch()

        from simpledet.native.assignment import atss_assign, sample_assignment
        from simpledet.native.geometry import build_feature_map_specs, generate_anchors

        specs = build_feature_map_specs(feature_sizes=((2, 2),), strides=(16,))
        anchors_per_level = generate_anchors(
            specs,
            base_sizes=(8,),
            aspect_ratios=(1.0,),
            scales=(1.0,),
            device="cpu",
            dtype=torch.float32,
        )
        anchors = torch.cat(anchors_per_level, dim=0)
        assignment = atss_assign(
            anchors,
            torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
            torch.tensor([1]),
            num_level_priors=(4,),
            topk=1,
            ignored_boxes=torch.tensor([[20.0, 4.0, 28.0, 12.0]]),
            ignore_iou_thr=0.5,
        )

        sample = sample_assignment(assignment, num_samples=2, positive_fraction=0.5)

        self.assertEqual(sample.positive_indices.tolist(), [0])
        self.assertEqual(sample.negative_indices.numel(), 1)
        self.assertEqual(sample.ignored_indices.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
