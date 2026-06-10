import unittest

from native_tensor_contracts import require_torch


class NativeGeometryTests(unittest.TestCase):
    def test_anchor_and_point_priors_share_feature_specs(self):
        torch = require_torch()

        from simpledet.native.geometry import build_feature_map_specs, generate_anchors, generate_points

        specs = build_feature_map_specs(
            feature_sizes=((2, 3), (1, 1)),
            strides=(8, (16, 16)),
        )

        anchors = generate_anchors(specs, base_sizes=(32, 64), device="cpu", dtype=torch.float32)
        points = generate_points(specs, device="cpu", dtype=torch.float32)

        self.assertEqual(tuple(anchors[0].shape), (2 * 3 * 9, 4))
        self.assertEqual(tuple(anchors[1].shape), (1 * 1 * 9, 4))
        self.assertEqual(tuple(points[0].shape), (2 * 3, 2))
        torch.testing.assert_close(points[0][0], torch.tensor([4.0, 4.0]))
        torch.testing.assert_close(points[0][1], torch.tensor([12.0, 4.0]))
        torch.testing.assert_close(points[0][3], torch.tensor([4.0, 12.0]))

    def test_bbox_delta_round_trip_and_iou(self):
        torch = require_torch()

        from simpledet.native.geometry import box_iou, decode_boxes, encode_boxes

        anchors = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 30.0, 50.0],
            ]
        )
        targets = torch.tensor(
            [
                [1.0, 2.0, 11.0, 14.0],
                [8.0, 12.0, 34.0, 44.0],
            ]
        )

        deltas = encode_boxes(anchors, targets)
        decoded = decode_boxes(anchors, deltas)

        torch.testing.assert_close(decoded, targets)
        overlaps = box_iou(anchors, targets)
        self.assertEqual(tuple(overlaps.shape), (2, 2))
        torch.testing.assert_close(overlaps[0, 0], torch.tensor(72.0 / 148.0))
        self.assertEqual(float(overlaps[0, 1]), 0.0)

    def test_point_box_round_trip_clip_scale_and_payload(self):
        torch = require_torch()

        from simpledet.native.geometry import (
            clip_boxes_to_image,
            decode_point_boxes,
            encode_point_boxes,
            make_batched_nms_payload,
            scale_boxes,
        )

        points = torch.tensor([[4.0, 4.0], [12.0, 8.0]])
        boxes = torch.tensor([[1.0, 2.0, 9.0, 12.0], [10.0, -2.0, 20.0, 10.0]])

        distances = encode_point_boxes(points, boxes)
        decoded = decode_point_boxes(points, distances)
        clipped = clip_boxes_to_image(decoded, (10, 16))
        scaled = scale_boxes(clipped, (0.5, 2.0))
        payload = make_batched_nms_payload(
            scaled,
            torch.tensor([0.8, 0.7]),
            torch.tensor([2, 2]),
            image_indices=torch.tensor([0, 1]),
        )

        torch.testing.assert_close(decoded[0], boxes[0])
        torch.testing.assert_close(clipped[1], torch.tensor([10.0, 0.0, 16.0, 10.0]))
        torch.testing.assert_close(scaled[1], torch.tensor([20.0, 0.0, 32.0, 5.0]))
        torch.testing.assert_close(payload["nms_indices"], torch.tensor([2, 5]))

    def test_empty_inputs_keep_tensor_shapes(self):
        torch = require_torch()

        from simpledet.native.geometry import box_iou, decode_boxes, encode_boxes, make_batched_nms_payload

        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        one_box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        self.assertEqual(tuple(box_iou(empty_boxes, one_box).shape), (0, 1))
        self.assertEqual(tuple(encode_boxes(empty_boxes, empty_boxes).shape), (0, 4))
        self.assertEqual(tuple(decode_boxes(empty_boxes, empty_boxes).shape), (0, 4))

        payload = make_batched_nms_payload(
            empty_boxes,
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
        self.assertEqual(tuple(payload["boxes"].shape), (0, 4))
        self.assertEqual(tuple(payload["scores"].shape), (0,))
        self.assertEqual(tuple(payload["labels"].shape), (0,))
        self.assertEqual(tuple(payload["nms_indices"].shape), (0,))

    def test_invalid_feature_sizes_and_strides_raise_before_generation(self):
        require_torch()

        from simpledet.native.geometry import FeatureMapSpec, build_feature_map_specs

        with self.assertRaisesRegex(ValueError, "feature height must be positive"):
            FeatureMapSpec(height=0, width=2, stride_y=4, stride_x=4)

        with self.assertRaisesRegex(ValueError, "stride"):
            build_feature_map_specs(feature_sizes=((2, 2),), strides=(0,))

        with self.assertRaisesRegex(ValueError, "stride count mismatch"):
            build_feature_map_specs(feature_sizes=((2, 2), (1, 1)), strides=(8,))


if __name__ == "__main__":
    unittest.main()
