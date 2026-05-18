import unittest

from native_tensor_contracts import (
    assert_dense_head_output_contract,
    make_cpu_detector_smoke_batch,
    require_torch,
)


class _ShapeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class NativeTensorContractTests(unittest.TestCase):
    def test_cpu_detector_smoke_batch_builds_images_features_targets_and_metadata(self):
        require_torch()

        batch = make_cpu_detector_smoke_batch(
            batch_size=2,
            image_size=(32, 48),
            feature_channels=8,
            feature_shapes=((8, 12), (4, 6), (2, 3)),
            boxes_per_image=2,
            num_classes=3,
        )

        self.assertEqual(len(batch.images), 2)
        self.assertEqual(tuple(batch.images[0].shape), (3, 32, 48))
        self.assertEqual(len(batch.feature_maps), 3)
        self.assertEqual(tuple(batch.feature_maps[0].shape), (2, 8, 8, 12))
        self.assertEqual(tuple(batch.boxes[0].shape), (2, 4))
        self.assertEqual(tuple(batch.labels[0].shape), (2,))
        self.assertEqual(batch.metadata[0]["image_shape"], (32, 48))
        self.assertEqual(set(batch.targets[0]), {"boxes", "labels", "image_id", "area", "iscrowd"})

    def test_dense_head_receives_multiscale_tensors_and_returns_batched_outputs(self):
        torch = require_torch()

        from simpledet.native.heads import FCOSDenseHead

        batch = make_cpu_detector_smoke_batch(
            batch_size=2,
            feature_channels=8,
            feature_shapes=((8, 8), (4, 4), (2, 2)),
            num_classes=3,
        )
        head = FCOSDenseHead(in_channels=8, num_classes=3, num_convs=1)
        head.eval()

        with torch.no_grad():
            outputs = head(batch.feature_maps)

        assert_dense_head_output_contract(
            outputs,
            batch.feature_maps,
            batch_size=2,
            required_keys=("cls_logits", "bbox_regression", "centerness"),
        )
        self.assertEqual(tuple(outputs["cls_logits"][0].shape), (2, 3, 8, 8))
        self.assertEqual(tuple(outputs["bbox_regression"][0].shape), (2, 4, 8, 8))

    def test_dense_head_level_mismatch_fails_with_clear_assertion(self):
        feature_maps = [
            _ShapeTensor((2, 8, 8, 8)),
            _ShapeTensor((2, 8, 4, 4)),
            _ShapeTensor((2, 8, 2, 2)),
        ]
        outputs = {
            "cls_logits": [
                _ShapeTensor((2, 3, 8, 8)),
                _ShapeTensor((2, 3, 4, 4)),
                _ShapeTensor((2, 3, 2, 2)),
            ],
            "bbox_regression": [
                _ShapeTensor((2, 4, 8, 8)),
                _ShapeTensor((2, 4, 4, 4)),
            ],
        }

        with self.assertRaisesRegex(
            AssertionError,
            "bbox_regression.*level count mismatch: expected 3 levels from feature_maps, got 2",
        ):
            assert_dense_head_output_contract(outputs, feature_maps, batch_size=2)


if __name__ == "__main__":
    unittest.main()
