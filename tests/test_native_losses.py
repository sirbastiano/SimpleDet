import unittest

from native_tensor_contracts import clear_native_runtime_state, require_torch


class NativeLossTests(unittest.TestCase):
    def setUp(self):
        clear_native_runtime_state()

    def test_loss_registry_exposes_common_detection_losses(self):
        require_torch()

        from simpledet.extensions import LOSSES
        import simpledet.native.losses  # noqa: F401

        expected = {
            "focal": "FocalLoss",
            "quality_focal": "QualityFocalLoss",
            "varifocal": "VarifocalLoss",
            "generalized_focal_distribution": "DistributionFocalLoss",
            "smooth_l1": "SmoothL1Loss",
            "l1": "L1Loss",
            "iou": "IoULoss",
            "giou": "GIoULoss",
            "diou": "DIoULoss",
            "ciou": "CIoULoss",
            "cross_entropy": "CrossEntropyLoss",
            "dice": "DiceLoss",
            "mask": "MaskLoss",
        }
        for alias, registered_name in expected.items():
            self.assertEqual(LOSSES.lookup(alias).name, registered_name)

    def test_common_losses_return_finite_scalars_and_propagate_gradients(self):
        torch = require_torch()

        from simpledet.extensions import LOSSES
        import simpledet.native.losses  # noqa: F401

        def dense_scores(name):
            prediction = torch.randn((5, 3), dtype=torch.float32, requires_grad=True)
            target = torch.rand((5, 3), dtype=torch.float32)
            return LOSSES.get(name)(), prediction, target

        def distribution_focal():
            prediction = torch.randn((6, 8), dtype=torch.float32, requires_grad=True)
            target = torch.tensor([0.0, 1.2, 2.7, 4.4, 6.1, 7.0], dtype=torch.float32)
            return LOSSES.get("dfl")(), prediction, target

        def regression(name):
            prediction = torch.randn((4, 4), dtype=torch.float32, requires_grad=True)
            target = torch.randn((4, 4), dtype=torch.float32)
            return LOSSES.get(name)(), prediction, target

        def boxes(name):
            prediction = torch.tensor(
                [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 4.0, 3.0]],
                dtype=torch.float32,
                requires_grad=True,
            )
            target = torch.tensor(
                [[0.5, 0.5, 2.5, 2.5], [0.0, 1.0, 3.0, 4.0]],
                dtype=torch.float32,
            )
            return LOSSES.get(name)(), prediction, target

        def cross_entropy():
            prediction = torch.randn((4, 3), dtype=torch.float32, requires_grad=True)
            target = torch.tensor([0, 2, 1, 0], dtype=torch.long)
            return LOSSES.get("cross_entropy")(), prediction, target

        def mask_style(name):
            prediction = torch.randn((2, 1, 4, 4), dtype=torch.float32, requires_grad=True)
            target = torch.rand((2, 1, 4, 4), dtype=torch.float32)
            return LOSSES.get(name)(), prediction, target

        cases = [
            dense_scores("focal"),
            dense_scores("quality_focal"),
            dense_scores("varifocal"),
            distribution_focal(),
            regression("smooth_l1"),
            regression("l1"),
            boxes("iou"),
            boxes("giou"),
            boxes("diou"),
            boxes("ciou"),
            cross_entropy(),
            mask_style("dice"),
            mask_style("mask"),
        ]
        for loss_fn, prediction, target in cases:
            loss = loss_fn(prediction, target)
            self.assertEqual(tuple(loss.shape), ())
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
            self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_incompatible_target_shape_raises_loss_contract_error(self):
        torch = require_torch()

        from simpledet.extensions import LOSSES
        from simpledet.native.losses import LossContractError

        prediction = torch.randn((2, 3), dtype=torch.float32)
        target = torch.randn((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(
            LossContractError,
            "VarifocalLoss expects prediction and target tensors with identical shapes",
        ):
            LOSSES.get("varifocal")()(prediction, target)

    def test_vfnet_head_can_call_registry_losses(self):
        torch = require_torch()

        from simpledet.extensions import LOSSES
        from simpledet.native.heads import VFNetHead

        head = VFNetHead(in_channels=4, num_classes=2, num_anchors=1, num_convs=1)
        self.assertIsInstance(head.loss_cls, LOSSES.get("varifocal"))
        self.assertIsInstance(head.loss_bbox, LOSSES.get("iou"))

        cls_logits = torch.randn((3, 2), dtype=torch.float32, requires_grad=True)
        cls_targets = torch.tensor([[0.8, 0.0], [0.0, 0.2], [0.0, 0.0]], dtype=torch.float32)
        pred_boxes = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 4.0, 3.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target_boxes = torch.tensor(
            [[0.5, 0.5, 2.5, 2.5], [0.0, 1.0, 3.0, 4.0]],
            dtype=torch.float32,
        )

        loss = head.loss_cls(cls_logits, cls_targets) + head.loss_bbox(pred_boxes, target_boxes)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(cls_logits.grad)
        self.assertIsNotNone(pred_boxes.grad)
        self.assertGreater(float(cls_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(pred_boxes.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
