from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from native_tensor_contracts import clear_native_runtime_state, make_dummy_targets, require_torch


class NativeSingleStageDetectorTests(unittest.TestCase):
    def setUp(self):
        clear_native_runtime_state()

    def _make_detector(self, *, num_classes=3):
        torch = require_torch()
        from simpledet.native.modeling import SingleStageDetector

        class _Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_shapes = []

            def forward(self, images):
                self.seen_shapes.append(tuple(images.shape))
                return (images.mean(dim=1, keepdim=True),)

        class _Neck(torch.nn.Module):
            def forward(self, features):
                return OrderedDict([("0", features[0])])

        class _Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_feature_keys = []

            def forward(self, features):
                self.seen_feature_keys.append(tuple(features.keys()))
                feature = features["0"]
                batch_size, _, height, width = feature.shape
                return {
                    "cls_logits": [
                        feature.new_full((batch_size, 1, num_classes, height, width), 0.25)
                    ],
                    "bbox_regression": [feature.new_zeros((batch_size, 1, 4, height, width))],
                }

        class _Loss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled = None
                self.received = None

            def forward(self, images, targets, feature_pyramids, head_outputs):
                self.grad_enabled = torch.is_grad_enabled()
                self.received = (images, targets, feature_pyramids, head_outputs)
                return {
                    "loss_cls": head_outputs[0]["cls_logits"][0].sum(),
                    "loss_bbox": head_outputs[0]["bbox_regression"][0].sum(),
                }

        class _Postprocessor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled = None
                self.training_states = []

            def forward(self, image, feature_pyramid, head_outputs):
                self.grad_enabled = torch.is_grad_enabled()
                self.training_states.append(self.training)
                return {
                    "boxes": image.new_zeros((1, 4)),
                    "scores": image.new_ones((1,)),
                    "labels": image.new_ones((1,), dtype=torch.long),
                }

        return SingleStageDetector(
            backbone=_Backbone(),
            neck=_Neck(),
            head=_Head(),
            backbone_spec=SimpleNamespace(feature_channels=(1,)),
            neck_spec=SimpleNamespace(out_channels=1),
            head_spec=SimpleNamespace(name="RetinaHead", num_classes=num_classes),
            loss_fn=_Loss(),
            postprocessor=_Postprocessor(),
        )

    def _images(self, torch, *, count=2):
        return [
            torch.full((3, 8, 8), fill_value=float(index + 1) / 10.0, dtype=torch.float32)
            for index in range(count)
        ]

    def test_single_stage_detector_owns_core_modules(self):
        require_torch()
        detector = self._make_detector()

        child_modules = dict(detector.named_children())

        self.assertIs(child_modules["backbone"], detector.backbone)
        self.assertIs(child_modules["neck"], detector.neck)
        self.assertIs(child_modules["head"], detector.head)
        self.assertIs(child_modules["loss_fn"], detector.loss_fn)
        self.assertIs(child_modules["postprocessor"], detector.postprocessor)

    def test_forward_loss_routes_images_targets_and_head_outputs(self):
        torch = require_torch()
        detector = self._make_detector()
        images = self._images(torch)
        boxes = [torch.tensor([[1.0, 1.0, 4.0, 4.0]]) for _ in images]
        labels = [torch.tensor([1], dtype=torch.long) for _ in images]
        targets = make_dummy_targets(
            boxes=boxes,
            labels=labels,
            metadata=[{"image_id": index} for index in range(len(images))],
        )

        losses = detector.forward_loss(images, targets)

        self.assertEqual(set(losses), {"loss_cls", "loss_bbox"})
        self.assertTrue(detector.loss_fn.grad_enabled)
        received_images, received_targets, feature_pyramids, head_outputs = detector.loss_fn.received
        self.assertIs(received_images, images)
        self.assertIs(received_targets, targets)
        self.assertEqual(len(feature_pyramids), len(images))
        self.assertEqual(len(head_outputs), len(images))
        self.assertEqual(detector.backbone.seen_shapes, [(1, 3, 8, 8), (1, 3, 8, 8)])
        self.assertEqual(detector.head.seen_feature_keys, [("0",), ("0",)])

    def test_predict_uses_eval_mode_and_no_grad(self):
        torch = require_torch()
        detector = self._make_detector()
        detector.eval()

        predictions = detector.predict(self._images(torch, count=1))

        self.assertFalse(detector.postprocessor.grad_enabled)
        self.assertEqual(detector.postprocessor.training_states, [False])
        self.assertFalse(detector.training)
        self.assertEqual(set(predictions[0]), {"boxes", "scores", "labels"})
        self.assertFalse(predictions[0]["boxes"].requires_grad)

    def test_native_build_detector_defaults_retinanet_to_single_stage_model(self):
        require_torch()
        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.modeling import SingleStageDetector, build_detector

        detector = self._make_detector()
        components = NativeModelComponents(
            plan=SimpleNamespace(
                architecture="retinanet",
                family="dense",
                head=SimpleNamespace(type="RetinaHead"),
            ),
            backbone=detector.backbone,
            backbone_spec=detector.backbone_spec,
            neck=detector.neck,
            neck_spec=detector.neck_spec,
            head=detector.head,
            head_spec=detector.head_spec,
        )

        with patch("simpledet.native.assemblers.build_native_components", return_value=components):
            model = build_detector(name="retinanet", num_classes=3, pretrained=False)

        self.assertIsInstance(model, SingleStageDetector)
        self.assertEqual(model.head_spec.num_classes, 3)

    def test_single_stage_detector_rejects_roi_only_head_plan(self):
        require_torch()
        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.modeling import build_native_model
        from simpledet.suite import build_detector, build_head

        detector = self._make_detector()
        spec = build_detector(
            "retinanet",
            num_classes=3,
            head=build_head("Shared2FCBBoxHead", num_classes=3),
        )
        components = NativeModelComponents(
            plan=SimpleNamespace(
                architecture="retinanet",
                family="dense",
                head=SimpleNamespace(type="Shared2FCBBoxHead"),
            ),
            backbone=detector.backbone,
            backbone_spec=detector.backbone_spec,
            neck=detector.neck,
            neck_spec=detector.neck_spec,
            head=detector.head,
            head_spec=detector.head_spec,
        )

        with patch("simpledet.native.assemblers.build_native_components", return_value=components):
            with self.assertRaisesRegex(ValueError, "dense head|single-stage"):
                build_native_model("retinanet", num_classes=3, detector_spec=spec)


if __name__ == "__main__":
    unittest.main()
