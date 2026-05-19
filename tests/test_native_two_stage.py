from collections import OrderedDict
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from native_tensor_contracts import make_cpu_detector_smoke_batch, require_torch


class NativeTwoStageDetectorTests(unittest.TestCase):
    def _components(self, torch):
        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.heads import RPNHead, Shared2FCBBoxHead

        class _Backbone(torch.nn.Module):
            def forward(self, images):
                feature = images.mean(dim=1, keepdim=True)
                feature = torch.nn.functional.adaptive_avg_pool2d(feature, (4, 4))
                return (feature.expand(-1, 4, -1, -1).contiguous(),)

        class _Neck(torch.nn.Module):
            def forward(self, features):
                return OrderedDict([("0", features[0])])

        return NativeModelComponents(
            plan=SimpleNamespace(architecture="faster_rcnn", family="roi", head=None),
            backbone=_Backbone(),
            backbone_spec=SimpleNamespace(feature_channels=(4,)),
            neck=_Neck(),
            neck_spec=SimpleNamespace(out_channels=4, num_outs=1),
            rpn_head=RPNHead(in_channels=4, num_classes=1, num_anchors=1, num_convs=1),
            bbox_head=Shared2FCBBoxHead(
                in_channels=4,
                num_classes=3,
                roi_feat_size=7,
                fc_out_channels=16,
            ),
        )

    def _fake_torchvision_modules(self, torch):
        class _FakeRoIAlign(torch.nn.Module):
            def __init__(self, featmap_names, output_size, sampling_ratio):
                super().__init__()
                self.featmap_names = tuple(featmap_names)
                self.output_size = int(output_size)
                self.sampling_ratio = int(sampling_ratio)

            def forward(self, features, proposals, image_shapes):
                reference = next(iter(features.values()))
                num_proposals = sum(int(proposal.shape[0]) for proposal in proposals)
                channels = int(reference.shape[1])
                value = reference.mean()
                return value.expand(
                    num_proposals,
                    channels,
                    self.output_size,
                    self.output_size,
                ).contiguous()

        fake_ops = ModuleType("torchvision.ops")
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = ModuleType("torchvision")
        fake_torchvision.ops = fake_ops
        return {
            "torchvision": fake_torchvision,
            "torchvision.ops": fake_ops,
        }

    def test_native_build_detector_faster_rcnn_runs_loss_and_prediction_paths(self):
        torch = require_torch()
        from simpledet.native.modeling import build_detector
        from simpledet.native.roi import TwoStageDetector

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(16, 16),
            boxes_per_image=1,
            num_classes=3,
        )
        components = self._components(torch)

        with patch.dict(sys.modules, self._fake_torchvision_modules(torch)), patch(
            "simpledet.native.assemblers.build_native_components",
            return_value=components,
        ):
            model = build_detector(name="faster_rcnn", num_classes=3, pretrained=False)

        self.assertIsInstance(model, TwoStageDetector)
        self.assertEqual(model.num_classes, 3)
        self.assertFalse(model.with_mask)
        self.assertEqual(model.rpn_head.__class__.__name__, "RPNHead")
        self.assertEqual(model.bbox_head.__class__.__name__, "Shared2FCBBoxHead")
        self.assertIsNone(model.proposal_head)
        self.assertIsNone(model.box_head)

        losses = model(batch.images, batch.targets)

        self.assertEqual(
            set(losses),
            {
                "loss_rpn_box_reg",
                "loss_rpn_objectness",
                "loss_roi_box_reg",
                "loss_roi_classifier",
                "loss_total",
            },
        )
        for loss in losses.values():
            self.assertEqual(tuple(loss.shape), ())
            self.assertTrue(bool(torch.isfinite(loss)))

        model.eval()
        with torch.no_grad():
            predictions = model(batch.images)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(set(predictions[0]), {"boxes", "scores", "labels"})
        self.assertEqual(predictions[0]["boxes"].dim(), 2)
        self.assertEqual(predictions[0]["boxes"].shape[-1], 4)
        self.assertEqual(predictions[0]["boxes"].shape[0], predictions[0]["scores"].shape[0])
        self.assertEqual(predictions[0]["boxes"].shape[0], predictions[0]["labels"].shape[0])

    def test_mask_rcnn_build_plan_requires_mask_head(self):
        from simpledet.suite import build_detector, build_head, compile_native_detector_plan

        spec = build_detector(
            "mask_rcnn",
            num_classes=3,
            head=build_head("Shared2FCBBoxHead", num_classes=3, with_mask=False),
            pretrained=False,
        )

        with self.assertRaisesRegex(ValueError, "mask.*head|with_mask"):
            compile_native_detector_plan(spec)


if __name__ == "__main__":
    unittest.main()
