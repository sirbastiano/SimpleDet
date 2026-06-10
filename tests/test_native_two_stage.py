from collections import OrderedDict
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from native_tensor_contracts import clear_native_runtime_state, make_cpu_detector_smoke_batch, require_torch


class NativeTwoStageDetectorTests(unittest.TestCase):
    def setUp(self):
        clear_native_runtime_state()

    def _components(self, torch, architecture="faster_rcnn"):
        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.heads import (
            CascadeBBoxHead,
            CascadeMaskHead,
            DoubleConvFCBBoxHead,
            DynamicBBoxHead,
            FCNMaskHead,
            GridHead,
            RPNHead,
            Shared2FCBBoxHead,
            SparseRoIHead,
        )

        class _Backbone(torch.nn.Module):
            def forward(self, images):
                feature = images.mean(dim=1, keepdim=True)
                feature = torch.nn.functional.adaptive_avg_pool2d(feature, (4, 4))
                return (feature.expand(-1, 4, -1, -1).contiguous(),)

        class _Neck(torch.nn.Module):
            def forward(self, features):
                return OrderedDict([("0", features[0])])

        bbox_head_cls = {
            "cascade_rcnn": CascadeBBoxHead,
            "cascade_mask_rcnn": CascadeBBoxHead,
            "double_head_rcnn": DoubleConvFCBBoxHead,
            "dynamic_rcnn": DynamicBBoxHead,
            "sparse_rcnn": SparseRoIHead,
        }.get(architecture, Shared2FCBBoxHead)
        bbox_head = bbox_head_cls(
            in_channels=4,
            num_classes=3,
            roi_feat_size=7,
            fc_out_channels=16,
            conv_out_channels=8,
        )
        rpn_head = None
        if architecture not in {"fast_rcnn", "sparse_rcnn"}:
            rpn_head = RPNHead(in_channels=4, num_classes=1, num_anchors=1, num_convs=1)
        mask_head = None
        if architecture == "mask_rcnn":
            mask_head = FCNMaskHead(
                in_channels=4,
                num_classes=3,
                roi_feat_size=14,
                conv_out_channels=8,
            )
        if architecture == "cascade_mask_rcnn":
            mask_head = CascadeMaskHead(
                in_channels=4,
                num_classes=3,
                roi_feat_size=14,
                conv_out_channels=8,
            )
        grid_head = None
        if architecture == "grid_rcnn":
            grid_head = GridHead(
                in_channels=4,
                num_classes=3,
                roi_feat_size=14,
                conv_out_channels=8,
                grid_size=7,
            )
        return NativeModelComponents(
            plan=SimpleNamespace(architecture=architecture, family="roi", head=None),
            backbone=_Backbone(),
            backbone_spec=SimpleNamespace(feature_channels=(4,)),
            neck=_Neck(),
            neck_spec=SimpleNamespace(out_channels=4, num_outs=1),
            rpn_head=rpn_head,
            rpn_head_spec=SimpleNamespace(name="RPNHead") if rpn_head is not None else None,
            bbox_head=bbox_head,
            bbox_head_spec=SimpleNamespace(name=bbox_head.__class__.__name__),
            head=bbox_head,
            head_spec=SimpleNamespace(name=bbox_head.__class__.__name__),
            mask_head=mask_head,
            mask_head_spec=SimpleNamespace(name=mask_head.__class__.__name__) if mask_head is not None else None,
            grid_head=grid_head,
            grid_head_spec=SimpleNamespace(name="GridHead") if grid_head is not None else None,
        )

    def _rpn_components(self, torch):
        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.heads import RPNHead

        class _Backbone(torch.nn.Module):
            def forward(self, images):
                feature = images.mean(dim=1, keepdim=True)
                feature = torch.nn.functional.adaptive_avg_pool2d(feature, (4, 4))
                return (feature.expand(-1, 4, -1, -1).contiguous(),)

        class _Neck(torch.nn.Module):
            def forward(self, features):
                return (features[0],)

        rpn_head = RPNHead(in_channels=4, num_classes=1, num_anchors=1, num_convs=1)
        return NativeModelComponents(
            plan=SimpleNamespace(architecture="rpn", family="proposal", head=None),
            backbone=_Backbone(),
            backbone_spec=SimpleNamespace(feature_channels=(4,)),
            neck=_Neck(),
            neck_spec=SimpleNamespace(out_channels=4, num_outs=1),
            head=rpn_head,
            head_spec=SimpleNamespace(name="RPNHead"),
            rpn_head=rpn_head,
            rpn_head_spec=SimpleNamespace(name="RPNHead"),
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

    def _ensure_sparse_test_components(self, torch):
        from simpledet.extensions import ENCODERS, NECKS

        if "US025SparseBackbone" not in ENCODERS.names():
            class US025SparseBackbone(torch.nn.Module):
                def __init__(self, **_kwargs):
                    super().__init__()
                    self.feature_channels = (4,)
                    self.projection = torch.nn.Conv2d(3, 4, kernel_size=1)

                def forward(self, images):
                    feature = torch.nn.functional.adaptive_avg_pool2d(images, (4, 4))
                    return (self.projection(feature),)

            ENCODERS.register(
                "US025SparseBackbone",
                required_dependencies=(("torch", "cpu"),),
                tensor_contracts=("feature_sequence", "feature_channels"),
                validation_status="runtime_validated",
                family="test",
            )(US025SparseBackbone)

        if "US025SparseNeck" not in NECKS.names():
            class US025SparseNeck(torch.nn.Module):
                def __init__(self, *, in_channels, out_channels=4, num_outs=1, **_kwargs):
                    super().__init__()
                    self.in_channels = tuple(int(channel) for channel in in_channels)
                    self.out_channels = int(out_channels)
                    self.num_outs = int(num_outs)
                    self.projection = torch.nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1)

                def forward(self, features):
                    return (self.projection(features[0]),)

            NECKS.register(
                "US025SparseNeck",
                required_dependencies=(("torch", "cpu"),),
                tensor_contracts=("feature_sequence", "feature_pyramid"),
                validation_status="runtime_validated",
                family="test",
            )(US025SparseNeck)

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

    def test_first_ten_roi_detector_families_run_tensor_smoke_paths(self):
        torch = require_torch()
        from simpledet.native.modeling import build_detector
        from simpledet.native.roi import TwoStageDetector

        architectures = (
            "fast_rcnn",
            "mask_rcnn",
            "cascade_rcnn",
            "cascade_mask_rcnn",
            "grid_rcnn",
            "libra_rcnn",
            "double_head_rcnn",
            "dynamic_rcnn",
            "sparse_rcnn",
        )
        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(16, 16),
            boxes_per_image=1,
            num_classes=3,
        )
        for architecture in architectures:
            with self.subTest(architecture=architecture):
                components = self._components(torch, architecture=architecture)
                with patch.dict(sys.modules, self._fake_torchvision_modules(torch)), patch(
                    "simpledet.native.assemblers.build_native_components",
                    return_value=components,
                ):
                    model = build_detector(name=architecture, num_classes=3, pretrained=False)

                self.assertIsInstance(model, TwoStageDetector)
                self.assertEqual(model.roi_variant, architecture)
                losses = model(batch.images, batch.targets)
                self.assertIn("loss_roi_classifier", losses)
                self.assertIn("loss_roi_box_reg", losses)
                self.assertIn("loss_total", losses)
                if architecture == "fast_rcnn":
                    self.assertNotIn("loss_rpn_objectness", losses)
                    self.assertEqual(model.proposal_source, "external")
                elif architecture == "sparse_rcnn":
                    self.assertNotIn("loss_rpn_objectness", losses)
                    self.assertEqual(model.proposal_source, "sparse")
                else:
                    self.assertIn("loss_rpn_objectness", losses)
                if architecture in {"mask_rcnn", "cascade_mask_rcnn"}:
                    self.assertIn("loss_mask", losses)
                if architecture == "grid_rcnn":
                    self.assertIn("loss_grid", losses)
                for loss in losses.values():
                    self.assertEqual(tuple(loss.shape), ())
                    self.assertTrue(bool(torch.isfinite(loss)))

                model.eval()
                with torch.no_grad():
                    predictions = model(batch.images)
                self.assertEqual(len(predictions), 1)
                self.assertIn("boxes", predictions[0])
                self.assertIn("scores", predictions[0])
                self.assertIn("labels", predictions[0])

    def test_sparse_rcnn_builds_learned_proposal_detector_and_runs_cpu_paths(self):
        torch = require_torch()
        self._ensure_sparse_test_components(torch)

        from simpledet.native.modeling import build_detector
        from simpledet.native.roi import SparseRCNNDetector
        from simpledet.suite import (
            build_custom_encoder,
            build_custom_head,
            build_custom_neck,
            build_detector as build_detector_spec,
        )

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(16, 16),
            boxes_per_image=1,
            num_classes=3,
        )
        spec = build_detector_spec(
            "Sparse R-CNN",
            num_classes=3,
            encoder=build_custom_encoder(
                "US025SparseBackbone",
                imports=(),
                feature_channels=(4,),
                in_channels=3,
            ),
            neck=build_custom_neck(
                "US025SparseNeck",
                imports=(),
                out_channels=4,
                num_outs=1,
            ),
            head=build_custom_head(
                "SparseRoIHead",
                imports=(),
                num_classes=3,
                roi_feat_size=1,
                fc_out_channels=16,
                conv_out_channels=4,
                with_avg_pool=True,
            ),
            num_proposals=5,
            num_stages=2,
            pretrained=False,
        )

        with patch.dict(sys.modules, self._fake_torchvision_modules(torch)):
            model = build_detector(
                name="Sparse R-CNN",
                num_classes=3,
                detector_spec=spec,
                pretrained=False,
            )

        self.assertIsInstance(model, SparseRCNNDetector)
        self.assertEqual(model.roi_variant, "sparse_rcnn")
        self.assertEqual(model.proposal_source, "sparse")
        self.assertIsNone(model.rpn_head)
        self.assertEqual(tuple(model.sparse_proposal_boxes.shape), (5, 4))
        self.assertEqual(tuple(model.sparse_proposal_features.shape), (5, 4))

        losses = model(batch.images, batch.targets)
        self.assertIn("loss_roi_classifier", losses)
        self.assertIn("loss_roi_box_reg", losses)
        self.assertIn("loss_total", losses)
        self.assertNotIn("loss_rpn_objectness", losses)
        for loss in losses.values():
            self.assertEqual(tuple(loss.shape), ())
            self.assertTrue(bool(torch.isfinite(loss)))

        model.eval()
        with torch.no_grad():
            predictions = model(batch.images)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(set(predictions[0]), {"boxes", "scores", "labels"})
        self.assertEqual(predictions[0]["boxes"].shape[-1], 4)

    def test_rpn_detector_runs_tensor_smoke_paths(self):
        torch = require_torch()
        from simpledet.native.modeling import build_detector
        from simpledet.native.modeling import SingleStageDetector

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(16, 16),
            boxes_per_image=1,
            num_classes=3,
        )
        components = self._rpn_components(torch)

        with patch(
            "simpledet.native.assemblers.build_native_components",
            return_value=components,
        ):
            model = build_detector(name="rpn", num_classes=3, pretrained=False)

        self.assertIsInstance(model, SingleStageDetector)
        losses = model(batch.images, batch.targets)
        self.assertEqual(
            set(losses),
            {"loss_rpn_objectness", "loss_rpn_box_reg", "loss_total"},
        )
        for loss in losses.values():
            self.assertEqual(tuple(loss.shape), ())
            self.assertTrue(bool(torch.isfinite(loss)))

        model.eval()
        with torch.no_grad():
            predictions = model(batch.images)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(set(predictions[0]), {"boxes", "scores", "labels"})
        self.assertEqual(predictions[0]["boxes"].shape[-1], 4)

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
