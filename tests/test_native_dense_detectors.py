from types import SimpleNamespace
import unittest
from unittest.mock import patch

from native_tensor_contracts import require_torch


_DENSE_DETECTOR_ALIASES = (
    ("retinanet", "retinanet", "RetinaHead"),
    ("RetinaNet", "retinanet", "RetinaHead"),
    ("fcos", "fcos", "FCOSHead"),
    ("FCOS", "fcos", "FCOSHead"),
    ("atss", "atss", "ATSSHead"),
    ("ATSS", "atss", "ATSSHead"),
    ("fsaf", "fsaf", "FSAFHead"),
    ("FSAF", "fsaf", "FSAFHead"),
    ("fovea", "fovea", "FoveaHead"),
    ("FOVEA", "fovea", "FoveaHead"),
    ("FoveaBox", "fovea", "FoveaHead"),
    ("free_anchor", "free_anchor", "FreeAnchorRetinaHead"),
    ("FreeAnchor", "free_anchor", "FreeAnchorRetinaHead"),
    ("gfl", "gfl", "GFLHead"),
    ("GFL", "gfl", "GFLHead"),
    ("gfocalv2", "gfocalv2", "GFLV2Head"),
    ("GFocalV2", "gfocalv2", "GFLV2Head"),
    ("vfnet", "vfnet", "VFNetHead"),
    ("VFNet", "vfnet", "VFNetHead"),
    ("paa", "paa", "PAAHead"),
    ("PAA", "paa", "PAAHead"),
    ("reppoints", "reppoints", "RepPointsHead"),
    ("RepPoints", "reppoints", "RepPointsHead"),
    ("yolof", "yolof", "YOLOFHead"),
    ("YOLOF", "yolof", "YOLOFHead"),
    ("tood", "tood", "TOODHead"),
    ("TOOD", "tood", "TOODHead"),
    ("ddod", "ddod", "DDODHead"),
    ("DDOD", "ddod", "DDODHead"),
    ("auto_assign", "auto_assign", "AutoAssignHead"),
    ("AutoAssign", "auto_assign", "AutoAssignHead"),
    ("nas_fcos", "nas_fcos", "NASFCOSHead"),
    ("NAS-FCOS", "nas_fcos", "NASFCOSHead"),
)

_FORWARD_SMOKE_ALIASES = (
    ("VFNet", "VFNetHead"),
    ("FoveaBox", "FoveaHead"),
    ("FreeAnchor", "FreeAnchorRetinaHead"),
    ("GFocalV2", "GFLV2Head"),
    ("RepPoints", "RepPointsHead"),
    ("NAS-FCOS", "NASFCOSHead"),
)

_TEST_BACKBONE = "US023TinyBackbone"
_TEST_NECK = "US023IdentityNeck"


class NativeDenseDetectorFamilyTests(unittest.TestCase):
    def test_suite_construction_for_every_dense_detector_alias(self):
        from simpledet.suite import compile_native_detector_plan

        for alias, expected_architecture, expected_head in _DENSE_DETECTOR_ALIASES:
            with self.subTest(alias=alias):
                spec = self._detector_spec(alias)
                plan = compile_native_detector_plan(spec)

                self.assertEqual(spec.family, "dense")
                self.assertEqual(spec.architecture, expected_architecture)
                self.assertEqual(spec.head.name, expected_head)
                self.assertEqual(plan.head.type, expected_head)

    def test_native_construction_for_every_dense_detector_alias(self):
        require_torch()
        self._ensure_test_components()

        from simpledet.native.modeling import SingleStageDetector, build_detector as build_native_detector

        for alias, _expected_architecture, expected_head in _DENSE_DETECTOR_ALIASES:
            with self.subTest(alias=alias):
                spec = self._detector_spec(alias)
                try:
                    model = build_native_detector(
                        name=alias,
                        num_classes=4,
                        detector_spec=spec,
                        pretrained=False,
                    )
                except ImportError as exc:
                    self.skipTest(str(exc))

                self.assertIsInstance(model, SingleStageDetector)
                self.assertEqual(model.head_spec.name, expected_head)
                self.assertEqual(model.head_spec.num_classes, 4)

    def test_native_build_detector_vfnet_example_uses_vfnet_head(self):
        torch = require_torch()

        from simpledet.native.assemblers import NativeModelComponents
        from simpledet.native.modeling import SingleStageDetector, build_detector

        components = NativeModelComponents(
            plan=SimpleNamespace(
                architecture="vfnet",
                family="dense",
                head=SimpleNamespace(type="VFNetHead"),
            ),
            backbone=torch.nn.Identity(),
            backbone_spec=SimpleNamespace(feature_channels=(8,)),
            neck=torch.nn.Identity(),
            neck_spec=SimpleNamespace(out_channels=8),
            head=torch.nn.Identity(),
            head_spec=SimpleNamespace(name="VFNetHead", num_classes=4),
        )

        with patch("simpledet.native.assemblers.build_native_components", return_value=components):
            model = build_detector(name="vfnet", num_classes=4, pretrained=False)

        self.assertIsInstance(model, SingleStageDetector)
        self.assertEqual(model.head_spec.name, "VFNetHead")
        self.assertEqual(model.head_spec.num_classes, 4)

    def test_representative_dense_detector_forward_smokes(self):
        torch = require_torch()
        self._ensure_test_components()

        from simpledet.native.modeling import build_detector as build_native_detector

        image = torch.rand((3, 32, 32), dtype=torch.float32)
        for alias, expected_head in _FORWARD_SMOKE_ALIASES:
            with self.subTest(alias=alias):
                spec = self._detector_spec(alias)
                try:
                    model = build_native_detector(
                        name=alias,
                        num_classes=4,
                        detector_spec=spec,
                        pretrained=False,
                    )
                except ImportError as exc:
                    self.skipTest(str(exc))
                model.eval()

                with torch.no_grad():
                    predictions = model.predict([image])

                self.assertEqual(model.head_spec.name, expected_head)
                self.assertEqual(len(predictions), 1)
                self.assertEqual(set(predictions[0]), {"boxes", "scores", "labels"})
                self.assertEqual(predictions[0]["boxes"].shape[-1], 4)
                self.assertEqual(predictions[0]["scores"].dim(), 1)
                self.assertEqual(predictions[0]["labels"].dim(), 1)

    def _detector_spec(self, alias):
        from simpledet.suite import (
            build_custom_encoder,
            build_custom_neck,
            build_detector,
        )

        return build_detector(
            alias,
            num_classes=4,
            encoder=build_custom_encoder(
                _TEST_BACKBONE,
                imports=(),
                feature_channels=(8, 8, 8, 8),
                in_channels=3,
            ),
            neck=build_custom_neck(
                _TEST_NECK,
                imports=(),
                out_channels=8,
                num_outs=4,
            ),
            pretrained=False,
        )

    def _ensure_test_components(self):
        torch = require_torch()
        from simpledet.extensions import ENCODERS, NECKS

        if _TEST_BACKBONE not in ENCODERS.names():
            class US023TinyBackbone(torch.nn.Module):
                def __init__(self, **_kwargs):
                    super().__init__()
                    self.feature_channels = (8, 8, 8, 8)
                    self.projections = torch.nn.ModuleList(
                        [torch.nn.Conv2d(3, 8, kernel_size=1) for _ in self.feature_channels]
                    )

                def forward(self, images):
                    features = []
                    current = images
                    for index, projection in enumerate(self.projections):
                        if index > 0:
                            current = torch.nn.functional.avg_pool2d(current, kernel_size=2, stride=2)
                        features.append(projection(current))
                    return tuple(features)

            ENCODERS.register(
                _TEST_BACKBONE,
                required_dependencies=(("torch", "cpu"),),
                tensor_contracts=("feature_sequence", "feature_channels"),
                validation_status="runtime_validated",
                family="test",
            )(US023TinyBackbone)

        if _TEST_NECK not in NECKS.names():
            class US023IdentityNeck(torch.nn.Module):
                def __init__(self, *, in_channels, out_channels=8, num_outs=4, **_kwargs):
                    super().__init__()
                    self.in_channels = tuple(int(channel) for channel in in_channels)
                    self.out_channels = int(out_channels)
                    self.num_outs = int(num_outs)
                    self.projections = torch.nn.ModuleList(
                        [
                            torch.nn.Conv2d(int(channel), self.out_channels, kernel_size=1)
                            for channel in self.in_channels[: self.num_outs]
                        ]
                    )

                def forward(self, features):
                    return tuple(
                        projection(feature)
                        for projection, feature in zip(self.projections, features)
                    )

            NECKS.register(
                _TEST_NECK,
                required_dependencies=(("torch", "cpu"),),
                tensor_contracts=("feature_sequence", "feature_pyramid"),
                validation_status="runtime_validated",
                family="test",
            )(US023IdentityNeck)


if __name__ == "__main__":
    unittest.main()
