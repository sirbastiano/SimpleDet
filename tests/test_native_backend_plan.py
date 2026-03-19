import unittest

from simpledet.extensions import DECODERS, ENCODERS, HEADS, NECKS
from simpledet.suite import (
    ComponentPlan,
    DetectorBuildPlan,
    build_custom_detector,
    build_custom_encoder,
    build_custom_head,
    build_custom_neck,
    build_detector,
    compile_native_detector_plan,
)


class NativeBuildPlanTests(unittest.TestCase):
    def test_compile_native_detector_plan_from_custom_components(self):
        spec = build_custom_detector(
            "my_dense_detector",
            family="dense",
            num_classes=3,
            imports=("pkg.bootstrap",),
            encoder=build_custom_encoder(
                "MyBackbone",
                imports=("pkg.backbones",),
                feature_channels=[64, 128, 256],
                depth=12,
            ),
            neck=build_custom_neck(
                "MyNeck",
                imports=("pkg.necks",),
                out_channels=192,
            ),
            head=build_custom_head(
                "MyHead",
                imports=("pkg.heads",),
                dropout=0.1,
            ),
        )

        plan = compile_native_detector_plan(spec)

        self.assertIsInstance(plan, DetectorBuildPlan)
        self.assertEqual(plan.architecture, "my_dense_detector")
        self.assertEqual(plan.family, "dense")
        self.assertEqual(plan.imports, ("pkg.bootstrap", "pkg.backbones", "pkg.necks", "pkg.heads"))
        self.assertIsInstance(plan.encoder, ComponentPlan)
        self.assertEqual(plan.encoder.type, "MyBackbone")
        self.assertEqual(plan.encoder.params["feature_channels"], [64, 128, 256])
        self.assertEqual(plan.neck.type, "MyNeck")
        self.assertEqual(plan.neck.params["out_channels"], 192)
        self.assertEqual(plan.head.type, "MyHead")
        self.assertEqual(plan.head.params["dropout"], 0.1)
        self.assertIsNone(plan.decoder)

    def test_compile_native_detector_plan_for_timm_encoder(self):
        spec = build_detector(
            "retinanet",
            num_classes=2,
            encoder="resnet18.a1_in1k",
            in_channels=3,
        )

        plan = compile_native_detector_plan(spec)

        self.assertEqual(plan.encoder.type, "timm")
        self.assertEqual(plan.encoder.source, "timm")
        self.assertEqual(plan.encoder.params["model_name"], "resnet18.a1_in1k")
        self.assertEqual(plan.encoder.params["in_channels"], 3)
        self.assertEqual(plan.head.params["num_classes"], 2)


class ExtensionRegistryTests(unittest.TestCase):
    def test_registry_registers_and_resolves_components(self):
        @ENCODERS.register("UnitTestEncoder")
        class UnitTestEncoder:
            pass

        @NECKS.register("UnitTestNeck")
        class UnitTestNeck:
            pass

        @HEADS.register("UnitTestHead")
        class UnitTestHead:
            pass

        @DECODERS.register("UnitTestDecoder")
        class UnitTestDecoder:
            pass

        self.assertIs(ENCODERS.get("UnitTestEncoder"), UnitTestEncoder)
        self.assertIs(NECKS.get("UnitTestNeck"), UnitTestNeck)
        self.assertIs(HEADS.get("UnitTestHead"), UnitTestHead)
        self.assertIs(DECODERS.get("UnitTestDecoder"), UnitTestDecoder)
