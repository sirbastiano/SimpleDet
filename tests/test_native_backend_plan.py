import unittest
import sys
import types
from unittest.mock import patch

from simpledet.extensions import DECODERS, DETECTORS, ENCODERS, HEADS, NECKS, ExtensionRegistry
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


def _fake_torch_modules():
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    class ModuleList(list):
        pass

    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self.layers = list(layers)

    class Conv2d(Module):
        pass

    class Identity(Module):
        pass

    class Linear(Module):
        pass

    class Embedding(Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = object()

    class TransformerEncoderLayer(Module):
        pass

    class TransformerDecoderLayer(Module):
        pass

    class TransformerEncoder(Module):
        pass

    class TransformerDecoder(Module):
        pass

    class ReLU(Module):
        pass

    fake_nn.Module = Module
    fake_nn.ModuleList = ModuleList
    fake_nn.Sequential = Sequential
    fake_nn.Conv2d = Conv2d
    fake_nn.ConvTranspose2d = Conv2d
    fake_nn.Identity = Identity
    fake_nn.Linear = Linear
    fake_nn.Embedding = Embedding
    fake_nn.TransformerEncoderLayer = TransformerEncoderLayer
    fake_nn.TransformerDecoderLayer = TransformerDecoderLayer
    fake_nn.TransformerEncoder = TransformerEncoder
    fake_nn.TransformerDecoder = TransformerDecoder
    fake_nn.ReLU = ReLU
    fake_torch.nn = fake_nn
    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_f,
    }


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

    def test_compile_native_detector_plan_for_two_stage_components(self):
        faster = compile_native_detector_plan(
            build_detector("faster_rcnn", num_classes=3, encoder="resnet18.a1_in1k")
        )
        self.assertEqual(faster.family, "roi")
        self.assertEqual(faster.rpn_head.type, "RPNHead")
        self.assertEqual(faster.rpn_head.params["num_anchors"], 1)
        self.assertEqual(faster.bbox_head.type, "Shared2FCBBoxHead")
        self.assertIs(faster.head, faster.bbox_head)
        self.assertIsNone(faster.mask_head)
        self.assertIsNone(faster.grid_head)

        mask = compile_native_detector_plan(
            build_detector("mask_rcnn", num_classes=3, encoder="resnet18.a1_in1k")
        )
        self.assertEqual(mask.bbox_head.type, "Shared2FCBBoxHead")
        self.assertTrue(mask.bbox_head.params["with_mask"])
        self.assertEqual(mask.mask_head.type, "FCNMaskHead")

        grid = compile_native_detector_plan(
            build_detector("grid_rcnn", num_classes=3, encoder="resnet18.a1_in1k")
        )
        self.assertEqual(grid.grid_head.type, "GridHead")
        self.assertEqual(grid.grid_head.params["grid_size"], 7)

        cascade = compile_native_detector_plan(
            build_detector("cascade_rcnn", num_classes=3, encoder="resnet18.a1_in1k")
        )
        self.assertEqual(cascade.bbox_head.type, "CascadeBBoxHead")

    def test_compile_native_detector_plan_for_query_components(self):
        expected_heads = {
            "detr": "DETRHead",
            "conditional_detr": "ConditionalDETRHead",
            "dab_detr": "DABDETRHead",
            "deformable_detr": "DeformableDETRHead",
            "dino": "DINOHead",
        }
        for architecture, expected_head in expected_heads.items():
            with self.subTest(architecture=architecture):
                plan = compile_native_detector_plan(
                    build_detector(
                        architecture,
                        num_classes=3,
                        encoder="resnet18.a1_in1k",
                        num_queries=20,
                        hidden_dim=16,
                        num_heads=4,
                    )
                )

                self.assertEqual(plan.family, "transformer")
                self.assertEqual(plan.head.type, expected_head)
                self.assertEqual(plan.head.params["num_classes"], 3)
                self.assertEqual(plan.head.params["num_queries"], 20)
                self.assertEqual(plan.head.params["hidden_dim"], 16)


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

    def test_registry_lookup_normalizes_alias_case_and_separators(self):
        registry = ExtensionRegistry("detector")

        def factory():
            return "vfnet"

        registry.register(
            "vfnet",
            aliases=("VFNet",),
            required_dependencies=(("torch", "cpu"),),
            tensor_contracts=("feature_pyramid", "dense_predictions"),
            validation_status="runtime_validated",
            family="dense",
            summary="VFNet detector family.",
        )(factory)

        lower = registry.lookup("vfnet")
        mixed = registry.lookup("VFNet")
        spaced = registry.lookup("vf net")
        self.assertIs(lower, mixed)
        self.assertIs(lower, spaced)
        self.assertEqual(lower.name, "vfnet")
        self.assertEqual(lower.kind, "detector")
        self.assertIs(lower.factory, factory)
        self.assertEqual(lower.required_dependencies[0].module, "torch")
        self.assertEqual(lower.required_dependencies[0].extra, "cpu")
        self.assertEqual(lower.tensor_contracts, ("feature_pyramid", "dense_predictions"))
        self.assertEqual(lower.validation_status, "runtime_validated")
        self.assertEqual(lower.family, "dense")
        self.assertIn("VFNet", lower.aliases)

    def test_registry_rejects_duplicate_aliases(self):
        registry = ExtensionRegistry("head")

        class FirstHead:
            pass

        class SecondHead:
            pass

        registry.register("FirstHead", aliases=("dense_alias",))(FirstHead)
        with self.assertRaises(ValueError) as context:
            registry.register("SecondHead", aliases=("Dense-Alias",))(SecondHead)
        self.assertIn("head component 'SecondHead' conflicts", str(context.exception))
        self.assertIn("FirstHead", str(context.exception))

    def test_registry_rejects_duplicate_aliases_on_same_registration(self):
        registry = ExtensionRegistry("detector")

        def factory():
            return "vfnet"

        with self.assertRaises(ValueError) as context:
            registry.register("vfnet", aliases=("VFNet", "VF Net"))(factory)
        self.assertIn("detector component 'vfnet' repeats alias 'VF Net'", str(context.exception))

    def test_registry_rejects_normalized_name_collisions(self):
        registry = ExtensionRegistry("detector")

        def first():
            return "first"

        def second():
            return "second"

        registry.register("vfnet")(first)
        with self.assertRaises(ValueError) as context:
            registry.register("VFNet")(second)
        self.assertIn("detector component 'VFNet' conflicts", str(context.exception))

    def test_registry_unknown_lookup_mentions_kind_names_and_aliases(self):
        registry = ExtensionRegistry("detector")

        def factory():
            return "vfnet"

        registry.register("vfnet", aliases=("VFNet",))(factory)

        with self.assertRaises(KeyError) as context:
            registry.lookup("unknown_detector")
        message = str(context.exception)
        self.assertIn("Unknown detector component 'unknown_detector'", message)
        self.assertIn("Registered detector names: vfnet", message)
        self.assertIn("Aliases: VFNet -> vfnet", message)

    def test_registry_missing_dependency_message_includes_optional_extra(self):
        registry = ExtensionRegistry("backbone")

        class TimmBackbone:
            pass

        registry.register(
            "timm",
            aliases=("TimmEncoder",),
            required_dependencies=(("missing_timm", "timm", "timm"),),
        )(TimmBackbone)

        with patch(
            "simpledet.extensions.registry.import_module",
            side_effect=ModuleNotFoundError("missing", name="missing_timm"),
        ):
            with self.assertRaises(ImportError) as context:
                registry.require_dependencies("TimmEncoder")
        message = str(context.exception)
        self.assertIn("backbone component 'timm' requires optional dependencies", message)
        self.assertIn("'missing_timm'", message)
        self.assertIn("simpledet[timm]", message)

    def test_native_detector_registry_exposes_major_alias_metadata(self):
        with patch.dict(sys.modules, _fake_torch_modules()):
            import simpledet.native.assemblers  # noqa: F401

        metadata = DETECTORS.lookup("VFNet")
        self.assertIs(metadata, DETECTORS.lookup("vfnet"))
        self.assertEqual(metadata.name, "vfnet")
        self.assertEqual(metadata.kind, "detector")
        self.assertEqual(metadata.family, "dense")
        self.assertIn("VFNet", metadata.aliases)
        self.assertIn("dense_predictions", metadata.tensor_contracts)
        self.assertEqual(metadata.required_dependencies[0].module, "torch")

        aliases = {
            "FOVEA": ("fovea", "dense"),
            "FoveaBox": ("foveabox", "dense"),
            "RepPoints": ("reppoints", "dense"),
            "YOLOF": ("yolof", "dense"),
            "CenterNet": ("centernet", "dense"),
            "Faster R-CNN": ("faster_rcnn", "roi"),
            "Mask R-CNN": ("mask_rcnn", "roi"),
            "Grid R-CNN": ("grid_rcnn", "roi"),
            "Cascade R-CNN": ("cascade_rcnn", "roi"),
            "DAB-DETR": ("dab_detr", "transformer"),
        }
        for alias, (expected_name, expected_family) in aliases.items():
            with self.subTest(alias=alias):
                alias_metadata = DETECTORS.lookup(alias)
                self.assertEqual(alias_metadata.name, expected_name)
                self.assertEqual(alias_metadata.family, expected_family)
                self.assertTrue(alias_metadata.required_dependencies)
                self.assertTrue(alias_metadata.tensor_contracts)
                self.assertNotEqual(alias_metadata.validation_status, "unvalidated")

    def test_native_head_registry_exposes_core_dense_aliases(self):
        with patch.dict(sys.modules, _fake_torch_modules()):
            import simpledet.native.heads  # noqa: F401
            from simpledet.suite import list_heads

            dense_heads = set(list_heads(kind="dense"))

        aliases = {
            "retina_head": "RetinaHead",
            "fcos_head": "FCOSHead",
            "atss_head": "ATSSHead",
            "fsaf_head": "FSAFHead",
            "fovea_head": "FoveaHead",
            "free_anchor_head": "FreeAnchorRetinaHead",
            "rpn_head": "RPNHead",
            "yolox_head": "YOLOXHead",
            "rtmdet_head": "RTMDetHead",
            "ssd_head": "SSDHead",
            "efficientdet_head": "EfficientDetHead",
        }
        for alias, expected_name in aliases.items():
            with self.subTest(alias=alias):
                self.assertIn(alias, dense_heads)
                metadata = HEADS.lookup(alias)
                self.assertEqual(metadata.name, expected_name)
                self.assertEqual(metadata.family, "dense")
                self.assertTrue(metadata.required_dependencies)
                self.assertTrue(metadata.tensor_contracts)
                self.assertNotEqual(metadata.validation_status, "unvalidated")

    def test_native_backbone_registry_exposes_major_alias_metadata(self):
        with patch.dict(sys.modules, _fake_torch_modules()):
            import simpledet.native.backbones  # noqa: F401

        aliases = {
            "ResNet": ("resnet50", "ResNet"),
            "ResNeXt": ("resnext50_32x4d", "ResNeXt"),
            "Res2Net": ("res2net50_26w_4s", "Res2Net"),
            "HRNet": ("hrnet_w18", "HRNet"),
            "CSPDarkNet": ("cspdarknet53", "CSPDarkNet"),
            "CSPNeXt": ("cspnext_tiny", "CSPNeXt"),
            "MobileNetV2": ("mobilenetv2_100", "MobileNetV2"),
            "MobileNetV3": ("mobilenetv3_large_100", "MobileNetV3"),
            "EfficientNet": ("efficientnet_b0", "EfficientNet"),
            "ConvNeXt": ("convnext_tiny", "ConvNeXt"),
            "Swin Transformer": ("swin_tiny_patch4_window7_224", "Swin Transformer"),
            "Vision Transformer": ("vit_base_patch16_224", "Vision Transformer"),
        }
        for alias, (expected_name, expected_family) in aliases.items():
            with self.subTest(alias=alias):
                metadata = ENCODERS.lookup(alias)
                self.assertEqual(metadata.name, expected_name)
                self.assertEqual(metadata.family, expected_family)
                self.assertTrue(metadata.required_dependencies)
                self.assertIn("feature_channels", metadata.tensor_contracts)
                self.assertNotEqual(metadata.validation_status, "unvalidated")

    def test_detector_alias_names_inherit_component_contract_metadata(self):
        with patch.dict(sys.modules, _fake_torch_modules()):
            import simpledet.native.assemblers  # noqa: F401

        aliases = {
            "retina": "dense",
            "deformable_detr": "transformer",
            "dab_detr": "transformer",
            "foveabox": "dense",
            "faster-rcnn": "roi",
            "mask-rcnn": "roi",
            "gridrcnn": "roi",
            "cascadercnn": "roi",
        }
        for alias, expected_family in aliases.items():
            with self.subTest(alias=alias):
                metadata = DETECTORS.lookup(alias)
                self.assertEqual(metadata.name, alias)
                self.assertEqual(metadata.family, expected_family)
                self.assertTrue(metadata.required_dependencies)
                self.assertTrue(metadata.tensor_contracts)
                self.assertNotEqual(metadata.validation_status, "unvalidated")
