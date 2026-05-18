import sys
import types
import unittest
from importlib.util import find_spec
from unittest.mock import patch

from simpledet.suite import (
    build_backbone,
    build_detector,
    build_encoder,
    compile_native_detector_plan,
    inspect_backbone,
    list_backbones,
)


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return list(self._channels)


class _FakeEncoder:
    def __init__(self, channels):
        self.feature_info = _FeatureInfo(channels)

    def __call__(self, x):
        return [("feature", len(self.feature_info.channels()))]


def _fake_torch_modules():
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return []

        def __call__(self, *args, **kwargs):
            forward = getattr(self, "forward", None)
            if callable(forward):
                return forward(*args, **kwargs)
            raise TypeError("forward not implemented")

    class ModuleList(list):
        pass

    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self.layers = list(layers)

        def forward(self, x):
            value = x
            for layer in self.layers:
                value = layer(value)
            return value

    class Conv2d(Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    class Linear(Module):
        def forward(self, x):
            return x

    class ReLU(Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    fake_nn.Module = Module
    fake_nn.ModuleList = ModuleList
    fake_nn.Sequential = Sequential
    fake_nn.Conv2d = Conv2d
    fake_nn.Linear = Linear
    fake_nn.ReLU = ReLU
    fake_torch.nn = fake_nn
    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_f,
    }


class NativeBackboneTests(unittest.TestCase):
    def test_backbone_alias_discovery_covers_required_families(self):
        self.assertIn("resnet50", list_backbones())

        expected_aliases = {
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
        for alias, (expected_name, expected_family) in expected_aliases.items():
            with self.subTest(alias=alias):
                metadata = inspect_backbone(alias)
                self.assertEqual(metadata["name"], expected_name)
                self.assertEqual(metadata["family"], expected_family)
                self.assertEqual(len(metadata["feature_channels"]), 4)

    def test_build_backbone_returns_expected_stage_metadata(self):
        backbone = build_backbone(
            name="resnet50",
            pretrained=False,
            in_channels=4,
            out_indices=(1, 2, 3, 4),
        )
        self.assertEqual(backbone.source, "native")
        self.assertEqual(backbone.name, "resnet50")
        self.assertEqual(backbone.feature_channels, (256, 512, 1024, 2048))
        self.assertEqual(backbone.backbone_cfg["type"], "resnet50")
        self.assertEqual(backbone.backbone_cfg["model_name"], "resnet50")
        self.assertEqual(backbone.backbone_cfg["pretrained"], False)
        self.assertEqual(backbone.backbone_cfg["in_channels"], 4)
        self.assertEqual(backbone.backbone_cfg["out_indices"], (1, 2, 3, 4))

        spec = build_detector("retinanet", num_classes=2, encoder=backbone)
        plan = compile_native_detector_plan(spec)

        self.assertEqual(plan.encoder.type, "resnet50")
        self.assertEqual(plan.encoder.source, "native")
        self.assertEqual(plan.encoder.params["model_name"], "resnet50")
        self.assertEqual(plan.encoder.params["feature_channels"], [256, 512, 1024, 2048])

    def test_build_backbone_unknown_name_is_actionable(self):
        with self.assertRaises(ValueError) as context:
            build_backbone(name="resnet999")

        message = str(context.exception)
        self.assertIn("Unknown backbone 'resnet999'", message)
        self.assertIn("Supported backbone aliases", message)
        self.assertIn("list_backbones()", message)

    def test_build_encoder_raw_timm_string_still_compiles_to_timm_plan(self):
        encoder = build_encoder("resnet18.a1_in1k", source="timm", out_indices=[0, 2])
        spec = build_detector("retinanet", num_classes=2, encoder=encoder)
        plan = compile_native_detector_plan(spec)

        self.assertEqual(plan.encoder.type, "timm")
        self.assertEqual(plan.encoder.source, "timm")
        self.assertEqual(plan.encoder.params["model_name"], "resnet18.a1_in1k")
        self.assertEqual(plan.encoder.params["out_indices"], [0, 2])

    def test_build_backbone_timm_prefix_compiles_to_timm_plan(self):
        encoder = build_backbone(
            name="timm:resnet18",
            pretrained=False,
            in_channels=4,
            out_indices=(1, 2, 3, 4),
            drop_rate=0.1,
        )
        spec = build_detector("retinanet", num_classes=2, encoder=encoder)
        plan = compile_native_detector_plan(spec)

        self.assertEqual(encoder.source, "timm")
        self.assertEqual(encoder.name, "resnet18")
        self.assertIsNone(encoder.feature_channels)
        self.assertEqual(plan.encoder.type, "timm")
        self.assertEqual(plan.encoder.source, "timm")
        self.assertEqual(plan.encoder.params["model_name"], "resnet18")
        self.assertEqual(plan.encoder.params["pretrained"], False)
        self.assertEqual(plan.encoder.params["in_channels"], 4)
        self.assertEqual(plan.encoder.params["out_indices"], (1, 2, 3, 4))
        self.assertEqual(plan.encoder.params["drop_rate"], 0.1)

    def test_build_backbone_timm_prefix_requires_explicit_out_indices(self):
        with self.assertRaises(ValueError) as context:
            build_backbone(name="timm:resnet18", pretrained=False)

        message = str(context.exception)
        self.assertIn("TIMM backbone names require explicit out_indices", message)
        self.assertIn("build_backbone('timm:resnet18'", message)

    def test_build_native_backbone_resolves_registered_alias(self):
        captured = {}
        fake_timm = types.ModuleType("timm")

        def create_model(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = dict(kwargs)
            return _FakeEncoder([256, 512, 1024, 2048])

        fake_timm.create_model = create_model
        fake_modules = _fake_torch_modules()
        fake_modules["timm"] = fake_timm

        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import build_native_backbone

            encoder = build_backbone(
                "ResNet",
                pretrained=False,
                in_channels=4,
                out_indices=(1, 2, 3, 4),
                drop_rate=0.1,
            )
            spec = build_detector("retinanet", num_classes=2, encoder=encoder)
            plan = compile_native_detector_plan(spec)
            backbone, metadata = build_native_backbone(plan.encoder)

        self.assertEqual(captured["model_name"], "resnet50")
        self.assertEqual(captured["kwargs"]["pretrained"], False)
        self.assertEqual(captured["kwargs"]["in_chans"], 4)
        self.assertEqual(captured["kwargs"]["out_indices"], (1, 2, 3, 4))
        self.assertEqual(captured["kwargs"]["drop_rate"], 0.1)
        self.assertEqual(backbone.feature_channels, (256, 512, 1024, 2048))
        self.assertEqual(metadata.name, "resnet50")
        self.assertEqual(metadata.source, "native")
        self.assertEqual(metadata.feature_channels, (256, 512, 1024, 2048))

    def test_build_native_backbone_preserves_custom_feature_channel_metadata(self):
        fake_modules = _fake_torch_modules()
        with patch.dict(sys.modules, fake_modules):
            from simpledet.extensions import ENCODERS
            from simpledet.native.backbones import build_native_backbone

            class UnitTestBackbone(fake_modules["torch.nn"].Module):
                def __init__(self, *, depth):
                    super().__init__()
                    self.depth = depth

            ENCODERS.register("UnitTestBackboneUS006")(UnitTestBackbone)
            plan = type(
                "Plan",
                (),
                {
                    "type": "UnitTestBackboneUS006",
                    "source": "native",
                    "params": {"depth": 12, "feature_channels": [7, 8, 9]},
                },
            )()
            backbone, metadata = build_native_backbone(plan)

        self.assertEqual(backbone.depth, 12)
        self.assertEqual(metadata.feature_channels, (7, 8, 9))

    def test_timm_backbone_forwards_extra_params_and_out_indices(self):
        captured = {}

        fake_timm = types.ModuleType("timm")

        def create_model(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = dict(kwargs)
            return _FakeEncoder([16, 32, 64])

        fake_timm.create_model = create_model

        fake_modules = _fake_torch_modules()
        fake_modules["timm"] = fake_timm
        with patch.dict(sys.modules, fake_modules):
            from simpledet.suite import build_encoder, build_detector, compile_native_detector_plan
            from simpledet.native.backbones import build_native_backbone

            spec = build_detector(
                "retinanet",
                num_classes=2,
                encoder=build_encoder(
                    "resnet18.a1_in1k",
                    source="timm",
                    pretrained=False,
                    in_channels=4,
                    out_indices=[0, 2],
                    drop_rate=0.2,
                ),
            )
            # keep consistent with expected plain spec path
            plan = compile_native_detector_plan(spec)
            build_native_backbone(plan.encoder)

        self.assertEqual(captured["model_name"], "resnet18.a1_in1k")
        self.assertTrue(captured["kwargs"]["features_only"])
        self.assertEqual(captured["kwargs"]["in_chans"], 4)
        self.assertEqual(captured["kwargs"]["pretrained"], False)
        self.assertEqual(captured["kwargs"]["out_indices"], (0, 2))
        self.assertEqual(captured["kwargs"]["drop_rate"], 0.2)

    def test_timm_backbone_forces_features_only_and_exposes_feature_info(self):
        captured = {}

        fake_timm = types.ModuleType("timm")

        def create_model(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = dict(kwargs)
            return _FakeEncoder([64, 128, 256, 512])

        fake_timm.create_model = create_model

        fake_modules = _fake_torch_modules()
        fake_modules["timm"] = fake_timm
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import build_native_backbone

            encoder = build_backbone(
                "timm:resnet18",
                pretrained=False,
                in_channels=4,
                out_indices=(1, 2, 3, 4),
                features_only=False,
                drop_rate=0.2,
            )
            spec = build_detector("retinanet", num_classes=2, encoder=encoder)
            plan = compile_native_detector_plan(spec)
            backbone, metadata = build_native_backbone(plan.encoder)

        self.assertEqual(captured["model_name"], "resnet18")
        self.assertTrue(captured["kwargs"]["features_only"])
        self.assertEqual(captured["kwargs"]["in_chans"], 4)
        self.assertEqual(captured["kwargs"]["pretrained"], False)
        self.assertEqual(captured["kwargs"]["out_indices"], (1, 2, 3, 4))
        self.assertEqual(captured["kwargs"]["drop_rate"], 0.2)
        self.assertIs(backbone.feature_info, backbone.encoder.feature_info)
        self.assertEqual(metadata.feature_channels, (64, 128, 256, 512))

    def test_timm_backbone_without_out_indices_preserves_timm_default(self):
        captured = {}

        fake_timm = types.ModuleType("timm")

        def create_model(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = dict(kwargs)
            return _FakeEncoder([16, 32, 64])

        fake_timm.create_model = create_model

        fake_modules = _fake_torch_modules()
        fake_modules["timm"] = fake_timm
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import TimmFeatureBackbone

            TimmFeatureBackbone(
                model_name="resnet18.a1_in1k",
                pretrained=False,
                timm_kwargs={"features_only": False},
            )

        self.assertEqual(captured["model_name"], "resnet18.a1_in1k")
        self.assertTrue(captured["kwargs"]["features_only"])
        self.assertNotIn("out_indices", captured["kwargs"])

    def test_timm_backbone_missing_extra_has_install_hint(self):
        fake_modules = _fake_torch_modules()
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import TimmFeatureBackbone

            with patch(
                "simpledet.detectors._deps.import_module",
                side_effect=ModuleNotFoundError("missing", name="timm"),
            ):
                with self.assertRaises(ImportError) as context:
                    TimmFeatureBackbone(
                        model_name="resnet18",
                        pretrained=False,
                        out_indices=(1, 2, 3, 4),
                    )

        self.assertIn("python -m pip install 'simpledet[timm]'", str(context.exception))

    def test_build_native_model_accepts_non_3_channel_input(self):
        captured = {}

        fake_timm = types.ModuleType("timm")

        def create_model(model_name, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return _FakeEncoder([16, 32, 64])

        fake_timm.create_model = create_model

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {str(index): feature for index, feature in enumerate(features)}

        class _FakeRetinaHead:
            def __init__(self, in_channels, num_anchors, num_classes):
                self.in_channels = in_channels
                self.num_anchors = num_anchors
                self.num_classes = num_classes

            def __call__(self, features):
                return {"cls_logits": ["logits"], "bbox_regression": ["bbox"]}

        fake_modules = _fake_torch_modules()
        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_retinanet = types.ModuleType("torchvision.models.detection.retinanet")
        fake_retinanet.RetinaNetHead = _FakeRetinaHead
        fake_detect = types.ModuleType("torchvision.models.detection")
        fake_models = types.ModuleType("torchvision.models")
        fake_torchvision = types.ModuleType("torchvision")
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": fake_models,
                "torchvision.models.detection": fake_detect,
                "torchvision.models.detection.retinanet": fake_retinanet,
            }
        )

        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model
            from simpledet.suite import build_detector

            spec = build_detector("retinanet", num_classes=2, encoder="resnet18.a1_in1k", in_channels=4)
            model = build_native_model("retinanet", num_classes=2, detector_spec=spec, in_channels=4)

        self.assertEqual(model.backbone.in_channels, 4)
        self.assertEqual(captured["kwargs"]["in_chans"], 4)

    def test_build_native_backbone_supports_timm(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([32, 64, 128, 256])

        fake_modules = _fake_torch_modules()
        fake_modules["timm"] = fake_timm
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import BackboneSpec, TimmFeatureBackbone, build_native_backbone

            spec = build_detector(
                "retinanet",
                num_classes=2,
                encoder="resnet18.a1_in1k",
                in_channels=3,
            )
            plan = compile_native_detector_plan(spec)
            backbone, metadata = build_native_backbone(plan.encoder)

        self.assertIsInstance(backbone, TimmFeatureBackbone)
        self.assertIsInstance(metadata, BackboneSpec)
        self.assertEqual(metadata.source, "timm")
        self.assertEqual(metadata.name, "resnet18.a1_in1k")
        self.assertEqual(metadata.feature_channels, (32, 64, 128, 256))

    def test_timm_backbone_returns_feature_tuple(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([16, 32, 64])

        fake_modules = _fake_torch_modules()
        fake_modules["timm"] = fake_timm
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import TimmFeatureBackbone

            backbone = TimmFeatureBackbone(model_name="tiny", in_channels=3, pretrained=False)
            outputs = backbone("tensor")

        self.assertEqual(outputs, (("feature", 3),))
        self.assertEqual(backbone.feature_channels, (16, 32, 64))

    def test_real_timm_prefixed_backbone_runs_cpu_when_installed(self):
        if find_spec("torch") is None:
            raise unittest.SkipTest(
                "PyTorch CPU runtime is not installed; install with python -m pip install 'simpledet[cpu]'."
            )
        if find_spec("timm") is None:
            raise unittest.SkipTest(
                "TIMM optional extra is not installed; install with python -m pip install 'simpledet[timm]'."
            )

        import torch

        from simpledet.native.backbones import build_native_backbone

        encoder = build_backbone(
            name="timm:resnet18",
            pretrained=False,
            out_indices=(1, 2, 3, 4),
        )
        spec = build_detector("retinanet", num_classes=2, encoder=encoder)
        plan = compile_native_detector_plan(spec)
        backbone, metadata = build_native_backbone(plan.encoder)

        with torch.no_grad():
            features = backbone(torch.zeros(1, 3, 64, 64))

        self.assertTrue(hasattr(backbone, "feature_info"))
        self.assertEqual(len(features), 4)
        self.assertEqual(
            metadata.feature_channels,
            tuple(int(feature.shape[1]) for feature in features),
        )
