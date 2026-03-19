import sys
import types
import unittest
from unittest.mock import patch

from simpledet.suite import build_detector, compile_native_detector_plan


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
