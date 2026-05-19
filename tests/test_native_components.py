import math
import itertools
import sys
import types
import unittest
from unittest.mock import patch

from simpledet.extensions import DETECTORS, ENCODERS, HEADS, LOSSES, NECKS
from simpledet.suite import (
    build_custom_detector,
    build_custom_encoder,
    build_detector,
    build_neck,
    compile_native_detector_plan,
)
from native_tensor_contracts import require_torch

_COUNTER = itertools.count()
_NATIVE_REGISTRIES = (DETECTORS, ENCODERS, HEADS, LOSSES, NECKS)


def _clear_native_modules():
    for module_name in list(sys.modules):
        if module_name == "simpledet.native" or module_name.startswith("simpledet.native."):
            sys.modules.pop(module_name, None)


def _snapshot_native_registries():
    return [(registry, dict(registry._items), dict(registry._metadata)) for registry in _NATIVE_REGISTRIES]


def _clear_native_registries():
    for registry in _NATIVE_REGISTRIES:
        registry._items = {}
        registry._metadata = {}


def _restore_native_registries(snapshots):
    for registry, items, metadata in snapshots:
        registry._items = items
        registry._metadata = metadata


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return list(self._channels)


class _FakeEncoder:
    def __init__(self, channels):
        self.feature_info = _FeatureInfo(channels)

    def __call__(self, x):
        return tuple(f"feat-{index}" for index, _ in enumerate(self.feature_info.channels()))


class _FakeTensorImage:
    shape = (3, 32, 32)

    def unsqueeze(self, dim):
        return self


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
        def forward(self, x):
            return x

    class Linear(Module):
        def forward(self, x):
            return x

    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.weight = f"embedding-{num_embeddings}-{embedding_dim}"

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
    fake_nn.Embedding = Embedding
    fake_nn.ReLU = ReLU
    fake_torch.nn = fake_nn
    fake_torch.tensor = lambda value, **kwargs: value
    fake_torch.float32 = "float32"
    fake_torch.long = "long"
    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_f,
    }


def _require_torchvision():
    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("Torchvision CPU runtime is not installed.") from exc


def _make_tensor_features(channels, spatial_shapes, *, batch_size=2):
    torch = require_torch()
    features = []
    for level, (channel, (height, width)) in enumerate(zip(channels, spatial_shapes)):
        features.append(
            torch.full(
                (batch_size, int(channel), int(height), int(width)),
                fill_value=float(level + 1) / 10.0,
                dtype=torch.float32,
            )
        )
    return tuple(features)


class NativeComponentTests(unittest.TestCase):
    def test_build_native_neck_aliases_resolved(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeNeckLayer:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {str(index): feature for index, feature in enumerate(features)}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeNeckLayer
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.necks import build_native_neck

            neck_cases = {
                "FPN": ((64, 128, 256, 512), {}),
                "FPNLite": ((64, 128, 256, 512), {}),
                "FPNLiteNeck": ((64, 128, 256, 512), {}),
                "PAN": ((64, 128, 256, 512), {}),
                "PANET": ((64, 128, 256, 512), {}),
                "BiFPN": ((64, 128, 256, 512), {}),
                "BiFPNV2": ((64, 128, 256, 512), {}),
                "PAFPN": ((64, 128, 256, 512), {}),
                "NASFPN": ((64, 128, 256, 512), {}),
                "DilatedEncoder": ((256,), {"in_channels": [256], "num_outs": 1}),
                "HRFPN": ((32, 64, 128, 256), {}),
                "SSDNeck": ((64, 128, 256), {"num_outs": 4}),
                "YOLOXPAFPN": ((64, 128, 256), {}),
            }
            for neck_name, (feature_channels, neck_kwargs) in neck_cases.items():
                detector_spec = build_detector(
                    "retinanet",
                    num_classes=2,
                    encoder="resnet18.a1_in1k",
                    neck=build_neck(neck_name, out_channels=256, **neck_kwargs),
                )
                plan = compile_native_detector_plan(detector_spec)
                _, neck_spec = build_native_neck(plan.neck, feature_channels=feature_channels)
                self.assertEqual(neck_spec.out_channels, 256)
                self.assertEqual(neck_spec.num_outs, int(neck_kwargs.get("num_outs", len(feature_channels))))

    def test_build_native_neck_aliases_normalized_names(self):
        fake_ops = types.ModuleType("torchvision.ops")

        class _FakeFPN:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, features):
                return features

        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.necks import build_native_neck

            aliases = {
                "fpn-lite": ("FPNLite", (64, 128, 256, 512)),
                "PANET": ("PANET", (64, 128, 256, 512)),
                "PAN": ("PAN", (64, 128, 256, 512)),
                "pa-fpn": ("PAFPN", (64, 128, 256, 512)),
                "nas-fpn": ("NASFPN", (64, 128, 256, 512)),
                "bi-fpn": ("BiFPN", (64, 128, 256, 512)),
                "dilated-encoder": ("DilatedEncoder", (256,)),
                "hr-fpn": ("HRFPN", (32, 64, 128, 256)),
                "ssd-neck": ("SSDNeck", (64, 128, 256)),
                "yolox-pafpn": ("YOLOXPAFPN", (64, 128, 256)),
            }
            for neck_name, (expected_name, feature_channels) in aliases.items():
                class _NeckPlan:
                    type = neck_name
                    params = {}

                _, neck_spec = build_native_neck(_NeckPlan(), feature_channels=feature_channels)
                self.assertEqual(neck_spec.name, expected_name)

    def test_native_neck_alias_metadata_covers_required_families(self):
        fake_ops = types.ModuleType("torchvision.ops")

        class _FakeFPN:
            def __init__(self, *args, **kwargs):
                pass

        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            import simpledet.native.necks  # noqa: F401

        aliases = {
            "fpn": "FPN",
            "pa_fpn": "PAFPN",
            "nas_fpn": "NASFPN",
            "bi_fpn": "BiFPN",
            "dilated_encoder": "DilatedEncoder",
            "hr_fpn": "HRFPN",
            "ssd_neck": "SSDNeck",
            "yolox_pafpn": "YOLOXPAFPN",
        }
        for alias, expected_name in aliases.items():
            with self.subTest(alias=alias):
                metadata = NECKS.lookup(alias)
                self.assertEqual(metadata.name, expected_name)
                self.assertEqual(metadata.family, "neck")
                self.assertTrue(metadata.required_dependencies)
                self.assertTrue(metadata.tensor_contracts)
                self.assertNotEqual(metadata.validation_status, "unvalidated")

    def test_native_neck_aliases_forward_shapes_with_real_tensors(self):
        torch = require_torch()
        _require_torchvision()

        from simpledet.native.necks import build_native_neck

        cases = [
            ("fpn", [8, 16, 32, 64], ((32, 32), (16, 16), (8, 8), (4, 4)), 12, 5),
            ("pa_fpn", [8, 16, 32, 64], ((32, 32), (16, 16), (8, 8), (4, 4)), 12, 5),
            ("nas_fpn", [8, 16, 32, 64], ((32, 32), (16, 16), (8, 8), (4, 4)), 12, 5),
            ("bi_fpn", [8, 16, 32, 64], ((32, 32), (16, 16), (8, 8), (4, 4)), 12, 5),
            ("dilated_encoder", [32], ((16, 16),), 16, 1),
            ("hr_fpn", [8, 16, 32, 64], ((32, 32), (16, 16), (8, 8), (4, 4)), 12, 5),
            ("ssd_neck", [8, 16, 32], ((32, 32), (16, 16), (8, 8)), 12, 5),
            ("yolox_pafpn", [8, 16, 32], ((32, 32), (16, 16), (8, 8)), 12, 3),
        ]
        for name, in_channels, input_shapes, out_channels, num_outs in cases:
            with self.subTest(name=name):
                detector_spec = build_detector(
                    "retinanet",
                    num_classes=2,
                    encoder=build_custom_encoder(
                        "UnitBackbone",
                        imports=(),
                        feature_channels=in_channels,
                    ),
                    neck=build_neck(
                        name,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        num_outs=num_outs,
                    ),
                )
                plan = compile_native_detector_plan(detector_spec)
                neck, neck_spec = build_native_neck(plan.neck, feature_channels=tuple(in_channels))
                neck.eval()
                features = _make_tensor_features(in_channels, input_shapes)

                with torch.no_grad():
                    outputs = neck(features)

                self.assertEqual(neck_spec.num_outs, num_outs)
                self.assertEqual(len(outputs), num_outs)
                self.assertTrue(all(tuple(output.shape[:2]) == (2, out_channels) for output in outputs))
                self.assertEqual(tuple(outputs[0].shape[-2:]), tuple(input_shapes[0]))

    def test_native_neck_tensor_contract_rejects_level_mismatch(self):
        require_torch()

        from simpledet.native.necks import TensorContractError, build_native_neck

        plan = type(
            "NeckPlan",
            (),
            {"type": "ssd_neck", "params": {"in_channels": [8, 16, 32], "out_channels": 12}},
        )()
        neck, _ = build_native_neck(plan, feature_channels=(8, 16, 32))
        features = _make_tensor_features([8, 16], ((16, 16), (8, 8)))

        with self.assertRaisesRegex(TensorContractError, "tensor contract feature level mismatch"):
            neck(features)

    def test_native_neck_tensor_contract_rejects_channel_mismatch(self):
        require_torch()

        from simpledet.native.necks import TensorContractError, build_native_neck

        plan = type(
            "NeckPlan",
            (),
            {"type": "ssd_neck", "params": {"in_channels": [8, 16], "out_channels": 12}},
        )()
        neck, _ = build_native_neck(plan, feature_channels=(8, 16))
        features = _make_tensor_features([8, 12], ((16, 16), (8, 8)))

        with self.assertRaisesRegex(TensorContractError, "tensor contract channel mismatch"):
            neck(features)

    def test_build_native_head_aliases_normalized_names(self):
        fake_modules = _fake_torch_modules()
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.heads import build_native_head

            for head_name in ("yolov10", "yolo_v4", "yolo5", "fcos", "atss"):
                head, head_spec = build_native_head(
                    type("HeadPlan", (), {"type": head_name, "params": {}})(),
                    out_channels=256,
                    num_classes=2,
                )
                if head_name.startswith("yolo"):
                    self.assertEqual(head_spec.name, "YOLOXHead")
                if head_name == "fcos":
                    self.assertEqual(head_spec.name, "FCOSHead")
                if head_name == "atss":
                    self.assertEqual(head_spec.name, "ATSSHead")

    def test_build_native_neck_and_head_for_retinanet(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRetinaHead:
            def __init__(self, in_channels, num_anchors, num_classes):
                self.in_channels = in_channels
                self.num_anchors = num_anchors
                self.num_classes = num_classes

            def __call__(self, features):
                return ("logits", len(features))

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_retinanet_module = types.ModuleType("torchvision.models.detection.retinanet")
        fake_retinanet_module.RetinaNetHead = _FakeRetinaHead
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
                "torchvision.models.detection.retinanet": fake_retinanet_module,
            }
        )
        snapshots = _snapshot_native_registries()
        _clear_native_modules()
        _clear_native_registries()
        try:
            with patch.dict(sys.modules, fake_modules):
                from simpledet.native.backbones import build_native_backbone
                from simpledet.native.heads import build_native_head
                from simpledet.native.necks import build_native_neck
                from simpledet.suite import compile_native_detector_plan

                spec = build_detector("retinanet", num_classes=2, encoder="resnet18.a1_in1k")
                plan = compile_native_detector_plan(spec)
                backbone, backbone_spec = build_native_backbone(plan.encoder)
                neck, neck_spec = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
                head, head_spec = build_native_head(plan.head, out_channels=neck_spec.out_channels, num_classes=2)

                module_cls = fake_modules["torch.nn"].Module
                self.assertIsInstance(backbone, module_cls)
                self.assertIsInstance(neck, module_cls)
                self.assertIsInstance(head, module_cls)
                self.assertTrue(type(backbone).__module__.startswith("simpledet.native."))
                self.assertTrue(type(neck).__module__.startswith("simpledet.native."))
                self.assertTrue(type(head).__module__.startswith("simpledet.native."))
                neck_outputs = neck(backbone("tensor"))
                head_outputs = head(neck_outputs)
        finally:
            _restore_native_registries(snapshots)

        self.assertEqual(backbone_spec.feature_channels, (64, 128, 256, 512))
        self.assertEqual(neck_spec.out_channels, 256)
        self.assertEqual(head_spec.name, "RetinaHead")
        self.assertEqual(head_outputs, ("logits", 4))

    def test_build_native_head_for_fcos(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import build_native_backbone
            from simpledet.native.heads import build_native_head
            from simpledet.native.necks import build_native_neck
            from simpledet.suite import compile_native_detector_plan

            spec = build_detector("fcos", num_classes=3, encoder="resnet18.a1_in1k")
            plan = compile_native_detector_plan(spec)
            backbone, backbone_spec = build_native_backbone(plan.encoder)
            neck, neck_spec = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
            head, head_spec = build_native_head(plan.head, out_channels=neck_spec.out_channels, num_classes=3)

        self.assertEqual(head_spec.name, "FCOSHead")
        self.assertEqual(head_spec.in_channels, 256)
        self.assertEqual(head_spec.num_classes, 3)
        self.assertEqual(neck_spec.out_channels, 256)

    def test_build_native_head_for_atss(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import build_native_backbone
            from simpledet.native.heads import build_native_head
            from simpledet.native.necks import build_native_neck
            from simpledet.suite import compile_native_detector_plan

            spec = build_detector("atss", num_classes=3, encoder="resnet18.a1_in1k")
            plan = compile_native_detector_plan(spec)
            backbone, backbone_spec = build_native_backbone(plan.encoder)
            neck, neck_spec = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
            head, head_spec = build_native_head(plan.head, out_channels=neck_spec.out_channels, num_classes=3)

        self.assertEqual(head_spec.name, "ATSSHead")
        self.assertEqual(head_spec.in_channels, 256)
        self.assertEqual(head_spec.num_classes, 3)
        self.assertEqual(head_spec.num_anchors, 9)

    def test_build_native_head_for_gfl(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import build_native_backbone
            from simpledet.native.heads import build_native_head
            from simpledet.native.necks import build_native_neck
            from simpledet.suite import compile_native_detector_plan

            spec = build_detector("gfl", num_classes=3, encoder="resnet18.a1_in1k")
            plan = compile_native_detector_plan(spec)
            backbone, backbone_spec = build_native_backbone(plan.encoder)
            neck, neck_spec = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
            head, head_spec = build_native_head(plan.head, out_channels=neck_spec.out_channels, num_classes=3)

        self.assertEqual(head_spec.name, "GFLHead")
        self.assertEqual(head_spec.in_channels, 256)
        self.assertEqual(head_spec.num_classes, 3)
        self.assertEqual(head_spec.num_anchors, 9)

    def test_build_native_model_composes_backbone_neck_and_head(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRetinaHead:
            def __init__(self, in_channels, num_anchors, num_classes):
                self.in_channels = in_channels
                self.num_anchors = num_anchors
                self.num_classes = num_classes

            def __call__(self, features):
                return {"cls_logits": ["logits"], "bbox_regression": ["bbox"]}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_retinanet_module = types.ModuleType("torchvision.models.detection.retinanet")
        fake_retinanet_module.RetinaNetHead = _FakeRetinaHead
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
                "torchvision.models.detection.retinanet": fake_retinanet_module,
            }
        )
        snapshots = _snapshot_native_registries()
        _clear_native_modules()
        _clear_native_registries()
        try:
            with patch.dict(sys.modules, fake_modules):
                from simpledet.native.modeling import build_native_model

                spec = build_detector("retinanet", num_classes=2, encoder="resnet18.a1_in1k")
                model = build_native_model("retinanet", num_classes=2, detector_spec=spec)
        finally:
            _restore_native_registries(snapshots)

        module_cls = fake_modules["torch.nn"].Module
        self.assertEqual(type(model).__name__, "NativeRetinaNetModel")
        self.assertIsInstance(model, module_cls)
        for component_name in ("backbone", "neck", "head", "loss_fn", "decoder"):
            component = getattr(model, component_name)
            self.assertIsInstance(component, module_cls)
            self.assertTrue(
                type(component).__module__.startswith("simpledet.native."),
                f"{component_name} came from {type(component).__module__}",
            )
        self.assertEqual(model.head_spec.name, "RetinaHead")
        self.assertEqual(model.neck_spec.out_channels, 256)

    def test_build_native_roi_backbone_exposes_named_feature_maps(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.backbones import build_native_backbone
            from simpledet.native.necks import build_native_neck
            from simpledet.native.roi import build_native_roi_backbone
            from simpledet.suite import compile_native_detector_plan

            spec = build_detector("faster_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            plan = compile_native_detector_plan(spec)
            backbone, backbone_spec = build_native_backbone(plan.encoder)
            neck, _ = build_native_neck(plan.neck, feature_channels=backbone_spec.feature_channels)
            roi_backbone = build_native_roi_backbone(
                backbone,
                neck,
                num_levels=len(backbone_spec.feature_channels),
            )
            outputs = roi_backbone("tensor")

        self.assertEqual(tuple(outputs.keys()), ("0", "1", "2", "3"))
        self.assertEqual(tuple(outputs.values()), ("p-0", "p-1", "p-2", "p-3"))
        self.assertEqual(roi_backbone.core_spec.featmap_names, ("0", "1", "2", "3"))

    def test_native_roi_model_generates_rpn_anchors_and_selects_candidates(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRoIAlign:
            def __init__(self, featmap_names, output_size, sampling_ratio):
                self.featmap_names = list(featmap_names)
                self.output_size = output_size
                self.sampling_ratio = sampling_ratio

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("faster_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("faster_rcnn", num_classes=2, detector_spec=spec)

        class _FakeFeatureMap:
            def __init__(self, shape):
                self.shape = shape

        feature_maps = {
            "0": _FakeFeatureMap((1, 64, 2, 2)),
            "1": _FakeFeatureMap((1, 64, 2, 2)),
        }
        anchors = model._generate_anchor_proposals(_FakeTensorImage(), feature_maps)

        self.assertEqual(len(anchors), 8)
        self.assertEqual(anchors[0], (0.0, 0.0, 16.0, 16.0))
        self.assertEqual(model._assign_proposal_index(anchors, (0.0, 0.0, 16.0, 16.0)), 0)
        self.assertEqual(model._select_proposal_index([0.2, 0.9, 0.05]), 1)

    def test_native_roi_model_samples_training_proposals_and_separates_losses(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRoIAlign:
            def __init__(self, featmap_names, output_size, sampling_ratio):
                self.featmap_names = list(featmap_names)
                self.output_size = output_size
                self.sampling_ratio = sampling_ratio

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("faster_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("faster_rcnn", num_classes=2, detector_spec=spec)

        anchors = [
            (0.0, 0.0, 16.0, 16.0),
            (8.0, 8.0, 24.0, 24.0),
            (16.0, 16.0, 31.0, 31.0),
        ]
        with patch.object(model, "_first_target_label", return_value=2):
            sampled_indices, sampled_labels = model._sample_training_proposals(
                anchors,
                anchors,
                [0.1, 0.9, 0.2],
                {"boxes": [(8.0, 8.0, 24.0, 24.0)], "labels": [2]},
            )

        self.assertEqual(sampled_indices, [1, 2])
        self.assertEqual(sampled_labels, [2, 0])

        decoded_calls = []
        with patch.object(model, "_decode_boxes", side_effect=lambda proposals, deltas: decoded_calls.append((len(proposals), len(deltas))) or "decoded"), patch.object(
            model,
            "_cross_entropy_loss",
            return_value=3.0,
        ), patch.object(
            model,
            "_smooth_l1_loss",
            return_value=4.0,
        ):
            losses = model._roi_losses(
                sampled_proposals=[
                    (0.0, 0.0, 16.0, 16.0),
                    (8.0, 8.0, 24.0, 24.0),
                    (16.0, 16.0, 31.0, 31.0),
                ],
                sampled_labels=[2, 0, 1],
                class_logits=[0.1, 0.9, 0.2],
                box_deltas=[
                    [0.1, 0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.2, 0.2],
                    [0.3, 0.3, 0.3, 0.3],
                ],
                target_box=(8.0, 8.0, 24.0, 24.0),
            )

        self.assertEqual(decoded_calls, [(2, 2)])
        self.assertEqual(set(losses.keys()), {"loss_roi_classifier", "loss_roi_box_reg"})
        self.assertEqual(losses["loss_roi_classifier"], 3.0)
        self.assertEqual(losses["loss_roi_box_reg"], 4.0)

    def test_native_roi_model_selects_class_specific_regression_deltas(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRoIAlign:
            def __init__(self, featmap_names, output_size, sampling_ratio):
                self.featmap_names = list(featmap_names)
                self.output_size = output_size
                self.sampling_ratio = sampling_ratio

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("faster_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("faster_rcnn", num_classes=2, detector_spec=spec)

        deltas = [
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            [[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0]],
        ]

        selected = model._select_class_specific_box_deltas(deltas, [1, 2], reference=deltas)

        self.assertEqual(selected, [[2.0, 2.0, 2.0, 2.0], [6.0, 6.0, 6.0, 6.0]])

    def test_native_roi_model_postprocesses_multiple_detections_with_class_specific_boxes(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRoIAlign:
            def __init__(self, featmap_names, output_size, sampling_ratio):
                self.featmap_names = list(featmap_names)
                self.output_size = output_size
                self.sampling_ratio = sampling_ratio

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("faster_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("faster_rcnn", num_classes=2, detector_spec=spec)

        proposals = [
            (0.0, 0.0, 16.0, 16.0),
            (8.0, 8.0, 24.0, 24.0),
            (16.0, 16.0, 31.0, 31.0),
        ]
        proposal_scores = [0.9, 0.5, 0.8]
        class_logits = [
            [0.1, 0.9, 0.0],
            [0.1, 0.0, 0.8],
            [0.1, 0.7, 0.2],
        ]
        box_deltas = [
            [[0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]],
        ]

        with patch.object(model, "_softmax_logits", side_effect=lambda logits: logits):
            boxes, scores, labels, indices = model._postprocess_detections(
                proposals,
                proposal_scores,
                class_logits,
                box_deltas,
            )

        self.assertEqual(indices, [0, 2, 1])
        self.assertEqual(labels, [1, 1, 2])
        self.assertEqual(len(boxes), 3)
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(boxes[0][0], math.tanh(0.2), places=6)
        self.assertAlmostEqual(boxes[1][0], 16.0 + math.tanh(0.6), places=6)
        self.assertAlmostEqual(boxes[2][0], 8.0 + math.tanh(0.5), places=6)

    def test_build_native_model_builds_fcos(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("fcos", num_classes=3, encoder="resnet18.a1_in1k")
            model = build_native_model("fcos", num_classes=3, detector_spec=spec)

        self.assertEqual(type(model).__name__, "NativeRetinaNetModel")

    def test_build_native_model_builds_atss(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("atss", num_classes=3, encoder="resnet18.a1_in1k")
            model = build_native_model("atss", num_classes=3, detector_spec=spec)

        self.assertEqual(type(model).__name__, "NativeRetinaNetModel")
        self.assertEqual(model.head_spec.name, "ATSSHead")

    def test_build_native_model_builds_gfl(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("gfl", num_classes=3, encoder="resnet18.a1_in1k")
            model = build_native_model("gfl", num_classes=3, detector_spec=spec)

        self.assertEqual(type(model).__name__, "NativeRetinaNetModel")
        self.assertEqual(model.head_spec.name, "GFLHead")

    def test_build_native_model_builds_transformer_detector(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeNeck:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return tuple({str(index): feature for index, feature in enumerate(features)}.values())

        fake_modules = _fake_torch_modules()
        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeNeck
        fake_torchvision = types.ModuleType("torchvision")
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("detr_r50_fpn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("detr", num_classes=2, detector_spec=spec)

        self.assertEqual(type(model).__name__, "NativeDetrModel")

    def test_build_transformer_decoder_rejects_invalid_architecture(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("yolox", num_classes=2, encoder="resnet18.a1_in1k")
            with self.assertRaises(ValueError):
                build_native_model("deformable_detr2", num_classes=2, detector_spec=spec)

    def test_build_native_model_builds_mask_rcnn(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRoIAlign:
            def __init__(self, featmap_names, output_size, sampling_ratio):
                self.featmap_names = list(featmap_names)
                self.output_size = output_size
                self.sampling_ratio = sampling_ratio

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("mask_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("mask_rcnn", num_classes=2, detector_spec=spec)

        self.assertEqual(type(model).__name__, "NativeRoIModel")
        self.assertEqual(type(model.backbone).__name__, "NativeRoIBackbone")
        self.assertTrue(model.with_mask)
        self.assertEqual(model.num_classes, 2)
        self.assertIsNotNone(model.mask_roi_pool)

    def test_build_native_model_builds_faster_rcnn(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRoIAlign:
            def __init__(self, featmap_names, output_size, sampling_ratio):
                self.featmap_names = list(featmap_names)
                self.output_size = output_size
                self.sampling_ratio = sampling_ratio

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("faster_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("faster_rcnn", num_classes=2, detector_spec=spec)

        self.assertEqual(type(model).__name__, "NativeRoIModel")
        self.assertEqual(type(model.backbone).__name__, "NativeRoIBackbone")
        self.assertFalse(model.with_mask)
        self.assertEqual(model.num_classes, 2)
        self.assertIsNotNone(model.box_roi_pool)

    def test_native_model_rejects_non_tensor_like_inputs(self):
        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([64, 128, 256, 512])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        class _FakeRetinaHead:
            def __init__(self, in_channels, num_anchors, num_classes):
                self.in_channels = in_channels
                self.num_anchors = num_anchors
                self.num_classes = num_classes

            def __call__(self, features):
                return {"cls_logits": ["logits"], "bbox_regression": ["bbox"]}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_retinanet_module = types.ModuleType("torchvision.models.detection.retinanet")
        fake_retinanet_module.RetinaNetHead = _FakeRetinaHead
        fake_torchvision = types.ModuleType("torchvision")

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
                "torchvision.models.detection.retinanet": fake_retinanet_module,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_detector("retinanet", num_classes=2, encoder="resnet18.a1_in1k")
            model = build_native_model("retinanet", num_classes=2, detector_spec=spec)

            with self.assertRaises(TypeError):
                model(["not-a-tensor"])

    def test_build_native_model_supports_custom_detector_assembler(self):
        custom_name = f"custom_unit_detector_{next(_COUNTER)}"
        if custom_name not in DETECTORS.names():
            @DETECTORS.register(custom_name)
            def _assemble_custom_detector(components, *, num_classes):
                return {
                    "architecture": custom_name,
                    "components": components,
                    "num_classes": num_classes,
                }

        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = lambda *args, **kwargs: _FakeEncoder([32, 64, 128, 256])

        class _FakeFPN:
            def __init__(self, in_channels_list, out_channels):
                self.in_channels_list = list(in_channels_list)
                self.out_channels = out_channels

            def __call__(self, features):
                return {key: f"p-{key}" for key in features}

        fake_ops = types.ModuleType("torchvision.ops")
        fake_ops.FeaturePyramidNetwork = _FakeFPN
        fake_torchvision = types.ModuleType("torchvision")
        fake_retinanet_module = types.ModuleType("torchvision.models.detection.retinanet")

        class _FakeRetinaHead:
            def __init__(self, in_channels, num_anchors, num_classes):
                self.in_channels = in_channels
                self.num_anchors = num_anchors
                self.num_classes = num_classes

            def __call__(self, features):
                return {"cls_logits": ["logits"], "bbox_regression": ["bbox"]}

        fake_retinanet_module.RetinaNetHead = _FakeRetinaHead

        fake_modules = _fake_torch_modules()
        fake_modules.update(
            {
                "timm": fake_timm,
                "torchvision": fake_torchvision,
                "torchvision.ops": fake_ops,
                "torchvision.models": types.ModuleType("torchvision.models"),
                "torchvision.models.detection": types.ModuleType("torchvision.models.detection"),
                "torchvision.models.detection.retinanet": fake_retinanet_module,
            }
        )
        with patch.dict(sys.modules, fake_modules):
            from simpledet.native.modeling import build_native_model

            spec = build_custom_detector(
                custom_name,
                family="dense",
                num_classes=5,
                encoder="resnet18.a1_in1k",
            )
            model = build_native_model(custom_name, num_classes=5, detector_spec=spec)

        self.assertEqual(model["architecture"], custom_name)
        self.assertEqual(model["num_classes"], 5)
