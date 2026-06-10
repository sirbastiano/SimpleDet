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
_VALID_DETECTOR_STATUSES = {"runtime_validated", "compatibility_alias"}
_DETECTOR_FAMILIES = {"dense", "proposal", "roi", "transformer"}
_MIN_VALIDATED_DETECTORS = 31


def _clear_native_modules():
    for registry in _NATIVE_REGISTRIES:
        for name, factory in tuple(registry._items.items()):
            if str(getattr(factory, "__module__", "")).startswith("simpledet.native"):
                registry._items.pop(name, None)
                registry._metadata.pop(name, None)
    for module_name in list(sys.modules):
        if module_name == "simpledet.native" or module_name.startswith("simpledet.native."):
            sys.modules.pop(module_name, None)
    simpledet_package = sys.modules.get("simpledet")
    if simpledet_package is not None:
        vars(simpledet_package).pop("native", None)


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

    class Identity(Module):
        def forward(self, x):
            return x

    class Linear(Module):
        def forward(self, x):
            return x

    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.weight = f"embedding-{num_embeddings}-{embedding_dim}"

    class TransformerEncoderLayer(Module):
        pass

    class TransformerDecoderLayer(Module):
        pass

    class TransformerEncoder(Module):
        pass

    class TransformerDecoder(Module):
        pass

    class ReLU(Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

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


def _reset_native_registry_cache():
    catalog = sys.modules.get("simpledet.suite.catalog")
    if catalog is not None:
        catalog._NATIVE_REGISTRY_IMPORTED = False
        catalog._NATIVE_REGISTRY_IMPORT_ERROR = None


def _fake_torchvision_modules(fake_torch):
    fake_ops = types.ModuleType("torchvision.ops")

    class _FakeRoIAlign(fake_torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.ops = fake_ops
    return {
        "torchvision": fake_torchvision,
        "torchvision.ops": fake_ops,
    }


def _detector_alias_tokens(entries):
    aliases = []
    seen = set()
    for metadata in entries:
        for alias in (metadata.name, *metadata.aliases):
            if alias in seen:
                continue
            seen.add(alias)
            aliases.append(alias)
    return aliases


def _tiny_detector_spec(alias):
    return build_detector(
        alias,
        num_classes=2,
        encoder=build_custom_encoder(
            "MatrixTinyBackbone",
            imports=(),
            feature_channels=(8, 8, 8, 8),
            in_channels=3,
        ),
        pretrained=False,
        num_queries=5,
        hidden_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        num_feature_levels=4,
    )


def _tiny_native_components(plan, module_cls):
    module = module_cls()
    feature_spec = types.SimpleNamespace(feature_channels=(8, 8, 8, 8))
    neck_spec = types.SimpleNamespace(name="MatrixTinyNeck", out_channels=8, num_outs=4)

    def _component(component_plan):
        return module_cls() if component_plan is not None else None

    def _spec(component_plan):
        if component_plan is None:
            return None
        return types.SimpleNamespace(
            name=str(component_plan.type),
            num_classes=int(getattr(plan, "num_classes", 2)),
            in_channels=8,
        )

    from simpledet.native.assemblers import NativeModelComponents

    head = _component(plan.head)
    head_spec = _spec(plan.head)
    rpn_head = _component(plan.rpn_head)
    rpn_head_spec = _spec(plan.rpn_head)
    bbox_head = _component(plan.bbox_head)
    bbox_head_spec = _spec(plan.bbox_head)
    mask_head = _component(plan.mask_head)
    mask_head_spec = _spec(plan.mask_head)
    grid_head = _component(plan.grid_head)
    grid_head_spec = _spec(plan.grid_head)
    if plan.family == "proposal":
        head = rpn_head
        head_spec = rpn_head_spec
    return NativeModelComponents(
        plan=plan,
        backbone=module,
        backbone_spec=feature_spec,
        neck=module,
        neck_spec=neck_spec,
        head=head,
        head_spec=head_spec,
        rpn_head=rpn_head,
        rpn_head_spec=rpn_head_spec,
        bbox_head=bbox_head,
        bbox_head_spec=bbox_head_spec,
        mask_head=mask_head,
        mask_head_spec=mask_head_spec,
        grid_head=grid_head,
        grid_head_spec=grid_head_spec,
    )


class NativeDetectorMatrixTests(unittest.TestCase):
    def tearDown(self):
        _clear_native_modules()
        _reset_native_registry_cache()

    def _fake_native_runtime(self):
        fake_modules = _fake_torch_modules()
        fake_modules.update(_fake_torchvision_modules(fake_modules["torch"]))
        return fake_modules

    def _with_fresh_fake_native_runtime(self):
        snapshots = _snapshot_native_registries()
        _clear_native_modules()
        _clear_native_registries()
        _reset_native_registry_cache()
        return snapshots

    def _restore_fake_native_runtime(self, snapshots):
        _clear_native_modules()
        _restore_native_registries(snapshots)
        _reset_native_registry_cache()

    def test_detector_matrix_discovers_at_least_thirty_one_validated_families(self):
        fake_modules = self._fake_native_runtime()
        snapshots = self._with_fresh_fake_native_runtime()
        try:
            with patch.dict(sys.modules, fake_modules):
                from simpledet.extensions import DETECTORS
                from simpledet.suite import list_detectors

                public_aliases = list_detectors()
                entries = [
                    metadata
                    for metadata in DETECTORS.entries()
                    if metadata.family in _DETECTOR_FAMILIES
                    and metadata.validation_status in _VALID_DETECTOR_STATUSES
                ]
                registry_aliases = _detector_alias_tokens(entries)
        finally:
            self._restore_fake_native_runtime(snapshots)

        self.assertGreaterEqual(len(entries), _MIN_VALIDATED_DETECTORS)
        self.assertGreaterEqual(len(registry_aliases), _MIN_VALIDATED_DETECTORS)
        self.assertGreaterEqual(len(public_aliases), _MIN_VALIDATED_DETECTORS)
        self.assertTrue(set(public_aliases).issubset(set(registry_aliases)))
        self.assertIn("Region Proposal Network", public_aliases)
        self.assertEqual(
            sorted({metadata.family for metadata in entries}),
            ["dense", "proposal", "roi", "transformer"],
        )

    def test_every_listed_detector_alias_constructs_spec_and_native_factory(self):
        fake_modules = self._fake_native_runtime()
        module_cls = fake_modules["torch"].nn.Module
        snapshots = self._with_fresh_fake_native_runtime()
        try:
            with patch.dict(sys.modules, fake_modules):
                import simpledet.native.assemblers as assemblers
                from simpledet.extensions import DETECTORS
                from simpledet.native.modeling import (
                    QueryDetector,
                    SingleStageDetector,
                    TwoStageDetector,
                    build_detector as build_native_detector,
                )
                from simpledet.native.roi import RoICoreSpec
                from simpledet.suite import list_detectors

                detector_types = (SingleStageDetector, TwoStageDetector, QueryDetector)

                def _fake_roi_detector(factory_name, components, *, num_classes):
                    return TwoStageDetector(
                        backbone=components.backbone,
                        neck=components.neck,
                        box_roi_pool=module_cls(),
                        num_classes=int(num_classes),
                        in_channels=int(components.neck_spec.out_channels),
                        core_spec=RoICoreSpec.from_num_levels(components.neck_spec.num_outs),
                        rpn_head=components.rpn_head,
                        bbox_head=components.bbox_head,
                        mask_head=components.mask_head,
                        grid_head=components.grid_head,
                        roi_variant=str(factory_name),
                    )

                public_aliases = list_detectors()
                registry_aliases = _detector_alias_tokens(DETECTORS.entries())
                resolved_architectures = set()
                with patch.object(assemblers, "build_native_roi_detector", _fake_roi_detector):
                    for alias in registry_aliases:
                        with self.subTest(alias=alias):
                            metadata = DETECTORS.lookup(alias)
                            spec = _tiny_detector_spec(alias)
                            plan = compile_native_detector_plan(spec)
                            components = _tiny_native_components(plan, module_cls)

                            self.assertIn(plan.family, _DETECTOR_FAMILIES)
                            self.assertTrue(callable(metadata.factory), metadata.factory_path)
                            self.assertNotRegex(
                                metadata.factory_path.lower(),
                                r"placeholder|stub|todo",
                            )

                            factory_model = metadata.factory(components, num_classes=2)
                            self.assertIsInstance(factory_model, detector_types)

                            with patch.object(
                                assemblers,
                                "build_native_components",
                                return_value=components,
                            ):
                                public_model = build_native_detector(
                                    name=alias,
                                    num_classes=2,
                                    detector_spec=spec,
                                    pretrained=False,
                                )
                            self.assertIsInstance(public_model, detector_types)
                            resolved_architectures.add(spec.architecture)

                self.assertTrue(set(public_aliases).issubset(set(registry_aliases)))
                self.assertGreaterEqual(
                    len(resolved_architectures),
                    _MIN_VALIDATED_DETECTORS,
                )
                self.assertIn("rpn", resolved_architectures)
        finally:
            self._restore_fake_native_runtime(snapshots)


class NativeComponentTests(unittest.TestCase):
    def tearDown(self):
        _clear_native_modules()
        _reset_native_registry_cache()

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

        self.assertEqual(type(model).__name__, "QueryDetector")
        self.assertEqual(model.head_spec.name, "DETRHead")

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

        self.assertEqual(type(model).__name__, "TwoStageDetector")
        self.assertEqual(type(model.backbone).__name__, "TimmFeatureBackbone")
        self.assertTrue(model.with_mask)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.rpn_head.__class__.__name__, "RPNHead")
        self.assertEqual(model.bbox_head.__class__.__name__, "Shared2FCBBoxHead")
        self.assertEqual(model.mask_head.__class__.__name__, "FCNMaskHead")
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

        self.assertEqual(type(model).__name__, "TwoStageDetector")
        self.assertEqual(type(model.backbone).__name__, "TimmFeatureBackbone")
        self.assertFalse(model.with_mask)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.rpn_head.__class__.__name__, "RPNHead")
        self.assertEqual(model.bbox_head.__class__.__name__, "Shared2FCBBoxHead")
        self.assertIsNotNone(model.box_roi_pool)

    def test_build_native_model_configures_cascade_and_grid_roi_variants(self):
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

            cascade_spec = build_detector("cascade_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            cascade = build_native_model("cascade_rcnn", num_classes=2, detector_spec=cascade_spec)
            grid_spec = build_detector("grid_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
            grid = build_native_model("grid_rcnn", num_classes=2, detector_spec=grid_spec)

        self.assertEqual(cascade.roi_variant, "cascade_rcnn")
        self.assertEqual(cascade.cascade_num_stages, 3)
        self.assertIsNone(cascade.grid_size)
        self.assertEqual(grid.roi_variant, "grid_rcnn")
        self.assertEqual(grid.grid_size, 7)

        cascade_proposals = []

        def _fake_roi_stage(features, proposals, image_shapes):
            cascade_proposals.append(proposals[0])
            return "class_logits", "box_deltas"

        def _fake_refine(proposals, class_logits, box_deltas, labels, *, image_shape):
            return f"{proposals}-refined"

        cascade._run_roi_box_head = _fake_roi_stage
        cascade._refine_cascade_proposals = _fake_refine

        class_logits, box_deltas, final_proposals = cascade._run_cascade_roi_box_head(
            "features",
            ["p0"],
            [(32, 32)],
            labels=[1],
        )

        self.assertEqual(class_logits, "class_logits")
        self.assertEqual(box_deltas, "box_deltas")
        self.assertEqual(cascade_proposals, ["p0", "p0-refined", "p0-refined-refined"])
        self.assertEqual(final_proposals, "p0-refined-refined")

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
