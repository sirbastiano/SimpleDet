import sys
import types
import unittest
from unittest.mock import patch

from native_tensor_contracts import require_torch

from simpledet.extensions import (
    ASSIGNERS,
    DECODERS,
    DETECTORS,
    ENCODERS,
    HEADS,
    LOSSES,
    NECKS,
    POSTPROCESSORS,
    ComponentMetadata,
    DependencyRequirement,
)


_HEAD_FAMILIES = {"dense", "roi", "transformer"}
_VALIDATED_HEAD_STATUS = "runtime_validated"
_MIN_VALIDATED_HEAD_ALIASES = 31
_NATIVE_REGISTRIES = (
    ASSIGNERS,
    DECODERS,
    DETECTORS,
    ENCODERS,
    HEADS,
    LOSSES,
    NECKS,
    POSTPROCESSORS,
)


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


def _reset_native_registry_cache():
    catalog = sys.modules.get("simpledet.suite.catalog")
    if catalog is not None:
        catalog._NATIVE_REGISTRY_IMPORTED = False
        catalog._NATIVE_REGISTRY_IMPORT_ERROR = None


def _fake_torch_modules():
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_functional = types.ModuleType("torch.nn.functional")

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

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)
            return value

    class Conv2d(Module):
        pass

    class Identity(Module):
        def forward(self, value):
            return value

    class Linear(Module):
        pass

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
        def forward(self, value):
            return value

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
    fake_torch.float32 = "float32"
    fake_torch.long = "long"
    fake_torch.as_tensor = lambda value, **_kwargs: value
    fake_torch.is_tensor = lambda _value: False
    return {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_functional,
    }


def _fake_torchvision_modules(fake_torch):
    fake_ops = types.ModuleType("torchvision.ops")

    class _FakeRoIAlign(fake_torch.nn.Module):
        pass

    class _FakeRetinaNetHead(fake_torch.nn.Module):
        def __init__(self, in_channels, num_anchors, num_classes):
            super().__init__()
            self.in_channels = in_channels
            self.num_anchors = num_anchors
            self.num_classes = num_classes

    fake_ops.MultiScaleRoIAlign = _FakeRoIAlign
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.ops = fake_ops
    fake_models = types.ModuleType("torchvision.models")
    fake_detection = types.ModuleType("torchvision.models.detection")
    fake_retinanet = types.ModuleType("torchvision.models.detection.retinanet")
    fake_retinanet.RetinaNetHead = _FakeRetinaNetHead
    return {
        "torchvision": fake_torchvision,
        "torchvision.ops": fake_ops,
        "torchvision.models": fake_models,
        "torchvision.models.detection": fake_detection,
        "torchvision.models.detection.retinanet": fake_retinanet,
    }


def _fake_native_runtime():
    fake_modules = _fake_torch_modules()
    fake_modules.update(_fake_torchvision_modules(fake_modules["torch"]))
    return fake_modules


def _with_fresh_fake_native_runtime():
    snapshots = _snapshot_native_registries()
    _clear_native_modules()
    _clear_native_registries()
    _reset_native_registry_cache()
    return snapshots


def _restore_fake_native_runtime(snapshots):
    _clear_native_modules()
    _restore_native_registries(snapshots)
    _reset_native_registry_cache()


def _head_alias_tokens(entries):
    aliases = []
    seen = set()
    for metadata in entries:
        for alias in (metadata.name, *metadata.aliases):
            if alias in seen:
                continue
            seen.add(alias)
            aliases.append(alias)
    return aliases


def _head_params(metadata):
    params = {}
    if metadata.family == "dense" and metadata.name not in {"RetinaHead", "FreeAnchorRetinaHead"}:
        params["num_convs"] = 1
    if metadata.name == "RepPointsHead":
        params["point_strides"] = (8, 16)
    if metadata.name in {
        "Shared2FCBBoxHead",
        "ConvFCBBoxHead",
        "DoubleConvFCBBoxHead",
        "DynamicBBoxHead",
        "CascadeBBoxHead",
        "SABLHead",
        "SparseRoIHead",
    }:
        params.update({"roi_feat_size": 2, "fc_out_channels": 16, "conv_out_channels": 8})
    if metadata.name in {"FCNMaskHead", "CascadeMaskHead"}:
        params.update({"roi_feat_size": 2, "output_size": 4, "conv_out_channels": 8, "num_convs": 1})
    if metadata.name == "GridHead":
        params.update(
            {
                "grid_size": 3,
                "roi_feat_size": 2,
                "output_size": 4,
                "conv_out_channels": 8,
                "num_convs": 1,
            }
        )
    if metadata.family == "transformer":
        params.update(
            {
                "num_queries": 5,
                "hidden_dim": 16,
                "num_heads": 4,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 32,
            }
        )
    if metadata.name in {"DeformableDETRHead", "DINOHead"}:
        params["num_feature_levels"] = 2
    return params


class NativeHeadMatrixTests(unittest.TestCase):
    def test_head_matrix_discovers_and_constructs_every_validated_alias(self):
        fake_modules = _fake_native_runtime()
        snapshots = _with_fresh_fake_native_runtime()
        try:
            with patch.dict(sys.modules, fake_modules):
                from simpledet.native.heads import build_native_head
                from simpledet.suite import list_heads

                public_aliases = []
                for family in sorted(_HEAD_FAMILIES):
                    public_aliases.extend(list_heads(kind=family))
                entries = [
                    metadata
                    for metadata in HEADS.entries()
                    if metadata.family in _HEAD_FAMILIES
                    and metadata.validation_status == _VALIDATED_HEAD_STATUS
                ]
                registry_aliases = _head_alias_tokens(entries)

                self.assertGreaterEqual(len(registry_aliases), _MIN_VALIDATED_HEAD_ALIASES)
                self.assertGreaterEqual(len(public_aliases), _MIN_VALIDATED_HEAD_ALIASES)
                self.assertTrue(set(public_aliases).issubset(set(registry_aliases)))
                self.assertEqual(
                    sorted({metadata.family for metadata in entries}),
                    ["dense", "roi", "transformer"],
                )

                for alias in registry_aliases:
                    with self.subTest(alias=alias):
                        metadata = HEADS.lookup(alias)
                        self._assert_validated_head_metadata(metadata, alias)
                        plan = types.SimpleNamespace(type=alias, params=_head_params(metadata))

                        head, head_spec = build_native_head(
                            plan,
                            out_channels=8,
                            num_classes=3,
                        )

                        self.assertIsInstance(head, fake_modules["torch"].nn.Module)
                        self.assertEqual(head_spec.name, metadata.name)
                        self.assertEqual(head_spec.num_classes, 3)
                        self.assertEqual(head_spec.in_channels, 8)
        finally:
            _restore_fake_native_runtime(snapshots)

    def test_head_matrix_rejects_incomplete_supported_alias_metadata(self):
        cases = (
            (
                "missing_factory_head",
                ComponentMetadata(
                    name="MissingFactoryHead",
                    aliases=("missing_factory_head",),
                    kind="head",
                    factory=None,
                    required_dependencies=(DependencyRequirement("torch", "cpu"),),
                    tensor_contracts=("feature_pyramid",),
                    validation_status=_VALIDATED_HEAD_STATUS,
                    family="dense",
                ),
                "callable factory",
            ),
            (
                "missing_contract_head",
                ComponentMetadata(
                    name="MissingContractHead",
                    aliases=("missing_contract_head",),
                    kind="head",
                    factory=lambda **_kwargs: None,
                    required_dependencies=(DependencyRequirement("torch", "cpu"),),
                    tensor_contracts=(),
                    validation_status=_VALIDATED_HEAD_STATUS,
                    family="dense",
                ),
                "tensor contract",
            ),
            (
                "missing_validation_head",
                ComponentMetadata(
                    name="MissingValidationHead",
                    aliases=("missing_validation_head",),
                    kind="head",
                    factory=lambda **_kwargs: None,
                    required_dependencies=(DependencyRequirement("torch", "cpu"),),
                    tensor_contracts=("feature_pyramid",),
                    validation_status="unvalidated",
                    family="dense",
                ),
                "runtime_validated",
            ),
        )

        for alias, metadata, message in cases:
            with self.subTest(alias=alias):
                with self.assertRaises(AssertionError) as raised:
                    self._assert_validated_head_metadata(metadata, alias)
                self.assertIn(message, str(raised.exception))

    def test_representative_head_kinds_forward_small_cpu_tensors(self):
        torch = require_torch()
        torch.manual_seed(0)

        from simpledet.suite import build_head

        cases = (
            (
                "dense",
                "fcos_head",
                {
                    "in_channels": 8,
                    "num_convs": 1,
                },
                lambda head: head([torch.full((1, 8, 2, 2), 0.25, dtype=torch.float32)]),
                {"cls_logits", "bbox_regression", "centerness"},
            ),
            (
                "roi",
                "shared_2fc_bbox_head",
                {
                    "in_channels": 8,
                    "roi_feat_size": 2,
                    "fc_out_channels": 16,
                },
                lambda head: head(torch.full((2, 8, 2, 2), 0.25, dtype=torch.float32)),
                {"cls_score", "bbox_pred"},
            ),
            (
                "transformer",
                "detr_head",
                {
                    "in_channels": 8,
                    "num_queries": 5,
                    "hidden_dim": 16,
                    "num_heads": 4,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 1,
                    "dim_feedforward": 32,
                },
                lambda head: head([torch.full((1, 8, 2, 2), 0.25, dtype=torch.float32)]),
                {"pred_logits", "pred_boxes", "class_logits", "box_predictions"},
            ),
        )

        for family, alias, params, forward, expected_keys in cases:
            with self.subTest(family=family):
                head = build_head(name=alias, num_classes=3, **params)
                head.eval()

                with torch.no_grad():
                    outputs = forward(head)

                self.assertTrue(expected_keys.issubset(set(outputs)))
                self.assertEqual(HEADS.lookup(alias).family, family)

    def _assert_validated_head_metadata(self, metadata, alias):
        self.assertIn(metadata.family, _HEAD_FAMILIES)
        self.assertTrue(callable(metadata.factory), f"{alias} must have a callable factory")
        self.assertTrue(metadata.tensor_contracts, f"{alias} must declare a tensor contract")
        self.assertEqual(metadata.validation_status, _VALIDATED_HEAD_STATUS)
        self.assertNotRegex(metadata.factory_path.lower(), r"placeholder|stub|todo")


if __name__ == "__main__":
    unittest.main()
