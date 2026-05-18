import fnmatch
import sys
import types
import unittest
from unittest.mock import patch

from simpledet.extensions import HEADS, NECKS
from simpledet._model_resolution import (
    ModelPatchError,
    apply_runtime_model_overrides,
    list_available_encoders,
    list_available_heads,
    list_available_necks,
)


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return list(self._channels)


class _DummyEncoder:
    def __init__(self, channels):
        self.feature_info = _FeatureInfo(channels)


class ModelResolutionTests(unittest.TestCase):
    def _fake_modules(self):
        torch = types.ModuleType("torch")
        torch_nn = types.ModuleType("torch.nn")
        torch_nn_functional = types.ModuleType("torch.nn.functional")
        torch_vision = types.ModuleType("torchvision")

        class Module:
            def __init__(self, *args, **kwargs):
                pass

        class ModuleList(list):
            pass

        class Sequential(Module):
            def __init__(self, *layers):
                self.layers = list(layers)

        class Linear(Module):
            pass

        class Embedding(Module):
            def __init__(self, num_embeddings, embedding_dim):
                self.weight = f"embedding-{num_embeddings}-{embedding_dim}"

        torch_nn.Module = Module
        torch_nn.ModuleList = ModuleList
        torch_nn.Sequential = Sequential
        torch_nn.Conv2d = lambda *args, **kwargs: object()
        torch_nn.Linear = Linear
        torch_nn.Embedding = Embedding
        torch_nn.ReLU = lambda *args, **kwargs: object()
        torch.nn = torch_nn

        timm = types.ModuleType("timm")
        model_names = ["tiny_encoder", "wide_encoder", "headless_encoder"]
        timm.list_models = lambda pattern=None: sorted(
            name for name in model_names if pattern is None or fnmatch.fnmatch(name, pattern)
        )
        timm.create_model = lambda *_args, **_kwargs: _DummyEncoder([16, 32, 64, 128])

        return {
            "torch": torch,
            "torch.nn": torch_nn,
            "torch.nn.functional": torch_nn_functional,
            "timm": timm,
            "torchvision": torch_vision,
        }

    def test_apply_runtime_model_overrides_patches_dense_head_models(self):
        model_cfg = {
            "type": "RetinaNet",
            "backbone": {"type": "ResNet", "depth": 50},
            "neck": {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 256},
            "bbox_head": {"type": "RetinaHead", "in_channels": 64, "feat_channels": 64, "num_classes": 80},
        }

        with patch.dict(sys.modules, self._fake_modules()):
            summary = apply_runtime_model_overrides(
                model_cfg,
                encoder_name="tiny_encoder",
                encoder_pretrained=False,
                encoder_in_chans=4,
                num_classes=3,
                strict=True,
            )

        self.assertEqual(model_cfg["backbone"]["type"], "TimmEncoder")
        self.assertEqual(model_cfg["backbone"]["model_name"], "tiny_encoder")
        self.assertEqual(model_cfg["backbone"]["in_chans"], 4)
        self.assertEqual(model_cfg["neck"]["in_channels"], [16, 32, 64, 128])
        self.assertEqual(model_cfg["bbox_head"]["in_channels"], 256)
        self.assertEqual(model_cfg["bbox_head"]["feat_channels"], 256)
        self.assertEqual(model_cfg["bbox_head"]["num_classes"], 3)
        self.assertTrue(any(item["path"] == "model.neck" for item in summary))

    def test_apply_runtime_model_overrides_patches_two_stage_models(self):
        model_cfg = {
            "type": "CascadeRCNN",
            "backbone": {"type": "ResNet", "depth": 50},
            "neck": {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 192},
            "rpn_head": {"type": "RPNHead", "in_channels": 256},
            "roi_head": {
                "type": "CascadeRoIHead",
                "bbox_head": [
                    {"type": "Shared2FCBBoxHead", "in_channels": 256, "num_classes": 80},
                    {"type": "Shared2FCBBoxHead", "in_channels": 256, "num_classes": 80},
                ],
                "mask_head": {"type": "FCNMaskHead", "in_channels": 256, "num_classes": 80},
            },
        }

        with patch.dict(sys.modules, self._fake_modules()):
            apply_runtime_model_overrides(
                model_cfg,
                encoder_name="wide_encoder",
                encoder_pretrained=True,
                encoder_in_chans=3,
                num_classes=5,
                strict=True,
            )

        self.assertEqual(model_cfg["neck"]["in_channels"], [16, 32, 64, 128])
        self.assertEqual(model_cfg["rpn_head"]["in_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["bbox_head"][0]["in_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["bbox_head"][1]["in_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["bbox_head"][0]["num_classes"], 5)
        self.assertEqual(model_cfg["roi_head"]["mask_head"]["in_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["mask_head"]["num_classes"], 5)

    def test_apply_runtime_model_overrides_fails_on_ambiguous_head_shapes(self):
        model_cfg = {
            "type": "ToyDetector",
            "backbone": {"type": "ResNet", "depth": 18},
            "neck": None,
            "bbox_head": {"type": "ToyHead", "in_channels": [8, 16, 32]},
        }

        with patch.dict(sys.modules, self._fake_modules()):
            with self.assertRaises(ModelPatchError) as context:
                apply_runtime_model_overrides(
                    model_cfg,
                    encoder_name="headless_encoder",
                    encoder_pretrained=False,
                    encoder_in_chans=1,
                    num_classes=2,
                    strict=True,
                )

        self.assertIn("Cannot safely patch", str(context.exception))

    def test_runtime_list_helpers_use_registry_and_timm_catalogs(self):
        class FCOSHead:
            pass

        class FPN:
            pass

        head_items = dict(HEADS._items)
        neck_items = dict(NECKS._items)
        try:
            HEADS._items = {"FCOSHead": FCOSHead}
            NECKS._items = {"FPN": FPN}

            with patch.dict(sys.modules, self._fake_modules()), patch(
                "simpledet._model_resolution._load_native_component_registries"
            ):
                self.assertEqual(list_available_encoders(pattern="tiny*"), ["tiny_encoder"])
                self.assertEqual(list_available_necks(), ["FPN"])
                self.assertEqual(list_available_heads(), ["FCOSHead"])
        finally:
            HEADS._items = head_items
            NECKS._items = neck_items

    def test_runtime_list_helpers_load_native_registries_without_fallback_lists(self):
        head_items = dict(HEADS._items)
        head_metadata = dict(HEADS._metadata)
        neck_items = dict(NECKS._items)
        neck_metadata = dict(NECKS._metadata)
        try:
            HEADS._items = {}
            HEADS._metadata = {}
            NECKS._items = {}
            NECKS._metadata = {}
            for module_name in list(sys.modules):
                if module_name == "simpledet.native" or module_name.startswith("simpledet.native."):
                    sys.modules.pop(module_name, None)

            with patch.dict(sys.modules, self._fake_modules()):
                self.assertIn("RetinaHead", list_available_heads())
                self.assertIn("FPN", list_available_necks())

            HEADS._items = {}
            HEADS._metadata = {}
            NECKS._items = {}
            NECKS._metadata = {}
            with patch("simpledet._model_resolution._load_native_component_registries"):
                self.assertEqual(list_available_heads(), [])
                self.assertEqual(list_available_necks(), [])

            HEADS._items = {}
            HEADS._metadata = {}
            NECKS._items = {}
            NECKS._metadata = {}
            with patch("simpledet._model_resolution.import_module", side_effect=ImportError("missing torch")):
                self.assertEqual(list_available_heads(), [])
                self.assertEqual(list_available_necks(), [])
        finally:
            HEADS._items = head_items
            HEADS._metadata = head_metadata
            NECKS._items = neck_items
            NECKS._metadata = neck_metadata



if __name__ == "__main__":
    unittest.main()
