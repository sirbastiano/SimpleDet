import fnmatch
import sys
import types
import unittest
from unittest.mock import patch

from simpledet.suite import (
    build_detector,
    build_encoder,
    build_neck,
    compile_detector_spec,
)


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return list(self._channels)


class _DummyEncoder:
    def __init__(self, channels):
        self.feature_info = _FeatureInfo(channels)


class _Registry:
    def __init__(self, module_dict):
        self.module_dict = module_dict


class SuiteCompilerTests(unittest.TestCase):
    def _fake_modules(self):
        numpy = types.ModuleType("numpy")
        torch = types.ModuleType("torch")
        mmengine = types.ModuleType("mmengine")

        class FPN:
            __module__ = "mmdet.models.necks.fpn"

        class PAFPN:
            __module__ = "mmdet.models.necks.pafpn"

        class ChannelMapper:
            __module__ = "mmdet.models.necks.channel_mapper"

        class StandardRoIHead:
            __module__ = "mmdet.models.roi_heads.standard_roi_head"

        class CascadeRoIHead:
            __module__ = "mmdet.models.roi_heads.cascade_roi_head"

        class FCNMaskHead:
            __module__ = "mmdet.models.roi_heads.mask_heads.fcn_mask_head"

        class RetinaHead:
            __module__ = "mmdet.models.dense_heads.retina_head"

        class DeformableDETRHead:
            __module__ = "mmdet.models.dense_heads.deformable_detr_head"

        registry_module = types.ModuleType("mmdet.registry")
        registry_module.MODELS = _Registry(
            {
                "FPN": FPN,
                "PAFPN": PAFPN,
                "ChannelMapper": ChannelMapper,
                "StandardRoIHead": StandardRoIHead,
                "CascadeRoIHead": CascadeRoIHead,
                "FCNMaskHead": FCNMaskHead,
                "RetinaHead": RetinaHead,
                "DeformableDETRHead": DeformableDETRHead,
            }
        )

        mmdet = types.ModuleType("mmdet")
        mmdet.registry = registry_module

        timm = types.ModuleType("timm")
        model_names = ["tiny_encoder", "wide_encoder", "transformer_encoder"]
        timm.list_models = lambda pattern=None: sorted(
            name for name in model_names if pattern is None or fnmatch.fnmatch(name, pattern)
        )
        timm.create_model = lambda *_args, **_kwargs: _DummyEncoder([16, 32, 64, 128])

        return {
            "numpy": numpy,
            "torch": torch,
            "mmengine": mmengine,
            "mmdet": mmdet,
            "mmdet.registry": registry_module,
            "timm": timm,
        }

    def test_compile_dense_detector_from_suite_spec(self):
        spec = build_detector(
            "retinanet",
            num_classes=4,
            encoder="tiny_encoder",
        )

        with patch.dict(sys.modules, self._fake_modules()):
            model_cfg = compile_detector_spec(spec)

        self.assertEqual(model_cfg["type"], "RetinaNet")
        self.assertEqual(model_cfg["backbone"]["type"], "TimmEncoder")
        self.assertEqual(model_cfg["neck"]["in_channels"], [16, 32, 64, 128])
        self.assertEqual(model_cfg["bbox_head"]["num_classes"], 4)
        self.assertEqual(model_cfg["bbox_head"]["in_channels"], 256)

    def test_compile_roi_detector_patches_extractors_and_heads(self):
        spec = build_detector(
            "mask_rcnn",
            num_classes=5,
            encoder="wide_encoder",
            neck=build_neck(out_channels=192),
        )

        with patch.dict(sys.modules, self._fake_modules()):
            model_cfg = compile_detector_spec(spec)

        self.assertEqual(model_cfg["neck"]["out_channels"], 192)
        self.assertEqual(model_cfg["rpn_head"]["in_channels"], 192)
        self.assertEqual(model_cfg["rpn_head"]["feat_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["bbox_head"]["in_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["bbox_roi_extractor"]["out_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["mask_head"]["in_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["mask_roi_extractor"]["out_channels"], 192)
        self.assertEqual(model_cfg["roi_head"]["mask_head"]["num_classes"], 5)

    def test_compile_transformer_detector_patches_embed_dims(self):
        spec = build_detector(
            "deformable_detr",
            num_classes=6,
            encoder="transformer_encoder",
            neck=build_neck(out_channels=192),
        )

        with patch.dict(sys.modules, self._fake_modules()):
            model_cfg = compile_detector_spec(spec)

        self.assertEqual(model_cfg["backbone"]["type"], "TimmEncoder")
        self.assertEqual(model_cfg["neck"]["in_channels"], [32, 64, 128])
        self.assertEqual(model_cfg["neck"]["out_channels"], 192)
        self.assertEqual(model_cfg["encoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"], 192)
        self.assertEqual(model_cfg["decoder"]["layer_cfg"]["self_attn_cfg"]["embed_dims"], 192)
        self.assertEqual(model_cfg["decoder"]["layer_cfg"]["cross_attn_cfg"]["embed_dims"], 192)
        self.assertEqual(model_cfg["bbox_head"]["num_classes"], 6)
        self.assertEqual(model_cfg["positional_encoding"]["num_feats"], 96)
        self.assertEqual(model_cfg["num_feature_levels"], 4)

    def test_compile_detector_with_direct_backbone_config(self):
        spec = build_detector(
            "retinanet",
            num_classes=3,
            encoder=build_encoder(
                "custom_backbone",
                source="config",
                backbone_cfg={"type": "CustomBackbone", "depth": 7},
                feature_channels=[24, 48, 96, 192],
            ),
        )

        with patch.dict(sys.modules, self._fake_modules()):
            model_cfg = compile_detector_spec(spec)

        self.assertEqual(model_cfg["backbone"]["type"], "CustomBackbone")
        self.assertEqual(model_cfg["backbone"]["depth"], 7)
        self.assertEqual(model_cfg["neck"]["in_channels"], [24, 48, 96, 192])
        self.assertEqual(model_cfg["bbox_head"]["num_classes"], 3)

    def test_pipeline_resolves_detector_spec_into_model_cfg(self):
        spec = build_detector(
            "retinanet",
            num_classes=2,
            encoder="tiny_encoder",
        )

        with patch.dict(sys.modules, self._fake_modules()):
            from simpledet.api import ObjectDetectionPipeline

            model_cfg = ObjectDetectionPipeline._resolve_model_cfg(
                model_cfg=None,
                detector_spec=spec,
            )

        self.assertEqual(model_cfg["type"], "RetinaNet")
        self.assertEqual(model_cfg["backbone"]["type"], "TimmEncoder")
        self.assertEqual(model_cfg["bbox_head"]["num_classes"], 2)


if __name__ == "__main__":
    unittest.main()
