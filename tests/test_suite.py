import unittest

from simpledet.suite import (
    build_head,
    build_custom_detector,
    build_custom_encoder,
    build_detector,
    compile_native_detector_plan,
)


class NativeSuiteTests(unittest.TestCase):
    def test_build_detector_defaults_to_native_v1_architectures(self):
        retina = build_detector("retinanet", num_classes=2, encoder="resnet18.a1_in1k")
        fcos = build_detector("fcos", num_classes=3, encoder="resnet18.a1_in1k")
        atss = build_detector("atss", num_classes=4, encoder="resnet18.a1_in1k")
        gfl = build_detector("gfl", num_classes=5, encoder="resnet18.a1_in1k")
        faster_rcnn = build_detector("faster_rcnn", num_classes=4, encoder="resnet18.a1_in1k")
        mask_rcnn = build_detector("mask_rcnn", num_classes=5, encoder="resnet18.a1_in1k")

        self.assertEqual(retina.family, "dense")
        self.assertEqual(retina.head.name, "RetinaHead")
        self.assertEqual(fcos.family, "dense")
        self.assertEqual(fcos.head.name, "FCOSHead")
        self.assertEqual(atss.family, "dense")
        self.assertEqual(atss.head.name, "ATSSHead")
        self.assertEqual(gfl.family, "dense")
        self.assertEqual(gfl.head.name, "GFLHead")
        self.assertEqual(faster_rcnn.family, "roi")
        self.assertEqual(faster_rcnn.head.name, "Shared2FCBBoxHead")
        self.assertEqual(mask_rcnn.family, "roi")
        self.assertEqual(mask_rcnn.head.name, "Shared2FCBBoxHead")
        self.assertTrue(mask_rcnn.head.with_mask)

    def test_build_detector_rejects_removed_legacy_architectures(self):
        detector = build_detector("detr", num_classes=2, encoder="resnet18.a1_in1k")
        plan = compile_native_detector_plan(detector)

        self.assertEqual(detector.family, "transformer")
        self.assertIsNone(detector.head)
        self.assertEqual(plan.head.type, "DETRHead")

    def test_build_head_rejects_non_positive_num_classes(self):
        with self.assertRaisesRegex(ValueError, "num_classes.*positive integer"):
            build_head("retina_head", num_classes=0)

    def test_build_transformer_name_aliases_normalize_to_transformer_family(self):
        aliases = {
            "detr_r50_fpn": "detr",
            "deformable-detr_r50": "deformable_detr",
            "conditional_detr-4scale": "conditional_detr",
            "dab-detr-r50": "dab_detr",
            "dino4": "dino",
            "deformable-detr-v2": "deformable_detr",
            "detr3d": "detr",
        }
        for architecture, expected_family in aliases.items():
            detector = build_detector(architecture, num_classes=2, encoder="resnet18.a1_in1k")
            self.assertEqual(detector.family, "transformer")
            self.assertEqual(detector.architecture, expected_family)

    def test_build_detector_supports_many_dense_and_roi_architectures(self):
        specs = {
            "vfnet": "VFNetHead",
            "fovea": "FoveaHead",
            "foveabox": "FoveaHead",
            "reppoints": "RepPointsHead",
            "yolof": "YOLOFHead",
            "centernet": "CenterNetHead",
        }
        for architecture, expected_head in specs.items():
            detector = build_detector(architecture, num_classes=3, encoder="resnet18.a1_in1k")
            self.assertEqual(detector.family, "dense")
            self.assertEqual(detector.head.name, expected_head)

        more_architectures = {
            "yolo": "YOLOXHead",
            "yolo4": "YOLOXHead",
            "yolo3": "YOLOXHead",
            "yolo_v3": "YOLOXHead",
            "yolov3": "YOLOXHead",
            "yolov5": "YOLOXHead",
            "yolov6": "YOLOXHead",
            "yolov7": "YOLOXHead",
            "yolov8": "YOLOXHead",
            "yolov9": "YOLOXHead",
            "yolov10": "YOLOXHead",
            "yolox": "YOLOXHead",
            "rtmdet_tiny": "RTMDetHead",
            "rtmdet_s": "RTMDetHead",
            "rtmdet": "RTMDetHead",
            "ssd300": "SSDHead",
            "ssdlite": "SSDHead",
            "efficientdet": "EfficientDetHead",
            "tood": "TOODHead",
            "ssd": "SSDHead",
            "sabl": "ATSSHead",
            "solov2": "FCOSHead",
        }
        for architecture, expected_head in more_architectures.items():
            detector = build_detector(architecture, num_classes=3, encoder="resnet18.a1_in1k")
            self.assertEqual(detector.family, "dense")
            self.assertEqual(detector.head.name, expected_head)

        grid = build_detector("grid_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
        cascade = build_detector("cascade_rcnn", num_classes=2, encoder="resnet18.a1_in1k")
        fast = build_detector("Fast R-CNN", num_classes=2, encoder="resnet18.a1_in1k")
        rpn = build_detector("RPN", num_classes=2, encoder="resnet18.a1_in1k")
        cascade_mask = build_detector("Cascade Mask R-CNN", num_classes=2, encoder="resnet18.a1_in1k")
        libra = build_detector("Libra R-CNN", num_classes=2, encoder="resnet18.a1_in1k")
        double_head = build_detector("Double-Head R-CNN", num_classes=2, encoder="resnet18.a1_in1k")
        dynamic = build_detector("Dynamic R-CNN", num_classes=2, encoder="resnet18.a1_in1k")
        self.assertEqual(grid.family, "roi")
        self.assertEqual(cascade.family, "roi")
        self.assertEqual(grid.head.name, "Shared2FCBBoxHead")
        self.assertEqual(cascade.head.name, "CascadeBBoxHead")
        self.assertEqual(fast.architecture, "fast_rcnn")
        self.assertEqual(fast.head.name, "Shared2FCBBoxHead")
        self.assertEqual(rpn.family, "proposal")
        self.assertEqual(rpn.head.name, "RPNHead")
        self.assertEqual(cascade_mask.head.name, "CascadeBBoxHead")
        self.assertTrue(cascade_mask.head.with_mask)
        self.assertEqual(libra.head.name, "Shared2FCBBoxHead")
        self.assertEqual(double_head.head.name, "DoubleConvFCBBoxHead")
        self.assertEqual(dynamic.head.name, "DynamicBBoxHead")

    def test_build_custom_detector_defaults_mask_rcnn_head_with_masking(self):
        detector = build_custom_detector("mask_rcnn", family="roi", num_classes=3, encoder="resnet18.a1_in1k")
        self.assertEqual(detector.family, "roi")
        self.assertTrue(detector.head.with_mask)

    def test_compile_native_detector_plan_uses_custom_backbone_channels(self):
        spec = build_detector(
            "retinanet",
            num_classes=3,
            encoder=build_custom_encoder(
                "CustomBackbone",
                imports=("pkg.backbones",),
                feature_channels=[24, 48, 96, 192],
                depth=18,
            ),
        )

        plan = compile_native_detector_plan(spec)

        self.assertEqual(plan.encoder.type, "CustomBackbone")
        self.assertEqual(plan.encoder.params["feature_channels"], [24, 48, 96, 192])
        self.assertEqual(plan.head.params["num_classes"], 3)

    def test_build_custom_detector_accepts_explicit_transformer_family(self):
        spec = build_custom_detector("custom-transformer", family="transformer", num_classes=2)

        self.assertEqual(spec.family, "transformer")
